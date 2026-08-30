"""Train and evaluate the reproducible NFHS-5 stunting TabNet baseline."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight

try:
    from pytorch_tabnet.tab_model import TabNetClassifier
except ImportError:
    raise ImportError(
        "pytorch-tabnet is required. Install with: pip install pytorch-tabnet"
    )

RANDOM_STATE = 42
TEST_SIZE = 0.10
VALIDATION_SIZE_OF_REMAINDER = 0.50
MAX_EPOCHS = 100
BATCH_SIZE = 1024
VIRTUAL_BATCH_SIZE = 128
LEARNING_RATE = 0.001
TARGET = "stunting"

# TabNet hyperparameters
TABNET_PARAMS = {
    "n_d": 16,
    "n_a": 16,
    "n_steps": 3,
    "gamma": 1.3,
    "lambda_sparse": 1e-4,
}

KNOWN_CATEGORICAL_COLUMNS = {
    "education",
    "wealth_quintile",
    "residence",
    "gender_hh_head",
    "dist_market_proxy",
    "child_sex",
    "birth_size",
    "birth_weight_source",
    "measles_vaccine",
    "diarrhea_recent",
    "fever_recent",
    "cough_recent",
    "mother_marital_status",
    "water_source",
    "toilet_type",
    "cooking_fuel",
}


def set_reproducible_seeds() -> None:
    """Set all random seeds for reproducibility."""
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def get_device() -> str:
    """Check available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both older and newer scikit-learn parameter names."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and extract target."""
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset was not found: {data_path}")

    df = pd.read_parquet(data_path)
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' is missing from {data_path}")

    target = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not target.isin([0, 1]).all():
        raise ValueError("Stunting target must contain only binary values 0 and 1")
    
    print(f"\n========================================")
    print(f"DATASET LOADING")
    print(f"========================================")
    print(f"Dataset shape: {df.shape}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {df.shape[1]}")
    print(f"\nStunting class distribution:")
    print(target.value_counts().sort_index().rename(index={0: "not_stunted", 1: "stunted"}).to_string())
    print(f"Class percentages: {((target.value_counts(normalize=True) * 100).sort_index()).round(2).to_dict()}")
    
    return df.drop(columns=[TARGET]), target


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Select features and prevent target leakage."""
    identifier_columns = {"caseid", "v001", "v002", "v003"}
    leakage_columns = {"hw70", "haz", "stunting", "wasting", "whz", "hw72"}
    excluded = identifier_columns | leakage_columns
    feature_names = [column for column in df.columns if column not in excluded]
    if not feature_names:
        raise ValueError("No usable predictor columns remain after leakage filtering")
    features = df[feature_names].copy()
    for column in KNOWN_CATEGORICAL_COLUMNS.intersection(features.columns):
        categorical_values = features[column].astype("string")
        features[column] = categorical_values.astype(object).where(categorical_values.notna(), np.nan)
    return features, feature_names


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    """Build preprocessing pipeline (fit on training data only)."""
    categorical_columns = [
        column for column in X_train.columns if column in KNOWN_CATEGORICAL_COLUMNS
    ]
    numeric_columns = [column for column in X_train.columns if column not in categorical_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("one_hot", make_one_hot_encoder()),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_columns, categorical_columns


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    """Calculate evaluation metrics."""
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }
    return metrics, predictions


def plot_feature_importance(model: TabNetClassifier, feature_names: list[str], output_path: Path) -> None:
    """Plot TabNet feature importance."""
    feature_importance = model.feature_importances_
    indices = np.argsort(feature_importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title("TabNet Stunting - Feature Importance")
    plt.bar(range(len(feature_importance)), feature_importance[indices])
    plt.xticks(range(len(feature_importance)), [feature_names[i] for i in indices], rotation=90)
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_feature_importance_csv(model: TabNetClassifier, feature_names: list[str], output_path: Path) -> None:
    """Save feature importance as CSV."""
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": feature_importance
    }).sort_values("importance", ascending=False)
    importance_df.to_csv(output_path, index=False)


def save_attention_masks(
    model: TabNetClassifier,
    X_test_processed: np.ndarray,
    feature_names: list[str],
    output_path: Path,
) -> None:
    """Save mean attention masks for each decision step."""
    _, masks = model.explain(X_test_processed)
    mean_masks = np.vstack([masks[step].mean(axis=0) for step in sorted(masks)])

    figure_height = max(5, 0.18 * len(feature_names))
    plt.figure(figsize=(12, figure_height))
    plt.imshow(mean_masks.T, aspect="auto", cmap="viridis")
    plt.colorbar(label="Mean attention")
    plt.xticks(range(mean_masks.shape[0]), [f"Step {step + 1}" for step in sorted(masks)])
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel("Decision step")
    plt.ylabel("Processed feature")
    plt.title("TabNet Stunting - Mean Attention Masks")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def train_tabnet_stunting(data_path: Path, model_dir: Path, results_dir: Path) -> None:
    """Train and evaluate TabNet model for stunting prediction."""
    set_reproducible_seeds()
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print(f"Device: {device.upper()}")

    start_time = time.time()

    # Load and preprocess data
    raw_features, target = load_dataset(data_path)
    X, feature_names = select_features(raw_features)
    print(f"\nFinal feature list ({len(feature_names)} features):")
    print(feature_names)

    # Train/validation/test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, target, test_size=TEST_SIZE + TEST_SIZE, stratify=target, random_state=RANDOM_STATE
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=VALIDATION_SIZE_OF_REMAINDER, stratify=y_temp, random_state=RANDOM_STATE
    )

    print(f"\n========================================")
    print(f"DATA SPLIT")
    print(f"========================================")
    print(f"Training samples     : {len(X_train)}")
    print(f"Validation samples   : {len(X_validation)}")
    print(f"Test samples         : {len(X_test)}")
    print(f"Total usable samples : {len(X_train) + len(X_validation) + len(X_test)}")

    # Build preprocessor (fit on training data only)
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_validation_processed = preprocessor.transform(X_validation).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)
    processed_feature_names = preprocessor.get_feature_names_out().tolist()

    print(f"\n========================================")
    print(f"PREPROCESSING")
    print(f"========================================")
    print(f"Number of numerical features: {len(numeric_columns)}")
    print(f"Number of categorical features: {len(categorical_columns)}")
    print(f"Final processed feature dimension: {X_train_processed.shape[1]}")
    print(f"Train shape: {X_train_processed.shape}")
    print(f"Validation shape: {X_validation_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")

    # Calculate class weights
    classes = np.array([0, 1])
    class_weights_array = compute_class_weight("balanced", classes=classes, y=y_train.to_numpy())
    class_weight = dict(zip(classes.tolist(), class_weights_array.tolist()))
    print(f"\nTraining class weights: {class_weight}")

    print(f"\n========================================")
    print(f"TABNET MODEL TRAINING")
    print(f"========================================")
    print(f"TabNet Configuration:")
    for key, value in TABNET_PARAMS.items():
        print(f"  {key}: {value}")
    print(f"  learning_rate: {LEARNING_RATE}")
    print(f"  batch_size: {BATCH_SIZE}")
    print(f"  virtual_batch_size: {VIRTUAL_BATCH_SIZE}")

    # Initialize TabNet model
    model = TabNetClassifier(
        n_d=TABNET_PARAMS["n_d"],
        n_a=TABNET_PARAMS["n_a"],
        n_steps=TABNET_PARAMS["n_steps"],
        gamma=TABNET_PARAMS["gamma"],
        lambda_sparse=TABNET_PARAMS["lambda_sparse"],
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": LEARNING_RATE},
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type="sparsemax",
        device_name=device,
        verbose=1,
    )

    # Convert targets to numpy
    y_train_np = y_train.to_numpy().reshape(-1, 1)
    y_train_np = y_train.to_numpy()
    y_validation_np = y_validation.to_numpy()

    # Apply sample weights for class imbalance
    sample_weights = np.zeros(len(y_train))
    for class_idx, weight in class_weight.items():
        sample_weights[y_train_np == class_idx] = weight

    # Fit model with early stopping
    model.fit(
        X_train_processed,
        y_train_np,
        eval_set=[(X_validation_processed, y_validation_np)],
        eval_metric=["auc"],
        max_epochs=MAX_EPOCHS,
        patience=15,
        batch_size=BATCH_SIZE,
        virtual_batch_size=VIRTUAL_BATCH_SIZE,
        weights=sample_weights,
    )

    training_time = time.time() - start_time
    best_epoch = model.best_epoch + 1  # TabNet uses 0-indexing

    # Get predictions
    probabilities_validation = model.predict_proba(X_validation_processed)[:, 1]
    probabilities_test = model.predict_proba(X_test_processed)[:, 1]
    
    test_metrics, test_predictions = evaluate_predictions(y_test.to_numpy(), probabilities_test)

    print(f"\nTraining completed in {training_time:.2f} seconds ({training_time/60:.2f} minutes)")
    print(f"Best epoch: {best_epoch}")

    # Save model
    model_path = model_dir / "tabnet_stunting"
    model.save_model(str(model_path))
    
    # Save preprocessor
    preprocessor_path = model_dir / "tabnet_stunting_preprocessor.pkl"
    with open(preprocessor_path, "wb") as f:
        pickle.dump(preprocessor, f)

    # Save predictions
    pd.DataFrame({
        "actual_label": y_test.to_numpy(),
        "predicted_label": test_predictions,
        "predicted_probability": probabilities_test,
    }).to_csv(results_dir / "tabnet_stunting_predictions.csv", index=False)

    # Create confusion matrix visualization
    cm = confusion_matrix(y_test, test_predictions)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("TabNet Stunting Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, cm[row, column], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix_tabnet_stunting.png", dpi=150)
    plt.close()

    # Create ROC curve
    RocCurveDisplay.from_predictions(y_test, probabilities_test)
    plt.title(f"TabNet Stunting ROC Curve (AUC = {test_metrics['roc_auc']:.4f})")
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve_tabnet_stunting.png", dpi=150)
    plt.close()

    # Save feature importance
    plot_feature_importance(model, processed_feature_names, results_dir / "tabnet_stunting_feature_importance.png")
    save_feature_importance_csv(model, processed_feature_names, results_dir / "tabnet_stunting_feature_importance.csv")
    save_attention_masks(
        model,
        X_test_processed,
        processed_feature_names,
        results_dir / "tabnet_stunting_attention_masks.png",
    )

    # Save metrics
    with (results_dir / "tabnet_stunting_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_epoch": int(best_epoch),
                "threshold": 0.5,
                "training_time_seconds": training_time,
                "feature_names": feature_names,
                "processed_feature_names": processed_feature_names,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "class_weight": class_weight,
                "tabnet_params": TABNET_PARAMS,
                "test_metrics": test_metrics,
                "confusion_matrix": cm.tolist(),
            },
            file,
            indent=2,
        )

    # Save classification report
    (results_dir / "tabnet_stunting_classification_report.txt").write_text(
        classification_report(y_test, test_predictions, target_names=["Not Stunted", "Stunted"], zero_division=0),
        encoding="utf-8",
    )

    # Print final results
    print(f"\n========================================")
    print(f"TABNET STUNTING MODEL COMPLETED")
    print(f"========================================")
    print(f"\nDataset:")
    print(f"Total samples       : {len(X_train) + len(X_validation) + len(X_test)}")
    print(f"Training samples    : {len(X_train)}")
    print(f"Validation samples  : {len(X_validation)}")
    print(f"Test samples        : {len(X_test)}")
    print(f"\nFeatures:")
    print(f"Total features      : {len(processed_feature_names)}")
    print(f"\nClass Distribution:")
    print(f"Not Stunted         : {(y_test == 0).sum()}")
    print(f"Stunted             : {(y_test == 1).sum()}")
    print(f"\nTabNet Configuration:")
    print(f"n_d                 : {TABNET_PARAMS['n_d']}")
    print(f"n_a                 : {TABNET_PARAMS['n_a']}")
    print(f"n_steps             : {TABNET_PARAMS['n_steps']}")
    print(f"gamma               : {TABNET_PARAMS['gamma']}")
    print(f"learning rate       : {LEARNING_RATE}")
    print(f"maximum epochs      : {MAX_EPOCHS}")
    print(f"\nBest Epoch:")
    print(f"{best_epoch}")
    print(f"\nTEST RESULTS")
    print(f"\nAccuracy            : {test_metrics['accuracy']:.4f}")
    print(f"Precision           : {test_metrics['precision']:.4f}")
    print(f"Recall              : {test_metrics['recall']:.4f}")
    print(f"F1 Score            : {test_metrics['f1_score']:.4f}")
    print(f"ROC-AUC             : {test_metrics['roc_auc']:.4f}")
    print(f"\nTraining Time:")
    print(f"{training_time/60:.2f} minutes")
    print(f"\nModel saved:")
    print(f"{model_path}/")
    print(f"\nPredictions saved:")
    print(f"{results_dir / 'tabnet_stunting_predictions.csv'}")
    print(f"\nMetrics saved:")
    print(f"{results_dir / 'tabnet_stunting_metrics.json'}")
    print(f"========================================")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project_root / "data" / "processed" / "dhs_clean.parquet")
    parser.add_argument("--model-dir", type=Path, default=project_root / "models")
    parser.add_argument("--results-dir", type=Path, default=project_root / "results")
    args = parser.parse_args()
    train_tabnet_stunting(args.data, args.model_dir, args.results_dir)


if __name__ == "__main__":
    main()
