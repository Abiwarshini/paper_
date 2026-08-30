"""Train and evaluate the reproducible NFHS-5 wasting TabNet baseline."""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from pytorch_tabnet.tab_model import TabNetClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.10
VALIDATION_SIZE_OF_REMAINDER = 0.50
MAX_EPOCHS = 100
BATCH_SIZE = 1024
VIRTUAL_BATCH_SIZE = 128
LEARNING_RATE = 0.001
TARGET = "wasting"
TABNET_PARAMS = {"n_d": 16, "n_a": 16, "n_steps": 3, "gamma": 1.3, "lambda_sparse": 1e-4}
KNOWN_CATEGORICAL_COLUMNS = {
    "education", "wealth_quintile", "residence", "gender_hh_head", "dist_market_proxy",
    "child_sex", "birth_size", "birth_weight_source", "measles_vaccine", "diarrhea_recent",
    "fever_recent", "cough_recent", "mother_marital_status", "water_source", "toilet_type", "cooking_fuel",
}


def set_reproducible_seeds() -> None:
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    torch.manual_seed(RANDOM_STATE)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_STATE)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(data_path)
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' is missing from {data_path}")
    target = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not target.isin([0, 1]).all():
        raise ValueError("Wasting target must contain only binary values 0 and 1")
    print("\n========================================")
    print("WASTING TARGET INFORMATION")
    print("========================================")
    print(f"Dataset shape: {df.shape}")
    print(f"Not Wasted count       : {(target == 0).sum()}")
    print(f"Wasted count            : {(target == 1).sum()}")
    print(f"Not Wasted percentage   : {(target == 0).mean():.2%}")
    print(f"Wasted percentage       : {(target == 1).mean():.2%}")
    return df.drop(columns=[TARGET]), target


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    identifier_columns = {"caseid", "v001", "v002", "v003"}
    leakage_columns = {"hw70", "haz", "stunting", "wasting", "whz", "hw72"}
    feature_names = [column for column in df.columns if column not in identifier_columns | leakage_columns]
    features = df[feature_names].copy()
    for column in KNOWN_CATEGORICAL_COLUMNS.intersection(features.columns):
        values = features[column].astype("string")
        features[column] = values.astype(object).where(values.notna(), np.nan)
    return features, feature_names


def build_preprocessor(X_train: pd.DataFrame) -> tuple[ColumnTransformer, list[str], list[str]]:
    categorical_columns = [column for column in X_train.columns if column in KNOWN_CATEGORICAL_COLUMNS]
    numeric_columns = [column for column in X_train.columns if column not in categorical_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", Pipeline([( "imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), numeric_columns),
            ("categorical", Pipeline([( "imputer", SimpleImputer(strategy="constant", fill_value="unknown")), ("one_hot", make_one_hot_encoder())]), categorical_columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, numeric_columns, categorical_columns


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    predictions = (probabilities >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }, predictions


def save_attention_masks(model: TabNetClassifier, X: np.ndarray, feature_names: list[str], path: Path) -> None:
    _, masks = model.explain(X)
    steps = sorted(masks)
    mean_masks = np.vstack([masks[step].mean(axis=0) for step in steps])
    plt.figure(figsize=(12, max(5, 0.18 * len(feature_names))))
    plt.imshow(mean_masks.T, aspect="auto", cmap="viridis")
    plt.colorbar(label="Mean attention")
    plt.xticks(range(len(steps)), [f"Step {step + 1}" for step in steps])
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel("Decision step")
    plt.ylabel("Processed feature")
    plt.title("TabNet Wasting - Mean Attention Masks")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def train_tabnet_wasting(data_path: Path, model_dir: Path, results_dir: Path) -> None:
    set_reproducible_seeds()
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Device: {'CUDA' if device == 'cuda' else 'CPU'}")
    start_time = time.time()

    raw_features, target = load_dataset(data_path)
    X, feature_names = select_features(raw_features)
    print(f"\nFinal feature list ({len(feature_names)} raw predictors):")
    print(feature_names)

    X_train, X_temp, y_train, y_temp = train_test_split(X, target, test_size=TEST_SIZE + TEST_SIZE, stratify=target, random_state=RANDOM_STATE)
    X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=VALIDATION_SIZE_OF_REMAINDER, stratify=y_temp, random_state=RANDOM_STATE)
    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_validation_processed = preprocessor.transform(X_validation).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)
    processed_feature_names = preprocessor.get_feature_names_out().tolist()

    print("\n========================================")
    print("DATA SPLIT AND PREPROCESSING")
    print("========================================")
    print(f"Training samples     : {len(X_train)}")
    print(f"Validation samples   : {len(X_validation)}")
    print(f"Test samples         : {len(X_test)}")
    print(f"Numerical features   : {len(numeric_columns)}")
    print(f"Categorical features : {len(categorical_columns)}")
    print(f"Encoded dimensions   : {len(processed_feature_names)}")

    classes = np.array([0, 1])
    weights = compute_class_weight("balanced", classes=classes, y=y_train.to_numpy())
    class_weight = dict(zip(classes.tolist(), weights.tolist()))
    sample_weights = np.zeros(len(y_train), dtype=np.float32)
    for class_idx, weight in class_weight.items():
        sample_weights[y_train.to_numpy() == class_idx] = weight

    model = TabNetClassifier(
        **TABNET_PARAMS,
        optimizer_fn=torch.optim.Adam,
        optimizer_params={"lr": LEARNING_RATE},
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type="sparsemax",
        device_name=device,
        verbose=1,
    )
    model.fit(
        X_train_processed,
        y_train.to_numpy(),
        eval_set=[(X_validation_processed, y_validation.to_numpy())],
        eval_metric=["logloss", "auc"],
        max_epochs=MAX_EPOCHS,
        patience=15,
        batch_size=BATCH_SIZE,
        virtual_batch_size=VIRTUAL_BATCH_SIZE,
        weights=sample_weights,
    )

    best_epoch = model.best_epoch + 1
    probabilities = model.predict_proba(X_test_processed)[:, 1]
    test_metrics, predictions = evaluate_predictions(y_test.to_numpy(), probabilities)
    training_time = time.time() - start_time

    model_path = model_dir / "tabnet_wasting"
    model.save_model(str(model_path))
    with (model_dir / "tabnet_wasting_preprocessor.pkl").open("wb") as file:
        pickle.dump(preprocessor, file)
    pd.DataFrame({"actual_label": y_test.to_numpy(), "predicted_label": predictions, "predicted_probability": probabilities}).to_csv(results_dir / "tabnet_wasting_predictions.csv", index=False)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("TabNet Wasting Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, cm[row, column], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix_tabnet_wasting.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title(f"TabNet Wasting ROC Curve (AUC = {test_metrics['roc_auc']:.4f})")
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve_tabnet_wasting.png", dpi=150)
    plt.close()

    importance = model.feature_importances_
    importance_df = pd.DataFrame({"feature": processed_feature_names, "importance": importance}).sort_values("importance", ascending=False)
    importance_df.to_csv(results_dir / "tabnet_wasting_feature_importance.csv", index=False)
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importance)), importance_df["importance"])
    plt.xticks(range(len(importance)), importance_df["feature"], rotation=90)
    plt.title("TabNet Wasting - Feature Importance")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(results_dir / "tabnet_wasting_feature_importance.png", dpi=150)
    plt.close()
    save_attention_masks(model, X_test_processed[:256], processed_feature_names, results_dir / "tabnet_wasting_attention_masks.png")

    metrics = {
        **test_metrics,
        "best_epoch": int(best_epoch),
        "training_time": float(training_time),
        "training_time_seconds": float(training_time),
        "threshold": 0.5,
        "device": "CUDA" if device == "cuda" else "CPU",
        "feature_names": feature_names,
        "processed_feature_names": processed_feature_names,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "class_weight": class_weight,
        "tabnet_params": TABNET_PARAMS,
        "confusion_matrix": cm.tolist(),
        "validation_best_logloss": float(min(model.history["val_0_logloss"])),
        "validation_best_auc": float(max(model.history["val_0_auc"])),
    }
    (results_dir / "tabnet_wasting_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (results_dir / "tabnet_wasting_classification_report.txt").write_text(classification_report(y_test, predictions, target_names=["Not Wasted", "Wasted"], zero_division=0), encoding="utf-8")

    print("\n========================================")
    print("TABNET WASTING MODEL COMPLETED")
    print("========================================")
    print("\nTarget:")
    print("WHZ <= -2 -> Wasting")
    print("WHZ > -2  -> Not Wasting")
    print("\nDataset:")
    print(f"Total usable samples : {len(X)}")
    print(f"Training samples     : {len(X_train)}")
    print(f"Validation samples   : {len(X_validation)}")
    print(f"Test samples         : {len(X_test)}")
    print("\nFeatures:")
    print(f"Raw predictors       : {len(feature_names)}")
    print(f"Numerical            : {len(numeric_columns)}")
    print(f"Categorical          : {len(categorical_columns)}")
    print(f"Encoded dimensions   : {len(processed_feature_names)}")
    print("\nClass Distribution:")
    print(f"Not Wasted           : {(y_test == 0).sum()}")
    print(f"Wasted               : {(y_test == 1).sum()}")
    print("\nTabNet Configuration:")
    for key, value in TABNET_PARAMS.items():
        print(f"{key:<21}: {value}")
    print(f"learning rate         : {LEARNING_RATE}")
    print(f"Maximum epochs        : {MAX_EPOCHS}")
    print("\nBest Epoch:")
    print(best_epoch)
    print("\nDevice:")
    print("CUDA" if device == "cuda" else "CPU")
    print("\nTEST RESULTS")
    for key in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]:
        print(f"{key.replace('_', ' ').title():<21}: {test_metrics[key]:.4f}")
    print("\nTraining Time:")
    print(f"{training_time / 60:.2f} minutes")
    print("========================================")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=root / "data" / "processed" / "dhs_clean.parquet")
    parser.add_argument("--model-dir", type=Path, default=root / "models")
    parser.add_argument("--results-dir", type=Path, default=root / "results")
    args = parser.parse_args()
    train_tabnet_wasting(args.data, args.model_dir, args.results_dir)


if __name__ == "__main__":
    main()
