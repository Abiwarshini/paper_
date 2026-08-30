"""Train and evaluate the reproducible NFHS-5 stunting DNN baseline."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
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

RANDOM_STATE = 42
TEST_SIZE = 0.10
VALIDATION_SIZE_OF_REMAINDER = 0.50
MAX_EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 0.001
TARGET = "stunting"
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
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    tf.random.set_seed(RANDOM_STATE)


def make_one_hot_encoder() -> OneHotEncoder:
    """Support both older and newer scikit-learn parameter names."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, pd.Series]:
    if not data_path.exists():
        raise FileNotFoundError(f"Processed dataset was not found: {data_path}")

    df = pd.read_parquet(data_path)
    if TARGET not in df.columns:
        raise KeyError(f"Target column '{TARGET}' is missing from {data_path}")

    target = pd.to_numeric(df[TARGET], errors="raise").astype(int)
    if not target.isin([0, 1]).all():
        raise ValueError("Stunting target must contain only binary values 0 and 1")
    return df.drop(columns=[TARGET]), target


def select_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    # These identifiers identify the survey record/cluster, not child nutrition risk.
    identifier_columns = {"caseid", "v001", "v002", "v003"}
    # HAZ is absent from the processed file, but exclude it if a future dataset adds it.
    leakage_columns = {
        "hw70",
        "haz",
        "stunting",
        "wasting",
        "whz",
        "hw72",
    }
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


def build_model(input_dimension: int) -> tf.keras.Model:
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(input_dimension,)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.30),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ],
        name="dnn_stunting_baseline",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.AUC(name="auc"),
        ],
    )
    return model


def save_training_plot(history: dict[str, list[float]], metric: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(history[metric], label=f"Training {metric.title()}")
    plt.plot(history[f"val_{metric}"], label=f"Validation {metric.title()}")
    plt.xlabel("Epoch")
    plt.ylabel(metric.title())
    plt.title(f"DNN Stunting Training {metric.title()}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
    predictions = (probabilities >= 0.5).astype(int)
    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1_score": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }
    return metrics, predictions


def train_dnn(data_path: Path, model_dir: Path, results_dir: Path) -> None:
    set_reproducible_seeds()
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    raw_features, target = load_dataset(data_path)
    X, feature_names = select_features(raw_features)
    print(f"Dataset shape: {(len(X), len(raw_features) + 1)}")
    print("Target distribution:")
    print(target.value_counts().sort_index().rename(index={0: "not_stunted", 1: "stunted"}).to_string())
    print(f"Class percentages: {((target.value_counts(normalize=True) * 100).sort_index()).round(2).to_dict()}")
    print(f"Final feature list ({len(feature_names)} columns): {feature_names}")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, target, test_size=TEST_SIZE + TEST_SIZE, stratify=target, random_state=RANDOM_STATE
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=VALIDATION_SIZE_OF_REMAINDER,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    preprocessor, numeric_columns, categorical_columns = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train).astype(np.float32)
    X_validation_processed = preprocessor.transform(X_validation).astype(np.float32)
    X_test_processed = preprocessor.transform(X_test).astype(np.float32)
    processed_feature_names = preprocessor.get_feature_names_out().tolist()

    print(f"Number of numerical features: {len(numeric_columns)}")
    print(f"Number of categorical features: {len(categorical_columns)}")
    print(f"Final processed feature dimension: {X_train_processed.shape[1]}")
    print(f"Train shape: {X_train_processed.shape}")
    print(f"Validation shape: {X_validation_processed.shape}")
    print(f"Test shape: {X_test_processed.shape}")

    classes = np.array([0, 1])
    class_weights_array = compute_class_weight("balanced", classes=classes, y=y_train.to_numpy())
    class_weight = dict(zip(classes.tolist(), class_weights_array.tolist()))
    print(f"Training class weights: {class_weight}")

    model = build_model(X_train_processed.shape[1])
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        ),
    ]
    history = model.fit(
        X_train_processed,
        y_train.to_numpy(),
        validation_data=(X_validation_processed, y_validation.to_numpy()),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=1,
    )

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    train_metrics = model.evaluate(
        X_train_processed, y_train.to_numpy(), verbose=0, return_dict=True
    )
    validation_metrics = model.evaluate(
        X_validation_processed, y_validation.to_numpy(), verbose=0, return_dict=True
    )
    probabilities = model.predict(X_test_processed, batch_size=BATCH_SIZE, verbose=0).ravel()
    test_metrics, predictions = evaluate_predictions(y_test.to_numpy(), probabilities)

    model_path = model_dir / "dnn_stunting.keras"
    preprocessor_path = model_dir / "dnn_preprocessor.pkl"
    model.save(model_path)
    pd.to_pickle(preprocessor, preprocessor_path)
    pd.DataFrame(
        {
            "actual_label": y_test.to_numpy(),
            "predicted_label": predictions,
            "predicted_probability": probabilities,
        }
    ).to_csv(results_dir / "dnn_stunting_predictions.csv", index=False)

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("DNN Stunting Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for row in range(2):
        for column in range(2):
            plt.text(column, row, cm[row, column], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(results_dir / "confusion_matrix_dnn_stunting.png", dpi=150)
    plt.close()

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.title(f"DNN Stunting ROC Curve (AUC = {test_metrics['roc_auc']:.4f})")
    plt.tight_layout()
    plt.savefig(results_dir / "roc_curve_dnn_stunting.png", dpi=150)
    plt.close()

    history_dict = {key: [float(value) for value in values] for key, values in history.history.items()}
    save_training_plot(history_dict, "accuracy", results_dir / "training_accuracy_dnn.png")
    save_training_plot(history_dict, "loss", results_dir / "training_loss_dnn.png")

    with (results_dir / "dnn_stunting_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "best_epoch": best_epoch,
                "threshold": 0.5,
                "feature_names": feature_names,
                "processed_feature_names": processed_feature_names,
                "numeric_columns": numeric_columns,
                "categorical_columns": categorical_columns,
                "class_weight": class_weight,
                "train_metrics": {key: float(value) for key, value in train_metrics.items()},
                "validation_metrics": {key: float(value) for key, value in validation_metrics.items()},
                "test_metrics": test_metrics,
                "confusion_matrix": cm.tolist(),
            },
            file,
            indent=2,
        )
    (results_dir / "dnn_stunting_classification_report.txt").write_text(
        classification_report(y_test, predictions, target_names=["Not Stunted", "Stunted"], zero_division=0),
        encoding="utf-8",
    )

    print("\n===== DNN STUNTING RESULTS =====")
    for name, value in test_metrics.items():
        print(f"{name.replace('_', ' ').title():10}: {value:.4f}")
    print(f"Best epoch: {best_epoch}")
    print(f"Saved model: {model_path}")
    print(f"Saved preprocessor: {preprocessor_path}")
    print(f"Saved results: {results_dir}")


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=project_root / "data" / "processed" / "dhs_clean.parquet")
    parser.add_argument("--model-dir", type=Path, default=project_root / "models")
    parser.add_argument("--results-dir", type=Path, default=project_root / "results")
    args = parser.parse_args()
    train_dnn(args.data, args.model_dir, args.results_dir)


if __name__ == "__main__":
    main()