from __future__ import annotations

import importlib.util
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_stunting_enhanced.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "dhs_clean.parquet"
SRC_PATH = PROJECT_ROOT / "src" / "02_run_models.py"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stunting" / "evaluation"


def load_run_models_module() -> object:
    spec = importlib.util.spec_from_file_location("run_models", str(SRC_PATH))
    run_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_models)
    return run_models


def load_saved_model() -> tuple[object, object, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found at {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        metadata = pickle.load(f)
    return metadata["model"], metadata["scaler"], metadata["feature_names"]


def load_data(run_models: object) -> tuple[np.ndarray, np.ndarray, list[str]]:
    return run_models.load_data("stunting", "enhanced")


def evaluate_model(model, scaler, X, y) -> dict[str, object]:
    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    sensitivity = recall_score(y, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    results = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": sensitivity,
        "f1_score": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(y, y_pred, digits=4, output_dict=False),
        "y_true": y,
        "y_pred": y_pred,
        "y_prob": y_prob,
    }
    return results


def create_plots(results: dict[str, object], feature_names: list[str], model) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    y_prob = results["y_prob"]

    # Confusion matrix
    cm = np.array(results["confusion_matrix"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Not Stunted (0)", "Stunted (1)"])
    ax.set_yticklabels(["Not Stunted (0)", "Stunted (1)"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black", fontsize=12)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC curve (AUC = {results['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "roc_curve.png", dpi=200)
    plt.close(fig)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label="Precision-Recall curve")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=200)
    plt.close(fig)

    # Feature importance plot if available
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = np.array(model.feature_importances_)
    elif hasattr(model, "get_booster"):
        booster = model.get_booster()
        if booster is not None:
            score = booster.get_score(importance_type="weight")
            importance = np.array([score.get(name, 0.0) for name in feature_names])

    if importance is not None:
        idx = np.argsort(importance)[::-1][:20]
        top_features = [feature_names[i] for i in idx]
        top_values = importance[idx]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(range(len(top_values)), top_values[::-1], color="tab:blue")
        ax.set_yticks(range(len(top_values)))
        ax.set_yticklabels(top_features[::-1])
        ax.set_title("Top 20 Feature Importances")
        ax.set_xlabel("Importance")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
        plt.close(fig)


def save_results(results: dict[str, object]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
        "roc_auc": results["roc_auc"],
        "sensitivity": results["sensitivity"],
        "specificity": results["specificity"],
    }
    pd.DataFrame([metrics]).to_csv(OUTPUT_DIR / "evaluation_metrics.csv", index=False)
    with open(OUTPUT_DIR / "classification_report.txt", "w", encoding="utf-8") as f:
        f.write(results["classification_report"])
    with open(OUTPUT_DIR / "confusion_matrix.json", "w", encoding="utf-8") as f:
        import json

        json.dump(results["confusion_matrix"], f, indent=2)


def main() -> None:
    run_models = load_run_models_module()
    model, scaler, feature_names = load_saved_model()
    X, y, loaded_feature_names = load_data(run_models)

    if feature_names != loaded_feature_names:
        raise ValueError("Saved model feature names do not match the current preprocessing output.")

    results = evaluate_model(model, scaler, X, y)
    save_results(results)
    create_plots(results, feature_names, model)

    print("Evaluation completed.")
    print(pd.DataFrame([{
        "accuracy": results["accuracy"],
        "precision": results["precision"],
        "recall": results["recall"],
        "f1_score": results["f1_score"],
        "roc_auc": results["roc_auc"],
        "sensitivity": results["sensitivity"],
        "specificity": results["specificity"],
    }]).T)
    print("Saved results to:", OUTPUT_DIR)
    print("Plots: confusion_matrix.png, roc_curve.png, precision_recall_curve.png, feature_importance.png")


if __name__ == "__main__":
    main()
