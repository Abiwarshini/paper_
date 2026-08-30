from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE
from imblearn.pipeline import Pipeline
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
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_stunting_enhanced.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stunting" / "optimization"
SRC_PATH = PROJECT_ROOT / "src" / "02_run_models.py"

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.85,
    "min_child_weight": 4,
    "reg_lambda": 2.0,
    "reg_alpha": 1.0,
    "gamma": 0.1,
    "scale_pos_weight": 1,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
}

SAMPLERS = {
    "SMOTE": SMOTE(random_state=42),
    "BorderlineSMOTE": BorderlineSMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42),
    "SMOTETomek": SMOTETomek(random_state=42),
}


def load_run_models_module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_models", str(SRC_PATH))
    run_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_models)
    return run_models


def load_saved_model() -> tuple[Any, Any, list[str]]:
    with open(MODEL_PATH, "rb") as f:
        metadata = pickle.load(f)
    return metadata["model"], metadata["scaler"], metadata["feature_names"]


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds=None) -> dict[str, Any]:
    if thresholds is None:
        thresholds = np.arange(0.20, 0.81, 0.01)
    best = None
    summary = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["threshold"] = thresh
        summary.append(metrics)
        if best is None or metrics["f1_score"] > best["f1_score"]:
            best = metrics
    return {"best": best, "summary": summary}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, Any]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if tn + fp > 0 else float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "sensitivity": recall_score(y_true, y_pred, zero_division=0),
        "specificity": specificity,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "classification_report": classification_report(y_true, y_pred, digits=4, zero_division=0),
    }


def evaluate_cv(model: XGBClassifier, sampler, X: np.ndarray, y: np.ndarray, n_splits=5) -> dict[str, Any]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all = []
    y_prob_all = []
    y_pred_all = []
    for train_idx, valid_idx in skf.split(X, y):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_valid_scaled = scaler.transform(X_valid)
        X_res, y_res = sampler.fit_resample(X_train_scaled, y_train)
        model_clone = XGBClassifier(**model.get_params())
        model_clone.fit(X_res, y_res)
        prob = model_clone.predict_proba(X_valid_scaled)[:, 1]
        pred = (prob >= 0.5).astype(int)
        y_true_all.append(y_valid)
        y_prob_all.append(prob)
        y_pred_all.append(pred)
    y_true = np.concatenate(y_true_all)
    y_prob = np.concatenate(y_prob_all)
    y_pred = np.concatenate(y_pred_all)
    return compute_metrics(y_true, y_pred, y_prob)


def run_baseline(X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    model = XGBClassifier(**DEFAULT_PARAMS)
    sampler = SMOTE(random_state=42)
    return evaluate_cv(model, sampler, X, y)


def run_hyperparameter_search(X: np.ndarray, y: np.ndarray, sampler) -> tuple[XGBClassifier, dict[str, Any]]:
    if HAS_OPTUNA:
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))

        def objective(trial: optuna.Trial) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
                "gamma": trial.suggest_float("gamma", 0.0, 0.5),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 3.0),
                "scale_pos_weight": 1,
                "use_label_encoder": False,
                "eval_metric": "logloss",
                "random_state": 42,
                "n_jobs": -1,
            }
            model = XGBClassifier(**params)
            metrics = evaluate_cv(model, sampler, X, y, n_splits=5)
            return metrics["f1_score"]

        study.optimize(objective, n_trials=20, show_progress_bar=True)
        best_params = study.best_params
        best_model = XGBClassifier(
            **{**best_params, "use_label_encoder": False, "eval_metric": "logloss", "random_state": 42, "n_jobs": -1}
        )
        best_model.fit(*sampler.fit_resample(RobustScaler().fit_transform(X), y))
        return best_model, {"best_score": study.best_value, "best_params": best_params}

    pipe = Pipeline([
        ("scaler", RobustScaler()),
        ("sampler", sampler),
        (
            "xgb",
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=-1),
        ),
    ])

    param_dist = {
        "xgb__n_estimators": [100, 150, 200, 250, 300],
        "xgb__max_depth": [4, 6, 8, 10],
        "xgb__learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "xgb__subsample": [0.7, 0.8, 0.9, 1.0],
        "xgb__colsample_bytree": [0.6, 0.7, 0.8, 0.85, 0.9],
        "xgb__min_child_weight": [1, 3, 5, 7],
        "xgb__gamma": [0, 0.1, 0.2, 0.3],
        "xgb__reg_alpha": [0.0, 0.5, 1.0, 2.0],
        "xgb__reg_lambda": [1.0, 2.0, 3.0],
    }

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=20,
        scoring="f1",
        cv=5,
        verbose=1,
        random_state=42,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X, y)
    best_model = search.best_estimator_.named_steps["xgb"]
    best_params = search.best_params_
    results = {
        "best_score": search.best_score_,
        "best_params": best_params,
    }
    return best_model, results


def feature_pruning_experiments(X: np.ndarray, y: np.ndarray, model: XGBClassifier, feature_names: list[str]) -> dict[str, Any]:
    importance = model.feature_importances_
    order = np.argsort(importance)[::-1]
    experiments = {}
    for top_frac in [0.9, 0.8, 0.7, 0.6]:
        n_keep = max(20, int(len(feature_names) * top_frac))
        keep_idx = order[:n_keep]
        X_sub = X[:, keep_idx]
        pruned_model = XGBClassifier(**model.get_params())
        metrics = evaluate_cv(pruned_model, SMOTE(random_state=42), X_sub, y)
        experiments[f"top_{int(top_frac*100)}pct ({n_keep})"] = {
            "metrics": metrics,
            "kept_features": [feature_names[i] for i in keep_idx],
        }
    return experiments


def save_experiment(name: str, results: dict[str, Any], path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with open(path / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def save_model(model: XGBClassifier, scaler: Any, feature_names: list[str], path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_names": feature_names}, f)


def plot_threshold_metrics(summary: list[dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(summary)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics_to_plot = ["accuracy", "precision", "recall", "f1_score"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in metrics_to_plot:
        ax.plot(df["threshold"], df[metric], marker="o", label=metric)
    ax.set_title("Threshold sweep performance")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xticks(df["threshold"])
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path / "threshold_metrics.png", dpi=200)
    plt.close(fig)

    for metric in metrics_to_plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["threshold"], df[metric], marker="o", color="#1f77b4")
        ax.set_title(f"{metric.capitalize()} vs Threshold")
        ax.set_xlabel("Threshold")
        ax.set_ylabel(metric.capitalize())
        ax.set_xticks(df["threshold"])
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.4)
        fig.tight_layout()
        fig.savefig(output_path / f"threshold_{metric}.png", dpi=200)
        plt.close(fig)


def save_threshold_csv(summary: list[dict[str, Any]], output_path: Path) -> None:
    df = pd.DataFrame(summary)
    df = df[["threshold", "accuracy", "precision", "recall", "f1_score", "specificity", "roc_auc"]]
    df.to_csv(output_path / "threshold_comparison.csv", index=False)


def main() -> None:
    run_models = load_run_models_module()
    X, y, feature_names = run_models.load_data("stunting", "enhanced")
    model_saved, scaler_saved, saved_features = load_saved_model()

    if feature_names != saved_features:
        raise ValueError("Saved model feature names do not match current preprocessing.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    X_scaled = scaler_saved.transform(X)
    y_prob_saved = model_saved.predict_proba(X_scaled)[:, 1]

    saved_threshold = 0.5
    saved_metrics = compute_metrics(y, (y_prob_saved >= saved_threshold).astype(int), y_prob_saved)
    threshold_results = threshold_sweep(y, y_prob_saved, thresholds=np.arange(0.20, 0.81, 0.05))
    optimized_threshold = threshold_results["best"]["threshold"]
    optimized_threshold_metrics = threshold_results["best"]

    save_threshold_csv(threshold_results["summary"], OUTPUT_DIR)
    plot_threshold_metrics(threshold_results["summary"], OUTPUT_DIR)

    baseline_cv = run_baseline(X, y)
    sampler_results = {}
    for name, sampler in SAMPLERS.items():
        sampler_results[name] = evaluate_cv(XGBClassifier(**DEFAULT_PARAMS), sampler, X, y)

    best_sampler = max(sampler_results.items(), key=lambda item: item[1]["f1_score"])
    tuned_model, tune_results = run_hyperparameter_search(X, y, SMOTE(random_state=42))
    tuned_model_metrics = evaluate_cv(tuned_model, SMOTE(random_state=42), X, y)

    pruning = feature_pruning_experiments(X, y, tuned_model, feature_names)
    pruning_metrics = {name: data["metrics"] for name, data in pruning.items()}
    best_pruning = max(pruning_metrics.items(), key=lambda item: item[1]["f1_score"])

    feature_importance = tuned_model.feature_importances_
    importance_order = np.argsort(feature_importance)[::-1]
    weak_features = [feature_names[i] for i in importance_order[-20:]]

    experiment_summary = {
        "saved_model_0.5": saved_metrics,
        "saved_model_best_threshold": optimized_threshold_metrics,
        "baseline_cv": baseline_cv,
        "sampler_results": sampler_results,
        "best_sampler": {"name": best_sampler[0], "metrics": best_sampler[1]},
        "hyperparameter_search": tune_results,
        "tuned_model_metrics": tuned_model_metrics,
        "feature_pruning": {k: v["metrics"] for k, v in pruning.items()},
        "best_pruning": {"name": best_pruning[0], "metrics": best_pruning[1]},
        "weak_features": weak_features,
    }

    save_experiment("experiment_summary", experiment_summary, OUTPUT_DIR)
    save_experiment("threshold_scan", {"best": optimized_threshold_metrics, "summary": threshold_results["summary"]}, OUTPUT_DIR)
    save_experiment("sampler_results", sampler_results, OUTPUT_DIR)
    save_experiment("pruning_results", {k: v["metrics"] for k, v in pruning.items()}, OUTPUT_DIR)

    comparison_table = pd.DataFrame([
        {
            "Experiment": "Saved model (0.5)",
            **{k: round(v, 4) for k, v in saved_metrics.items() if k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]},
        },
        {
            "Experiment": f"Saved model (threshold={optimized_threshold:.2f})",
            **{k: round(v, 4) for k, v in optimized_threshold_metrics.items() if k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]},
        },
        {
            "Experiment": "Baseline CV (default params + SMOTE)",
            **{k: round(v, 4) for k, v in baseline_cv.items() if k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]},
        },
        {
            "Experiment": f"Best sampler ({best_sampler[0]})",
            **{k: round(v, 4) for k, v in best_sampler[1].items() if k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]},
        },
        {
            "Experiment": "Hyperparameter tuned + SMOTE",
            **{k: round(v, 4) for k, v in tuned_model_metrics.items() if k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]},
        },
        {
            "Experiment": f"Feature pruning {best_pruning[0]}",
            **{k: round(v, 4) for k, v in best_pruning[1].items() if k in ["accuracy", "precision", "recall", "f1_score", "roc_auc"]},
        },
    ])
    comparison_table.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)

    best_new_model = None
    if tuned_model_metrics["f1_score"] > saved_metrics["f1_score"]:
        best_new_model = tuned_model
        best_model_name = "Hyperparameter tuned + SMOTE"
    elif best_sampler[1]["f1_score"] > saved_metrics["f1_score"]:
        best_new_model = XGBClassifier(**DEFAULT_PARAMS)
        best_model_name = f"Best sampler ({best_sampler[0]})"
    if best_new_model is not None:
        save_model(best_new_model, scaler_saved, feature_names, OUTPUT_DIR / "xgboost_stunting_enhanced_optimized.pkl")

    print("Optimization completed.")
    print(comparison_table.to_string(index=False))
    print("Saved threshold comparison CSV to:", OUTPUT_DIR / "threshold_comparison.csv")
    print("Saved threshold plots to:", OUTPUT_DIR)
    if best_new_model is not None:
        print("Saved better model to", OUTPUT_DIR / "xgboost_stunting_enhanced_optimized.pkl", "(from", best_model_name, ")")
    else:
        print("No new model improved F1 over the current saved model.")


if __name__ == "__main__":
    main()
