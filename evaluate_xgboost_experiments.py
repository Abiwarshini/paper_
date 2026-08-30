from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, BorderlineSMOTE, SMOTE, SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src" / "02_run_models.py"
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_stunting_enhanced.pkl"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "stunting" / "optimization"

SAMPLERS = {
    "SMOTE": SMOTE(random_state=42),
    "BorderlineSMOTE": BorderlineSMOTE(random_state=42),
    "ADASYN": ADASYN(random_state=42),
    "SMOTEENN": SMOTEENN(random_state=42),
    "SMOTETomek": SMOTETomek(random_state=42),
}

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
    "n_jobs": 1,
}


def load_run_models_module() -> Any:
    import importlib.util

    spec = importlib.util.spec_from_file_location("run_models", str(SRC_PATH))
    run_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_models)
    return run_models


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp) if tn + fp > 0 else float("nan")
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "specificity": float(specificity),
    }


def evaluate_cv(model: XGBClassifier, sampler: Any, X: np.ndarray, y: np.ndarray, n_splits=3) -> dict[str, float]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_true_all, y_prob_all, y_pred_all = [], [], []
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


def threshold_sweep(y_true: np.ndarray, y_prob: np.ndarray, thresholds=None) -> dict[str, Any]:
    if thresholds is None:
        thresholds = np.arange(0.20, 0.81, 0.01)
    best = None
    summary = []
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        metrics = compute_metrics(y_true, y_pred, y_prob)
        metrics["threshold"] = float(thresh)
        summary.append(metrics)
        if best is None or metrics["f1_score"] > best["f1_score"]:
            best = metrics
    return {"best": best, "summary": summary}


def run_hyperparameter_search(X: np.ndarray, y: np.ndarray, sampler: Any) -> tuple[XGBClassifier, dict[str, Any]]:
    pipe = ImbPipeline([
        ("scaler", RobustScaler()),
        ("sampler", sampler),
        (
            "xgb",
            XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42, n_jobs=1),
        ),
    ])
    param_dist = {
        "xgb__n_estimators": [100, 150, 200],
        "xgb__max_depth": [4, 6, 8],
        "xgb__learning_rate": [0.01, 0.03, 0.05],
        "xgb__subsample": [0.7, 0.8, 0.9],
        "xgb__colsample_bytree": [0.6, 0.7, 0.8],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__gamma": [0, 0.1, 0.2],
        "xgb__reg_alpha": [0.0, 0.5, 1.0],
        "xgb__reg_lambda": [1.0, 2.0],
    }
    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=8,
        scoring="f1",
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=1,
        refit=True,
    )
    search.fit(X, y)
    best_model = search.best_estimator_.named_steps["xgb"]
    return best_model, {"best_score": float(search.best_score_), "best_params": search.best_params_}


def load_saved_model() -> tuple[XGBClassifier, Any, list[str]]:
    with open(MODEL_PATH, "rb") as f:
        saved = pickle.load(f)
    return saved["model"], saved["scaler"], saved["feature_names"]


def save_model(model: XGBClassifier, scaler: Any, feature_names: list[str], path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "feature_names": feature_names}, f)


def main() -> None:
    run_models = load_run_models_module()
    X, y, feature_names = run_models.load_data("stunting", "enhanced")
    model_saved, scaler_saved, saved_features = load_saved_model()
    if feature_names != saved_features:
        raise ValueError("Saved model features mismatch.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_scaled = scaler_saved.transform(X)
    y_prob_saved = model_saved.predict_proba(X_scaled)[:, 1]
    saved_metrics = compute_metrics(y, (y_prob_saved >= 0.5).astype(int), y_prob_saved)
    threshold_results = threshold_sweep(y, y_prob_saved, thresholds=np.arange(0.20, 0.81, 0.05))
    best_threshold_metrics = threshold_results["best"]

    baseline_metrics = evaluate_cv(XGBClassifier(**DEFAULT_PARAMS), SMOTE(random_state=42), X, y, n_splits=3)

    sampler_metrics = {}
    for name, sampler in SAMPLERS.items():
        sampler_metrics[name] = evaluate_cv(XGBClassifier(**DEFAULT_PARAMS), sampler, X, y, n_splits=3)

    try:
        tuned_model, tuned_info = run_hyperparameter_search(X, y, SMOTE(random_state=42))
        tuned_metrics = evaluate_cv(tuned_model, SMOTE(random_state=42), X, y, n_splits=3)
    except Exception as exc:
        tuned_info = {"error": str(exc)}
        tuned_metrics = None

    if tuned_metrics is not None and tuned_metrics["f1_score"] > saved_metrics["f1_score"]:
        save_model(tuned_model, scaler_saved, feature_names, OUTPUT_DIR / "xgboost_stunting_enhanced_optimized.pkl")
        model_saved_status = "saved_tuned_model"
    else:
        model_saved_status = "no_improvement"

    experiments = [
        {"Experiment": "Saved model (0.5)", **saved_metrics},
        {"Experiment": f"Saved model (threshold={best_threshold_metrics['threshold']:.2f})", **best_threshold_metrics},
        {"Experiment": "Baseline CV (default params + SMOTE)", **baseline_metrics},
    ]
    for name, metrics in sampler_metrics.items():
        experiments.append({"Experiment": f"Sampler {name}", **metrics})
    if tuned_metrics is not None:
        experiments.append({"Experiment": "Hyperparameter tuned + SMOTE", **tuned_metrics})

    df = pd.DataFrame(experiments)
    df.to_csv(OUTPUT_DIR / "comparison_table.csv", index=False)
    with open(OUTPUT_DIR / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "saved_metrics": saved_metrics,
            "best_threshold_metrics": best_threshold_metrics,
            "baseline_metrics": baseline_metrics,
            "sampler_metrics": sampler_metrics,
            "tuned_info": tuned_info,
            "tuned_metrics": tuned_metrics,
            "model_saved_status": model_saved_status,
        }, f, indent=2)

    print("Saved comparison_table.csv and experiment_summary.json to", OUTPUT_DIR)
    if tuned_metrics is not None:
        print("Hyperparameter tuned F1:", tuned_metrics["f1_score"], "vs saved model F1:", saved_metrics["f1_score"])
    else:
        print("Hyperparameter tuning failed")


if __name__ == "__main__":
    main()
