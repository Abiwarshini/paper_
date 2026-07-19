"""Extended explainable AI pipeline for child malnutrition prediction."""

from __future__ import annotations

import argparse
import importlib.util
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from recommendation_engine import generate_recommendations
from risk_score import probability_to_risk_score, risk_level_from_score
from shap_analysis import (
    compute_shap_values,
    get_sample_top_features,
    get_sample_top_features_batch,
    get_top_features,
    save_feature_importance_plot,
    save_shap_summary_plot,
    save_waterfall_plot,
)

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

TARGET_LABEL_MAP = {
    "stunting": {1: "Stunted", 0: "Not Stunted"},
    "wasting": {1: "Wasted", 0: "Not Wasted"},
}


def load_run_models_module() -> object:
    module_path = PROJECT_ROOT / "src" / "02_run_models.py"
    spec = importlib.util.spec_from_file_location("run_models", str(module_path))
    run_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_models)
    return run_models


def _ensure_directory(directory: Path) -> None:
    os.makedirs(directory, exist_ok=True)


def train_final_xgboost(target: str, mode: str = "enhanced") -> tuple[object, RobustScaler, list[str], np.ndarray, np.ndarray]:
    run_models = load_run_models_module()
    X, y, feature_names = run_models.load_data(target, mode)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    smote = SMOTE(random_state=run_models.RANDOM_STATE)
    X_bal, y_bal = smote.fit_resample(X_scaled, y)

    class_weight_dict = {int(c): len(y_bal) / (2 * count) for c, count in zip(*np.unique(y_bal, return_counts=True))}
    scale_pos_weight = class_weight_dict.get(0, 1.0) / class_weight_dict.get(1, 1.0)

    xgb_params = run_models.TUNED_PARAMS["XGBoost"].copy()
    model = XGBClassifier(
        n_estimators=xgb_params["n_estimators"],
        max_depth=xgb_params["max_depth"],
        learning_rate=xgb_params["learning_rate"],
        subsample=xgb_params["subsample"],
        colsample_bytree=xgb_params["colsample_bytree"],
        min_child_weight=xgb_params["min_child_weight"],
        reg_lambda=xgb_params["reg_lambda"],
        reg_alpha=xgb_params["reg_alpha"],
        gamma=xgb_params["gamma"],
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=run_models.RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_bal, y_bal)

    model_path = MODEL_DIR / f"xgboost_{target}_{mode}.pkl"
    with open(model_path, "wb") as artifact:
        pickle.dump({"model": model, "scaler": scaler, "feature_names": feature_names}, artifact)

    print(f"Saved trained XGBoost model for {target} to {model_path}")
    return model, scaler, feature_names, X, y


def load_saved_model(target: str, mode: str = "enhanced") -> tuple[object, RobustScaler, list[str]]:
    model_path = MODEL_DIR / f"xgboost_{target}_{mode}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Saved model not found at {model_path}. Run the pipeline first.")
    with open(model_path, "rb") as artifact:
        metadata = pickle.load(artifact)
    return metadata["model"], metadata["scaler"], metadata["feature_names"]


def build_prediction_frame(
    target: str,
    model: object,
    scaler: RobustScaler,
    X: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    X_scaled = scaler.transform(X)
    probabilities = model.predict_proba(X_scaled)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    sample_df = pd.DataFrame(X, columns=feature_names)
    sample_df["prediction"] = predictions
    sample_df["prediction_label"] = sample_df["prediction"].map(TARGET_LABEL_MAP[target])
    sample_df["predicted_probability"] = probabilities
    sample_df["risk_score"] = sample_df["predicted_probability"].apply(probability_to_risk_score)
    sample_df["risk_level"] = sample_df["risk_score"].apply(risk_level_from_score)

    explainer, shap_values = compute_shap_values(model, X_scaled)
    global_top = get_top_features(shap_values, feature_names, top_n=5)
    print(f"Global top SHAP features for {target}: {global_top}")
    print(f"Computing local SHAP top features and recommendations for {len(sample_df)} samples...")

    top_feature_rows = []
    recommendation_rows = []
    sample_top_features = get_sample_top_features_batch(shap_values, feature_names, top_n=4)
    for idx, top_features in enumerate(sample_top_features):
        if idx % 500 == 0 and idx > 0:
            print(f"  Processed {idx}/{len(sample_df)} samples for recommendations...")
        top_feature_rows.append(", ".join(top_features))
        row_features = sample_df.iloc[idx][feature_names].to_dict()
        recommendation_rows.append(
            "; ".join(generate_recommendations(row_features, top_features, sample_df.at[idx, "risk_level"]))
        )

    sample_df["top_shap_features"] = top_feature_rows
    sample_df["recommendations"] = recommendation_rows
    sample_df["explainability_top_features"] = ", ".join(global_top)
    return sample_df, global_top, shap_values, X_scaled


def save_outputs(target: str, sample_df: pd.DataFrame, global_top: list[str], shap_values, X_scaled: np.ndarray, feature_names: list[str]) -> None:
    out_path = OUTPUT_DIR / target
    _ensure_directory(out_path)

    csv_path = out_path / f"predictions_{target}.csv"
    sample_df.to_csv(csv_path, index=False)
    print(f"Saved predictions CSV for {target} to {csv_path}")

    summary_path = out_path / f"shap_summary_{target}.png"
    print(f"Saving SHAP summary plot for {target}...")
    save_shap_summary_plot(shap_values, X_scaled, feature_names, str(summary_path))
    print(f"Saved SHAP summary plot to {summary_path}")

    importance_path = out_path / f"feature_importance_{target}.png"
    print(f"Saving SHAP feature importance plot for {target}...")
    save_feature_importance_plot(shap_values, feature_names, str(importance_path))
    print(f"Saved feature importance plot to {importance_path}")

    waterfall_sample = int(np.argmax(sample_df["predicted_probability"].values))
    waterfall_path = out_path / f"waterfall_{target}.png"
    print(f"Saving SHAP waterfall plot for sample {waterfall_sample}...")
    save_waterfall_plot(shap_values, feature_names, waterfall_sample, str(waterfall_path))
    print(f"Saved waterfall plot to {waterfall_path}")

    print(f"Saved SHAP summary, feature importance, and waterfall plots for {target} in {out_path}")


def build_combined_output(stunting_df: pd.DataFrame, wasting_df: pd.DataFrame) -> None:
    combined = pd.DataFrame(
        {
            "stunting_prediction": stunting_df["prediction_label"],
            "stunting_probability": stunting_df["predicted_probability"],
            "stunting_risk_score": stunting_df["risk_score"],
            "stunting_risk_level": stunting_df["risk_level"],
            "stunting_top_shap_features": stunting_df["top_shap_features"],
            "stunting_recommendations": stunting_df["recommendations"],
            "wasting_prediction": wasting_df["prediction_label"],
            "wasting_probability": wasting_df["predicted_probability"],
            "wasting_risk_score": wasting_df["risk_score"],
            "wasting_risk_level": wasting_df["risk_level"],
            "wasting_top_shap_features": wasting_df["top_shap_features"],
            "wasting_recommendations": wasting_df["recommendations"],
        }
    )
    combined_path = OUTPUT_DIR / "predictions.csv"
    combined.to_csv(combined_path, index=False)
    print(f"Saved combined health monitoring output to {combined_path}")


def run(targets: list[str], mode: str = "enhanced") -> None:
    _ensure_directory(MODEL_DIR)
    _ensure_directory(OUTPUT_DIR)

    results = {}
    target_dfs = {}
    for target in targets:
        model, scaler, feature_names, X, y = train_final_xgboost(target, mode)
        predictions_df, global_top, shap_values, X_scaled = build_prediction_frame(target, model, scaler, X, feature_names)
        save_outputs(target, predictions_df, global_top, shap_values, X_scaled, feature_names)
        target_dfs[target] = predictions_df
        results[target] = {
            "model_path": str(MODEL_DIR / f"xgboost_{target}_{mode}.pkl"),
            "prediction_count": len(predictions_df),
            "global_top_features": global_top,
        }

    if len(target_dfs) == 2:
        build_combined_output(target_dfs["stunting"], target_dfs["wasting"])

    print("\nExtended explainable AI pipeline completed successfully.")
    for target, meta in results.items():
        print(f"- {target}: {meta['prediction_count']} samples, model saved at {meta['model_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the explainable AI health monitoring pipeline.")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=["stunting", "wasting"],
        default=["stunting", "wasting"],
        help="Targets to process: stunting, wasting.",
    )
    parser.add_argument(
        "--mode",
        choices=["standard", "enhanced"],
        default="enhanced",
        help="Pipeline mode to reuse existing feature engineering and model settings.",
    )
    args = parser.parse_args()
    run(args.targets, args.mode)
