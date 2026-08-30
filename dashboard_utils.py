"""Dashboard utilities that reuse the existing malnutrition preprocessing and model pipeline."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prediction_pipeline import load_saved_model
from recommendation_engine import generate_recommendations
from risk_score import probability_to_risk_score, risk_level_from_score
from shap_analysis import compute_shap_values, get_sample_top_features

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "dhs_clean.parquet"


def load_run_models_module() -> object:
    module_path = PROJECT_ROOT / "src" / "02_run_models.py"
    spec = importlib.util.spec_from_file_location("run_models", str(module_path))
    run_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(run_models)
    return run_models


def load_stunting_model(mode: str = "enhanced") -> tuple[object, object, list[str]]:
    return load_saved_model("stunting", mode)


def _parse_float(value: Any) -> float | np.nan:
    if value is None:
        return np.nan
    if isinstance(value, str):
        value = value.strip()
        if value == "":
            return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _build_raw_sample(raw_inputs: dict[str, Any]) -> pd.DataFrame:
    sample = pd.DataFrame([raw_inputs])
    if "stunting" not in sample.columns:
        sample["stunting"] = 0
    return sample


def preprocess_raw_input(raw_inputs: dict[str, Any], feature_names: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    run_models = load_run_models_module()
    train_df = pd.read_parquet(DATASET_PATH)
    sample_df = _build_raw_sample(raw_inputs)

    combined = pd.concat([train_df, sample_df], ignore_index=True, sort=False)
    combined = run_models.engineer_features(combined)

    continuous_cols = (
        run_models.CONTINUOUS_STANDARD
        + run_models.CONTINUOUS_EXTRA
        + [
            "is_weaning",
            "birth_order_sq",
            "child_age_months_sq",
            "mother_age_sq",
            "sanitation_risk_index",
        ]
    )
    categorical_cols = (
        run_models.CATEGORICAL_STANDARD
        + run_models.CATEGORICAL_EXTRA
        + ["child_age_group", "mother_bmi_cat"]
    )

    X = combined[continuous_cols + categorical_cols].copy()
    for c in categorical_cols:
        X[c] = X[c].astype(str)
        X.loc[X[c].isna() | (X[c] == "<NA>") | (X[c] == "nan"), c] = "unknown"

    for c in continuous_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[f"{c}_missing"] = X[c].isna().astype(float)
        X[c] = X[c].fillna(X[c].median())

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=False)
    X = X.reindex(columns=feature_names, fill_value=0)

    sample_features = X.iloc[-1].to_dict()
    processed_df = X.iloc[[-1]].astype(float)
    return processed_df, sample_features


def predict_stunting(raw_inputs: dict[str, Any], mode: str = "enhanced") -> dict[str, Any]:
    model, scaler, feature_names = load_stunting_model(mode)
    X_processed, sample_features = preprocess_raw_input(raw_inputs, feature_names)
    X_scaled = scaler.transform(X_processed)

    probability = float(model.predict_proba(X_scaled)[:, 1][0])
    prediction = "Stunted" if probability >= 0.5 else "Not Stunted"
    risk_score = probability_to_risk_score(probability)
    risk_level = risk_level_from_score(risk_score)

    explainer, shap_values = compute_shap_values(model, X_scaled)
    shap_top_features = get_sample_top_features(shap_values, feature_names, sample_index=0, top_n=5)
    recommendations = generate_recommendations(sample_features, shap_top_features, risk_level)

    return {
        "prediction": prediction,
        "probability": probability,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "shap_values": shap_values,
        "feature_names": feature_names,
        "shap_top_features": shap_top_features,
        "recommendations": recommendations,
        "sample_features": sample_features,
        "X_scaled": X_scaled,
        "model": model,
    }


def get_dashboard_field_options() -> dict[str, list[Any]]:
    return {
        "education": [0, 1, 2, 3],
        "wealth_quintile": [1, 2, 3, 4, 5],
        "residence": [1, 2],
        "gender_hh_head": [1, 2, 3],
        "dist_market_proxy": [0, 1, 2],
        "child_sex": [1, 2],
        "birth_size": [1, 2, 3, 4, 5, 8],
        "birth_weight_source": [0, 1, 2, 8],
        "measles_vaccine": [0.0, 1.0, 2.0, 3.0, 4.0, 8.0],
        "diarrhea_recent": [0.0, 2.0, 8.0],
        "fever_recent": [0.0, 1.0, 8.0],
        "cough_recent": [0.0, 2.0, 8.0],
        "mother_marital_status": [0, 1, 3, 4, 5],
        "water_source": [11, 12, 13, 14, 21, 31, 32, 41, 42, 43, 51, 61, 62, 71, 92, 96, 97],
        "toilet_type": [11.0, 12.0, 13.0, 14.0, 15.0, 21.0, 22.0, 23.0, 31.0, 41.0, 44.0, 96.0, 97.0],
        "cooking_fuel": [1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 95, 96, 97],
    }


def get_dashboard_numeric_fields() -> list[str]:
    return [
        "age_hh_head_proxy",
        "hhsize",
        "wealth_score",
        "child_age_months",
        "birth_order",
        "birth_weight",
        "breastfeeding_duration",
        "mother_weight",
        "mother_height",
        "mother_bmi",
        "anc_visits",
    ]
