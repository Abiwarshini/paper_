import os
import sys
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import joblib

# Add parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import DataPreprocessor, TARGET_CONDITIONS, ALL_FEATURES
from training.train_dnn import TabularDNN

_MODEL = None
_PREPROCESSOR = None
_METRICS = None


def load_dnn_artifacts():
    global _MODEL, _PREPROCESSOR, _METRICS
    if _MODEL is not None:
        return _MODEL, _PREPROCESSOR, _METRICS

    models_dir = Path(__file__).resolve().parent.parent / "models"
    model_path = models_dir / "dnn" / "best_dnn.pt"
    prep_path = models_dir / "preprocessor.pkl"
    metrics_path = models_dir / "dnn" / "dnn_metrics.json"

    if not model_path.exists():
        raise FileNotFoundError(f"DNN weights not found at {model_path}. Run train_dnn.py first.")
    if not prep_path.exists():
        raise FileNotFoundError(f"Preprocessor not found at {prep_path}.")

    _PREPROCESSOR = joblib.load(prep_path)

    checkpoint = torch.load(model_path, map_location="cpu")
    _MODEL = TabularDNN(
        input_dim=checkpoint["input_dim"],
        num_targets=checkpoint["num_targets"],
        dropout=checkpoint.get("dropout", 0.2)
    )
    _MODEL.load_state_dict(checkpoint["model_state_dict"])
    _MODEL.eval()

    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            _METRICS = json.load(f)
    else:
        _METRICS = {}

    return _MODEL, _PREPROCESSOR, _METRICS


def predict_dnn(input_data):
    """
    Generate Malnutrition predictions, probabilities, and gradient-based explainability using DNN.
    """
    model, preprocessor, metrics = load_dnn_artifacts()
    df_transformed = preprocessor.transform_single_input(input_data)

    x_tensor = torch.tensor(df_transformed[ALL_FEATURES].values.astype(np.float32), dtype=torch.float32, requires_grad=True)

    logits = model(x_tensor)
    probs = torch.sigmoid(logits).detach().numpy().flatten()

    # Compute input gradient attribution for Malnutrition target (index 2)
    mal_idx = TARGET_CONDITIONS.index("Malnutrition") if "Malnutrition" in TARGET_CONDITIONS else 0
    target_logit = logits[0, mal_idx]
    target_logit.backward()

    grads = x_tensor.grad.detach().abs().numpy().flatten()
    top_indices = np.argsort(grads)[::-1][:5]

    total_grad = np.sum(grads) + 1e-8
    top_factors = []
    feature_labels = {
        "child_age_months": "Child Age (months)",
        "birth_weight": "Low Birth Weight",
        "breastfeeding_duration": "Breastfeeding Duration",
        "birth_order": "High Birth Order",
        "mother_bmi": "Maternal BMI Deficit",
        "anc_visits": "Suboptimal Antenatal Visits",
        "hhsize": "Large Household Size",
        "sanitation_risk_index": "Poor Water/Sanitation/Fuel",
        "maternal_risk_score": "Composite Maternal Risk Score",
        "wealth_quintile": "Household Wealth Level",
        "education": "Maternal Education Level",
        "residence": "Rural Geographic Setting",
        "birth_size": "Small Size at Birth",
        "diarrhea_recent": "Recent Diarrhea Illness",
        "fever_recent": "Recent Fever Episode",
        "cough_recent": "Recent Respiratory Symptoms",
        "measles_vaccine": "Unvaccinated for Measles"
    }

    for idx in top_indices:
        feat_name = ALL_FEATURES[idx]
        pct = round(float((grads[idx] / total_grad) * 100), 1)
        top_factors.append({
            "feature": feature_labels.get(feat_name, feat_name),
            "raw_feature": feat_name,
            "contribution_pct": pct,
            "impact_level": "High Impact" if pct >= 20 else ("Moderate Impact" if pct >= 10 else "Minor Influence")
        })

    # Extract conditions
    thresholds = metrics.get("threshold_calibration", {})
    conditions_results = []

    for idx, cond in enumerate(TARGET_CONDITIONS):
        th = thresholds.get(cond, {}).get("selected_threshold", 0.50)
        p = float(probs[idx])
        is_positive = p >= th

        if p >= 0.65:
            severity = "HIGH"
        elif p >= th:
            severity = "MODERATE"
        else:
            severity = "LOW"

        conditions_results.append({
            "condition": cond,
            "probability": round(p, 4),
            "percentage": round(p * 100, 1),
            "risk_flag": bool(is_positive),
            "severity": severity,
            "threshold_used": th
        })

    # Overall Malnutrition Risk
    mal_res = next((c for c in conditions_results if c["condition"] == "Malnutrition"), conditions_results[0])
    overall_p = mal_res["probability"]

    if overall_p >= 0.65:
        overall_risk = "HIGH"
        prediction_label = "Malnourished (High Risk)"
    elif overall_p >= mal_res["threshold_used"]:
        overall_risk = "MODERATE"
        prediction_label = "Malnourished (Moderate Risk)"
    else:
        overall_risk = "LOW"
        prediction_label = "Normal (Well-Nourished)"

    return {
        "model": "DNN (Deep Neural Network)",
        "architecture": "Dense-BatchNorm-ReLU-Dropout",
        "overall_risk": overall_risk,
        "prediction": prediction_label,
        "probability": round(overall_p, 4),
        "percentage": round(overall_p * 100, 1),
        "conditions": conditions_results,
        "top_factors": top_factors,
        "explainability_method": "Input Gradient Attribution"
    }
