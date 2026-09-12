import os
import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import (
    DataPreprocessor, TARGET_CONDITIONS, NUTRITION_CONDITIONS,
    DISEASE_CONDITIONS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
)


def find_model_dir():
    candidates = [
        Path(__file__).resolve().parent.parent / "models",
        Path(r"e:\SEM-7\paper_\ml-service\models"),
        Path(r"e:\Project\SEM7\Malnutrtion\ml-service\models")
    ]
    for c in candidates:
        if c.exists() and (c / "preprocessor.pkl").exists():
            return c
    return candidates[0]


class XGBoostPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = find_model_dir()
        else:
            model_dir = Path(model_dir)

        prep_path = model_dir / "preprocessor.pkl"
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")
        self.preprocessor = DataPreprocessor.load(str(prep_path))

        xgb_path = model_dir / "xgboost" / "xgboost_multilabel.pkl"
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model artifact not found at {xgb_path}")
        self.model = joblib.load(str(xgb_path))

        # Load calibrated thresholds if available
        self.thresholds = {}
        metrics_path = model_dir / "xgboost" / "xgboost_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                    self.thresholds = data.get("threshold_info", {})
            except Exception:
                self.thresholds = {}

        self.feature_name_map = {
            "weight_kg": "Weight",
            "height_cm": "Height",
            "muac_cm": "MUAC (Mid-Upper Arm Circumference)",
            "bmi": "Body Mass Index (BMI)",
            "dietary_diversity": "Dietary Diversity Score",
            "meal_frequency": "Meal Frequency",
            "age_months": "Child Age (months)",
            "water_sanitation_index": "Water & Sanitation Access",
            "gender": "Sex / Gender",
            "breastfeeding_status": "Breastfeeding Status"
        }

    def predict(self, input_dict):
        df_trans = self.preprocessor.transform_single_input(input_dict)
        X_in = df_trans[ALL_FEATURES]

        probs_list = [est.predict_proba(X_in)[:, 1][0] for est in self.model.estimators_]

        nutrition_assessment = []
        disease_screening = []
        all_predictions = []

        for cond, prob in zip(TARGET_CONDITIONS, probs_list):
            p_val = float(np.round(prob, 4))
            p_pct = float(np.round(p_val * 100, 1))

            # Use calibrated threshold if available, otherwise default
            th_data = self.thresholds.get(cond, {})
            cal_th = th_data.get("selected_threshold", 0.50)

            if p_val >= max(0.60, cal_th):
                risk_level = "High Risk"
            elif p_val >= min(0.30, cal_th * 0.7):
                risk_level = "Moderate Risk"
            else:
                risk_level = "Low Risk"

            item = {
                "condition": cond,
                "probability": p_val,
                "percentage": p_pct,
                "risk_level": risk_level,
                "threshold": cal_th
            }

            all_predictions.append(item)
            if cond in NUTRITION_CONDITIONS:
                nutrition_assessment.append(item)
            else:
                disease_screening.append(item)

        # Primary findings
        nutri_sorted = sorted(nutrition_assessment, key=lambda x: x["probability"], reverse=True)
        top_nutri = nutri_sorted[0]["condition"] if nutri_sorted else "Normal"
        top_nutri_prob = nutri_sorted[0]["probability"] if nutri_sorted else 0.0

        disease_sorted = sorted(disease_screening, key=lambda x: x["probability"], reverse=True)
        top_disease = disease_sorted[0]["condition"] if disease_sorted else "None Detected"
        top_disease_prob = disease_sorted[0]["probability"] if disease_sorted else 0.0

        top_prob = max(top_nutri_prob, top_disease_prob)

        # Feature Importance / Explainability across estimators
        importances = np.zeros(len(ALL_FEATURES))
        for est in self.model.estimators_:
            importances += est.feature_importances_
        importances /= len(self.model.estimators_)
        total_imp = sum(importances) + 1e-6
        norm_imp = importances / total_imp

        factors = []
        for fname, imp in zip(ALL_FEATURES, norm_imp):
            hname = self.feature_name_map.get(fname, fname)
            imp_val = float(np.round(imp, 3))

            if imp_val >= 0.15:
                impact_label = "High Impact"
                influence = "Primary driving factor"
            elif imp_val >= 0.08:
                impact_label = "Moderate Impact"
                influence = "Contributing factor"
            else:
                impact_label = "Slight Impact"
                influence = "Minor factor"

            factors.append({
                "feature": hname,
                "raw_feature": fname,
                "importance": imp_val,
                "impact": impact_label,
                "influence": influence
            })

        factors_sorted = sorted(factors, key=lambda x: x["importance"], reverse=True)[:5]

        return {
            "model": "XGBoost",
            "status": "success",
            "overall_risk": "HIGH" if top_prob >= 0.60 else ("MODERATE" if top_prob >= 0.30 else "LOW"),
            "top_prediction": top_nutri,
            "top_probability": top_nutri_prob,
            "top_disease_risk": top_disease,
            "top_disease_probability": top_disease_prob,
            "nutrition_assessment": nutrition_assessment,
            "disease_screening": disease_screening,
            "predictions": all_predictions,
            "top_factors": factors_sorted,
            "medical_disclaimer": "AI-based risk screening only — this result is for research/academic screening and is not a medical diagnosis. High-risk results should be evaluated by a qualified healthcare professional."
        }
