import os
import sys
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import DataPreprocessor, TARGET_CONDITIONS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES

class XGBoostPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(r"e:\Project\SEM7\Malnutrtion", "ml-service", "models")

        prep_path = os.path.join(model_dir, "preprocessor.pkl")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")

        self.preprocessor = DataPreprocessor.load(prep_path)

        xgb_path = os.path.join(model_dir, "xgboost", "xgboost_multilabel.pkl")
        if not os.path.exists(xgb_path):
            raise FileNotFoundError(f"XGBoost model artifact not found at {xgb_path}")

        self.model = joblib.load(xgb_path)

        self.feature_name_map = {
            "weight_kg": "Weight",
            "height_cm": "Height",
            "waz": "Weight-for-age Z-score",
            "haz": "Height-for-age Z-score",
            "whz": "Weight-for-height Z-score",
            "muac_cm": "MUAC (Arm Circumference)",
            "bmi": "Body Mass Index (BMI)",
            "dietary_diversity": "Dietary Diversity",
            "meal_frequency": "Meal Frequency",
            "age_months": "Child Age (months)",
            "water_sanitation_index": "Water & Sanitation Access",
            "gender": "Gender / Sex",
            "breastfeeding_status": "Breastfeeding Status"
        }

    def predict(self, input_dict):
        df_trans = self.preprocessor.transform_single_input(input_dict)
        X_in = df_trans[ALL_FEATURES]

        probs_list = [est.predict_proba(X_in)[:, 1][0] for est in self.model.estimators_]

        predictions = []
        for cond, prob in zip(TARGET_CONDITIONS, probs_list):
            p_val = float(np.round(prob, 4))
            p_pct = float(np.round(p_val * 100, 1))

            if p_val >= 0.60:
                risk_level = "High Risk"
            elif p_val >= 0.30:
                risk_level = "Moderate Risk"
            else:
                risk_level = "Low Risk"

            predictions.append({
                "condition": cond,
                "probability": p_val,
                "percentage": p_pct,
                "risk_level": risk_level
            })

        predictions_sorted = sorted(predictions, key=lambda x: x["probability"], reverse=True)
        top_pred = predictions_sorted[0]["condition"] if predictions_sorted else "Normal"
        top_prob = predictions_sorted[0]["probability"] if predictions_sorted else 0.0

        # Calculate average feature importance across XGBoost estimators
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
                influence = "Strong influence"
            elif imp_val >= 0.08:
                impact_label = "Moderate Impact"
                influence = "Moderate influence"
            else:
                impact_label = "Slight Impact"
                influence = "Slight influence"

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
            "top_prediction": top_pred,
            "top_probability": top_prob,
            "predictions": predictions,
            "top_factors": factors_sorted,
            "medical_disclaimer": "This AI prediction is intended for screening, research, and decision-support purposes only. It is not a medical diagnosis and should not replace evaluation by a qualified healthcare professional."
        }
