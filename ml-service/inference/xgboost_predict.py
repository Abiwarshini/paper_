import os
import sys
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import (
    DataPreprocessor, TARGET_CONDITIONS, ALL_FEATURES
)

_PREDICTOR = None


class XGBoostPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
        else:
            model_dir = Path(model_dir)

        prep_path = model_dir / "preprocessor.pkl"
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")
        self.preprocessor = joblib.load(prep_path)

        xgb_path = model_dir / "xgboost" / "xgboost_multilabel.pkl"
        if not xgb_path.exists():
            raise FileNotFoundError(f"XGBoost model artifact not found at {xgb_path}")
        self.model = joblib.load(xgb_path)

        # Calibrated thresholds
        self.thresholds = {}
        metrics_path = model_dir / "xgboost" / "xgboost_metrics.json"
        if metrics_path.exists():
            try:
                with open(metrics_path, "r") as f:
                    data = json.load(f)
                    self.thresholds = data.get("threshold_calibration", {})
            except Exception:
                self.thresholds = {}

        self.feature_name_map = {
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

    def predict(self, input_dict):
        df_trans = self.preprocessor.transform_single_input(input_dict)
        X_in = df_trans[ALL_FEATURES]

        probs_list = [float(est.predict_proba(X_in)[:, 1][0]) for est in self.model.estimators_]

        conditions_results = []
        for cond, prob in zip(TARGET_CONDITIONS, probs_list):
            p_val = float(np.round(prob, 4))
            p_pct = float(np.round(p_val * 100, 1))

            th_data = self.thresholds.get(cond, {})
            cal_th = th_data.get("selected_threshold", 0.50)

            if p_val >= 0.65:
                severity = "HIGH"
            elif p_val >= cal_th:
                severity = "MODERATE"
            else:
                severity = "LOW"

            conditions_results.append({
                "condition": cond,
                "probability": p_val,
                "percentage": p_pct,
                "risk_flag": bool(p_val >= cal_th),
                "severity": severity,
                "threshold_used": cal_th
            })

        # Feature Importance / SHAP for Malnutrition estimator (idx 2)
        mal_idx = TARGET_CONDITIONS.index("Malnutrition") if "Malnutrition" in TARGET_CONDITIONS else 0
        target_est = self.model.estimators_[mal_idx]

        # Use SHAP if available, otherwise tree feature importances
        top_factors = []
        try:
            import shap
            explainer = shap.TreeExplainer(target_est)
            shap_values = explainer.shap_values(X_in)
            if isinstance(shap_values, list):
                sv = np.abs(shap_values[1][0])
            elif len(shap_values.shape) == 2:
                sv = np.abs(shap_values[0])
            else:
                sv = np.abs(shap_values)

            total_shap = np.sum(sv) + 1e-8
            top_indices = np.argsort(sv)[::-1][:5]
            for idx in top_indices:
                feat = ALL_FEATURES[idx]
                pct = round(float((sv[idx] / total_shap) * 100), 1)
                top_factors.append({
                    "feature": self.feature_name_map.get(feat, feat),
                    "raw_feature": feat,
                    "contribution_pct": pct,
                    "impact_level": "High Impact" if pct >= 20 else ("Moderate Impact" if pct >= 10 else "Minor Influence")
                })
        except Exception:
            # Fallback to feature_importances_
            importances = target_est.feature_importances_
            top_indices = np.argsort(importances)[::-1][:5]
            total_imp = np.sum(importances) + 1e-8
            for idx in top_indices:
                feat = ALL_FEATURES[idx]
                pct = round(float((importances[idx] / total_imp) * 100), 1)
                top_factors.append({
                    "feature": self.feature_name_map.get(feat, feat),
                    "raw_feature": feat,
                    "contribution_pct": pct,
                    "impact_level": "High Impact" if pct >= 20 else ("Moderate Impact" if pct >= 10 else "Minor Influence")
                })

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
            "model": "XGBoost",
            "architecture": "Multi-Output Gradient Boosted Decision Trees",
            "overall_risk": overall_risk,
            "prediction": prediction_label,
            "probability": round(overall_p, 4),
            "percentage": round(overall_p * 100, 1),
            "conditions": conditions_results,
            "top_factors": top_factors,
            "explainability_method": "SHAP (TreeExplainer)"
        }


def get_xgboost_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = XGBoostPredictor()
    return _PREDICTOR


def predict_xgboost(input_dict):
    predictor = get_xgboost_predictor()
    return predictor.predict(input_dict)
