import os
import sys
import json
from pathlib import Path
import torch
import numpy as np
import pandas as pd
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import (
    DataPreprocessor, TARGET_CONDITIONS, NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES, ALL_FEATURES
)
from training.train_transformer import FTTransformer

_PREDICTOR = None


class FTTransformerPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = Path(__file__).resolve().parent.parent / "models"
        else:
            model_dir = Path(model_dir)

        prep_path = model_dir / "preprocessor.pkl"
        if not prep_path.exists():
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")
        self.preprocessor = joblib.load(prep_path)

        trans_path = model_dir / "transformer" / "best_transformer.pt"
        if not trans_path.exists():
            raise FileNotFoundError(f"Transformer model artifact not found at {trans_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(str(trans_path), map_location=self.device)

        cardinalities = checkpoint.get("categorical_cardinalities", [2, 4, 5, 2, 5, 2, 2, 2, 2])
        n_targets = checkpoint.get("n_targets", len(TARGET_CONDITIONS))
        d_token = checkpoint.get("d_token", 64)
        n_layers = checkpoint.get("n_layers", 3)
        n_heads = checkpoint.get("n_heads", 4)
        d_ffn = checkpoint.get("d_ffn", 128)

        self.model = FTTransformer(
            num_numerical=len(NUMERICAL_FEATURES),
            categorical_cardinalities=cardinalities,
            d_token=d_token,
            n_layers=n_layers,
            n_heads=n_heads,
            d_ffn=d_ffn,
            dropout=0.1,
            n_targets=n_targets
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.thresholds = {}
        metrics_path = model_dir / "transformer" / "transformer_metrics.json"
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

        x_num = torch.tensor(df_trans[NUMERICAL_FEATURES].values.astype(np.float32), dtype=torch.float32).to(self.device)
        x_cat = torch.tensor(df_trans[CATEGORICAL_FEATURES].values.astype(np.int64), dtype=torch.long).to(self.device)

        x_num.requires_grad_(True)

        logits = self.model(x_num, x_cat)
        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]

        # Gradient attribution for Malnutrition
        mal_idx = TARGET_CONDITIONS.index("Malnutrition") if "Malnutrition" in TARGET_CONDITIONS else 0
        target_logit = logits[0, mal_idx]
        target_logit.backward()

        grads = x_num.grad.detach().cpu().abs().numpy().flatten()
        top_num_indices = np.argsort(grads)[::-1][:5]
        total_grad = np.sum(grads) + 1e-8

        top_factors = []
        for idx in top_num_indices:
            feat_name = NUMERICAL_FEATURES[idx]
            pct = round(float((grads[idx] / total_grad) * 100), 1)
            top_factors.append({
                "feature": self.feature_name_map.get(feat_name, feat_name),
                "raw_feature": feat_name,
                "contribution_pct": pct,
                "impact_level": "High Impact" if pct >= 20 else ("Moderate Impact" if pct >= 10 else "Minor Influence")
            })

        conditions_results = []
        for cond, prob in zip(TARGET_CONDITIONS, probs):
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
            "model": "FT-Transformer",
            "architecture": "Tabular Feature Tokenizer with Self-Attention",
            "overall_risk": overall_risk,
            "prediction": prediction_label,
            "probability": round(overall_p, 4),
            "percentage": round(overall_p * 100, 1),
            "conditions": conditions_results,
            "top_factors": top_factors,
            "explainability_method": "Input Gradient Attribution"
        }


def get_transformer_predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = FTTransformerPredictor()
    return _PREDICTOR


def predict_transformer(input_dict):
    predictor = get_transformer_predictor()
    return predictor.predict(input_dict)
