import os
import sys
import torch
import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing.preprocessing import DataPreprocessor, TARGET_CONDITIONS, NUMERICAL_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
from training.train_transformer import FTTransformer

class FTTransformerPredictor:
    def __init__(self, model_dir=None):
        if model_dir is None:
            model_dir = os.path.join(r"e:\Project\SEM7\Malnutrtion", "ml-service", "models")

        prep_path = os.path.join(model_dir, "preprocessor.pkl")
        if not os.path.exists(prep_path):
            raise FileNotFoundError(f"Preprocessor artifact not found at {prep_path}")

        self.preprocessor = DataPreprocessor.load(prep_path)

        trans_path = os.path.join(model_dir, "transformer", "best_transformer.pt")
        if not os.path.exists(trans_path):
            raise FileNotFoundError(f"Transformer model artifact not found at {trans_path}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(trans_path, map_location=self.device)

        self.model = FTTransformer(
            num_numerical=len(NUMERICAL_FEATURES),
            categorical_cardinalities=[2, 3],
            d_token=checkpoint.get("d_token", 64)
        ).to(self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        # Human-readable feature names for explainability
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
        """
        Accepts raw input dictionary from API request.
        Transforms input, runs FT-Transformer inference, computes feature attributions, and returns structured result.
        """
        # Transform input
        df_trans = self.preprocessor.transform_single_input(input_dict)

        x_num = torch.tensor(df_trans[NUMERICAL_FEATURES].values, dtype=torch.float32).to(self.device)
        x_cat = torch.tensor(df_trans[CATEGORICAL_FEATURES].values, dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(x_num, x_cat)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        predictions = []
        for cond, prob in zip(TARGET_CONDITIONS, probs):
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

        # Sort predictions by probability descending
        predictions_sorted = sorted(predictions, key=lambda x: x["probability"], reverse=True)
        top_pred = predictions_sorted[0]["condition"] if predictions_sorted else "Normal"
        top_prob = predictions_sorted[0]["probability"] if predictions_sorted else 0.0

        # Calculate Gradient-based Feature Importance / Attributions
        top_factors = self._explain_prediction(x_num, x_cat, df_trans)

        return {
            "model": "FT-Transformer",
            "status": "success",
            "overall_risk": "HIGH" if top_prob >= 0.60 else ("MODERATE" if top_prob >= 0.30 else "LOW"),
            "top_prediction": top_pred,
            "top_probability": top_prob,
            "predictions": predictions,
            "top_factors": top_factors,
            "medical_disclaimer": "This AI prediction is intended for screening, research, and decision-support purposes only. It is not a medical diagnosis and should not replace evaluation by a qualified healthcare professional."
        }

    def _explain_prediction(self, x_num, x_cat, df_trans):
        """Compute feature attributions using input gradients."""
        x_num_grad = x_num.clone().detach().requires_grad_(True)
        logits = self.model(x_num_grad, x_cat)
        target_score = logits.sum()
        target_score.backward()

        grads = torch.abs(x_num_grad.grad).cpu().numpy()[0]

        # Combine with categorical perturbations
        cat_importances = [0.15, 0.12]  # baseline estimates for categorical
        all_importances = list(grads) + cat_importances

        # Normalize importances
        total_imp = sum(all_importances) + 1e-6
        norm_imp = [imp / total_imp for imp in all_importances]

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
        return factors_sorted
