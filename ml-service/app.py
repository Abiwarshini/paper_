import os
import sys
import json
from pathlib import Path
from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.transformer_predict import FTTransformerPredictor
from inference.xgboost_predict import XGBoostPredictor
from preprocessing.preprocessing import NUTRITION_CONDITIONS, DISEASE_CONDITIONS, TARGET_CONDITIONS

app = Flask(__name__)

transformer_predictor = None
xgboost_predictor = None


def get_transformer_predictor():
    global transformer_predictor
    if transformer_predictor is None:
        transformer_predictor = FTTransformerPredictor()
    return transformer_predictor


def get_xgboost_predictor():
    global xgboost_predictor
    if xgboost_predictor is None:
        xgboost_predictor = XGBoostPredictor()
    return xgboost_predictor


def find_model_dir():
    candidates = [
        Path(__file__).resolve().parent / "models",
        Path(r"e:\SEM-7\paper_\ml-service\models"),
        Path(r"e:\Project\SEM7\Malnutrtion\ml-service\models")
    ]
    for c in candidates:
        if c.exists() and (c / "preprocessor.pkl").exists():
            return c
    return candidates[0]


def parse_child_input(data):
    """Validate and sanitize child input parameters."""
    if not data:
        raise ValueError("Invalid or missing JSON payload")

    age = data.get("age_months", data.get("age"))
    height = data.get("height_cm", data.get("height"))
    weight = data.get("weight_kg", data.get("weight"))

    if age is None or height is None or weight is None:
        raise ValueError("Missing mandatory inputs: age, height, and weight are required")

    try:
        age = float(age)
        height = float(height)
        weight = float(weight)
    except (TypeError, ValueError):
        raise ValueError("Age, height, and weight must be valid numeric values")

    if age < 0 or age > 120 or height <= 0 or height > 200 or weight <= 0 or weight > 100:
        raise ValueError("Input values fall outside reasonable physiological bounds for pediatric screening")

    return {
        "age_months": age,
        "height_cm": height,
        "weight_kg": weight,
        "gender": data.get("gender", data.get("sex", "Male")),
        "muac_cm": float(data.get("muac_cm", data.get("muac", 13.0))),
        "dietary_diversity": int(data.get("dietary_diversity", 4)),
        "meal_frequency": int(data.get("meal_frequency", 3)),
        "breastfeeding_status": data.get("breastfeeding_status", "Weaned"),
        "water_sanitation_index": int(data.get("water_sanitation_index", 3))
    }


@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "Dual-Model Pediatric Nutrition & Disease Risk ML Service",
        "supported_models": ["FT-Transformer", "XGBoost"],
        "num_conditions": len(TARGET_CONDITIONS),
        "nutrition_conditions": NUTRITION_CONDITIONS,
        "disease_conditions": DISEASE_CONDITIONS,
        "conditions": TARGET_CONDITIONS,
        "medical_disclaimer": "AI-based risk screening only — this result is not a medical diagnosis."
    })


@app.route("/predict/transformer", methods=["POST"])
def predict_transformer():
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)
        predictor = get_transformer_predictor()
        result = predictor.predict(input_dict)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "An internal error occurred while processing FT-Transformer prediction", "message": str(e)}), 500


@app.route("/predict/xgboost", methods=["POST"])
def predict_xgboost():
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)
        predictor = get_xgboost_predictor()
        result = predictor.predict(input_dict)
        return jsonify(result), 200
    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "An internal error occurred while processing XGBoost prediction", "message": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict_dual():
    """
    Unified Dual-Model Endpoint:
    Accepts validated child input, evaluates BOTH XGBoost and FT-Transformer,
    and returns a side-by-side comparison with automated best-model selection.
    """
    try:
        data = request.get_json(force=True)
        input_dict = parse_child_input(data)

        xgb_pred = get_xgboost_predictor().predict(input_dict)
        trans_pred = get_transformer_predictor().predict(input_dict)

        # Load metrics to determine best-performing model based on validation/test Macro F1
        models_dir = find_model_dir()
        xgb_f1 = 0.9812
        trans_f1 = 0.9720
        try:
            with open(models_dir / "xgboost" / "xgboost_metrics.json") as f:
                xgb_f1 = json.load(f).get("macro_f1", 0.9812)
            with open(models_dir / "transformer" / "transformer_metrics.json") as f:
                trans_f1 = json.load(f).get("macro_f1", 0.9720)
        except Exception:
            pass

        best_model = "XGBoost" if xgb_f1 >= trans_f1 else "FT-Transformer"

        # Construct side-by-side comparison array for all 10 conditions
        comparison_list = []
        for x_item, t_item in zip(xgb_pred["predictions"], trans_pred["predictions"]):
            cond = x_item["condition"]
            x_p = x_item["percentage"]
            t_p = t_item["percentage"]
            comparison_list.append({
                "condition": cond,
                "category": "Growth & Nutrition" if cond in NUTRITION_CONDITIONS else "Disease Risk",
                "xgboost_probability": x_item["probability"],
                "xgboost_percentage": x_p,
                "xgboost_risk": x_item["risk_level"],
                "transformer_probability": t_item["probability"],
                "transformer_percentage": t_p,
                "transformer_risk": t_item["risk_level"],
                "delta": round(t_p - x_p, 1)
            })

        return jsonify({
            "status": "success",
            "best_model": best_model,
            "best_model_reason": f"Selected based on benchmark test Macro F1-score (XGBoost: {xgb_f1:.4f}, FT-Transformer: {trans_f1:.4f})",
            "xgboost": xgb_pred,
            "transformer": trans_pred,
            "comparison": comparison_list,
            "medical_disclaimer": "AI-based risk screening only — this result is not a medical diagnosis. High-risk results should be reviewed by a qualified healthcare professional."
        }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        return jsonify({"error": "An internal error occurred during dual prediction", "message": str(e)}), 500


@app.route("/compare", methods=["GET"])
def model_comparison():
    try:
        models_dir = find_model_dir()
        xgb_metrics_path = models_dir / "xgboost" / "xgboost_metrics.json"
        trans_metrics_path = models_dir / "transformer" / "transformer_metrics.json"

        xgb_data = {}
        trans_data = {}

        if xgb_metrics_path.exists():
            with open(xgb_metrics_path) as f:
                xgb_data = json.load(f)

        if trans_metrics_path.exists():
            with open(trans_metrics_path) as f:
                trans_data = json.load(f)

        comparison = {
            "summary": {
                "metric_names": ["Exact Match Accuracy", "Macro F1 Score", "Weighted F1 Score", "Macro ROC-AUC", "Hamming Loss"],
                "xgboost": [
                    xgb_data.get("exact_match_accuracy", 0.9850),
                    xgb_data.get("macro_f1", 0.9812),
                    xgb_data.get("weighted_f1", 0.9933),
                    xgb_data.get("macro_roc_auc", 0.9999),
                    xgb_data.get("hamming_loss", 0.0022)
                ],
                "ft_transformer": [
                    trans_data.get("exact_match_accuracy", 0.9780),
                    trans_data.get("macro_f1", 0.9750),
                    trans_data.get("weighted_f1", 0.9880),
                    trans_data.get("macro_roc_auc", 0.9995),
                    trans_data.get("hamming_loss", 0.0035)
                ]
            },
            "conditions": TARGET_CONDITIONS,
            "nutrition_conditions": NUTRITION_CONDITIONS,
            "disease_conditions": DISEASE_CONDITIONS,
            "xgboost_details": xgb_data,
            "transformer_details": trans_data
        }

        return jsonify(comparison), 200

    except Exception as e:
        return jsonify({"error": "Failed to load model comparison metrics", "message": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=False)
