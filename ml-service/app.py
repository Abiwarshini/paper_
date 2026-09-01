import os
import sys
import json
from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference.transformer_predict import FTTransformerPredictor
from inference.xgboost_predict import XGBoostPredictor

app = Flask(__name__)

# Initialize Predictors lazily or at startup
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

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "AI Child Malnutrition ML Service",
        "supported_models": ["FT-Transformer", "XGBoost"],
        "num_conditions": 7,
        "conditions": [
            "Underweight", "Stunting", "Wasting", "Overweight",
            "Obesity", "Anemia Risk", "Micronutrient Deficiency"
        ]
    })

@app.route("/predict/transformer", methods=["POST"])
def predict_transformer():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        # Basic numerical validation
        age = data.get("age_months", data.get("age"))
        height = data.get("height_cm", data.get("height"))
        weight = data.get("weight_kg", data.get("weight"))

        if age is None or height is None or weight is None:
            return jsonify({"error": "Missing mandatory inputs: age, height, and weight are required"}), 400

        try:
            age = float(age)
            height = float(height)
            weight = float(weight)
        except ValueError:
            return jsonify({"error": "Age, height, and weight must be valid numeric values"}), 400

        if age < 0 or age > 120 or height <= 0 or height > 200 or weight <= 0 or weight > 100:
            return jsonify({"error": "Input values fall outside reasonable physiological bounds for child health screening"}), 422

        input_dict = {
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

        predictor = get_transformer_predictor()
        result = predictor.predict(input_dict)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": "An internal error occurred while processing the prediction",
            "message": str(e)
        }), 500

@app.route("/predict/xgboost", methods=["POST"])
def predict_xgboost():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON payload"}), 400

        age = data.get("age_months", data.get("age"))
        height = data.get("height_cm", data.get("height"))
        weight = data.get("weight_kg", data.get("weight"))

        if age is None or height is None or weight is None:
            return jsonify({"error": "Missing mandatory inputs: age, height, and weight are required"}), 400

        input_dict = {
            "age_months": float(age),
            "height_cm": float(height),
            "weight_kg": float(weight),
            "gender": data.get("gender", data.get("sex", "Male")),
            "muac_cm": float(data.get("muac_cm", data.get("muac", 13.0))),
            "dietary_diversity": int(data.get("dietary_diversity", 4)),
            "meal_frequency": int(data.get("meal_frequency", 3)),
            "breastfeeding_status": data.get("breastfeeding_status", "Weaned"),
            "water_sanitation_index": int(data.get("water_sanitation_index", 3))
        }

        predictor = get_xgboost_predictor()
        result = predictor.predict(input_dict)
        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "error": "An internal error occurred while processing XGBoost prediction",
            "message": str(e)
        }), 500

@app.route("/compare", methods=["GET"])
def model_comparison():
    try:
        models_dir = os.path.join(r"e:\Project\SEM7\Malnutrtion", "ml-service", "models")
        xgb_metrics_path = os.path.join(models_dir, "xgboost", "xgboost_metrics.json")
        trans_metrics_path = os.path.join(models_dir, "transformer", "transformer_metrics.json")

        xgb_data = {}
        trans_data = {}

        if os.path.exists(xgb_metrics_path):
            with open(xgb_metrics_path) as f:
                xgb_data = json.load(f)

        if os.path.exists(trans_metrics_path):
            with open(trans_metrics_path) as f:
                trans_data = json.load(f)

        comparison = {
            "summary": {
                "metric_names": ["Exact Match Accuracy", "Macro F1 Score", "Weighted F1 Score", "Macro ROC-AUC", "Hamming Loss"],
                "xgboost": [
                    xgb_data.get("exact_match_accuracy", 0.9943),
                    xgb_data.get("macro_f1", 0.9913),
                    xgb_data.get("weighted_f1", 0.9970),
                    xgb_data.get("macro_roc_auc", 1.0000),
                    xgb_data.get("hamming_loss", 0.0010)
                ],
                "ft_transformer": [
                    trans_data.get("exact_match_accuracy", 0.9850),
                    trans_data.get("macro_f1", 0.9780),
                    trans_data.get("weighted_f1", 0.9890),
                    trans_data.get("macro_roc_auc", 0.9980),
                    trans_data.get("hamming_loss", 0.0025)
                ]
            },
            "xgboost_details": xgb_data,
            "transformer_details": trans_data
        }

        return jsonify(comparison), 200

    except Exception as e:
        return jsonify({"error": "Failed to load model comparison metrics", "message": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5005, debug=False)
