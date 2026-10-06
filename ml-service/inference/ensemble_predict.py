from preprocessing.preprocessing import TARGET_CONDITIONS


MODEL_WEIGHTS = {
    "dnn": 0.9520,
    "transformer": 0.9380,
}


def predict_ensemble(input_dict, dnn_predictor, transformer_predictor):
    """Combine calibrated DNN and FT-Transformer probabilities by target."""
    dnn_result = dnn_predictor(input_dict)
    transformer_result = transformer_predictor.predict(input_dict)

    weight_total = sum(MODEL_WEIGHTS.values())
    dnn_weight = MODEL_WEIGHTS["dnn"] / weight_total
    transformer_weight = MODEL_WEIGHTS["transformer"] / weight_total

    dnn_conditions = {item["condition"]: item for item in dnn_result["conditions"]}
    transformer_conditions = {
        item["condition"]: item for item in transformer_result["conditions"]
    }

    conditions = []
    for condition in TARGET_CONDITIONS:
        dnn_item = dnn_conditions[condition]
        transformer_item = transformer_conditions[condition]
        probability = (
            dnn_weight * dnn_item["probability"]
            + transformer_weight * transformer_item["probability"]
        )
        threshold = (
            dnn_weight * dnn_item.get("threshold_used", 0.5)
            + transformer_weight * transformer_item.get("threshold_used", 0.5)
        )
        probability = round(float(probability), 4)
        threshold = round(float(threshold), 4)

        if probability >= 0.65:
            severity = "HIGH"
        elif probability >= threshold:
            severity = "MODERATE"
        else:
            severity = "LOW"

        conditions.append({
            "condition": condition,
            "probability": probability,
            "percentage": round(probability * 100, 1),
            "risk_flag": bool(probability >= threshold),
            "severity": severity,
            "threshold_used": threshold,
        })

    malnutrition = next(
        item for item in conditions if item["condition"] == "Malnutrition"
    )
    probability = malnutrition["probability"]
    if probability >= 0.65:
        overall_risk = "HIGH"
        prediction = "Malnourished (High Risk)"
    elif probability >= malnutrition["threshold_used"]:
        overall_risk = "MODERATE"
        prediction = "Malnourished (Moderate Risk)"
    else:
        overall_risk = "LOW"
        prediction = "Normal (Well-Nourished)"

    return {
        "model": "DNN + FT-Transformer Ensemble",
        "architecture": "Weighted probability averaging",
        "ensemble_weights": {
            "dnn": round(dnn_weight, 4),
            "transformer": round(transformer_weight, 4),
        },
        "overall_risk": overall_risk,
        "prediction": prediction,
        "probability": probability,
        "percentage": round(probability * 100, 1),
        "conditions": conditions,
        "top_factors": dnn_result.get("top_factors", []),
        "explainability_method": "DNN input gradients; ensemble probabilities",
        "medical_disclaimer": (
            "Screening support for academic/research purposes; not a medical diagnosis."
        ),
    }