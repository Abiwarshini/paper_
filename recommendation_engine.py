"""Rule-based recommendation engine for child malnutrition risk profiling."""

from __future__ import annotations

from typing import Iterable


def _contains_any(feature_names: Iterable[str], patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        for feature_name in feature_names:
            if pattern in feature_name:
                return True
    return False


def generate_recommendations(sample_features: dict[str, object], top_shap_features: list[str], risk_level: str) -> list[str]:
    """Generate personalized recommendations from sample features and SHAP-driven risk."""
    recommendations: list[str] = []
    top_features = [feature.lower() for feature in top_shap_features]

    if risk_level in ["High", "Critical"]:
        recommendations.append("Visit a nearby healthcare center for targeted assessment")
        recommendations.append("Monitor the child's growth every month")

    mother_bmi = sample_features.get("mother_bmi")
    if mother_bmi is not None and mother_bmi < 18.5:
        recommendations.append("Improve maternal nutrition and monitor maternal BMI")

    if sample_features.get("low_anc_visits") == 1:
        recommendations.append("Ensure complete antenatal visits and maternal follow-up")

    if sample_features.get("low_birth_weight") == 1 or (sample_features.get("birth_weight") is not None and sample_features.get("birth_weight") < 2.5):
        recommendations.append("Follow up on low birth weight with pediatric growth monitoring")

    child_age = sample_features.get("child_age_months")
    breastfeeding_duration = sample_features.get("breastfeeding_duration")
    if child_age is not None and child_age <= 6:
        recommendations.append("Encourage exclusive breastfeeding for infants under 6 months")
    elif child_age is not None and 6 < child_age <= 24:
        recommendations.append("Improve age-appropriate dietary diversity and protein intake")

    if sample_features.get("sanitation_risk_index") is not None and sample_features.get("sanitation_risk_index") >= 1:
        recommendations.append("Improve household sanitation and access to safe drinking water")

    if _contains_any(top_features, ["wealth_score", "wealth_quintile"]):
        recommendations.append("Strengthen household food security and nutritious food access")

    if _contains_any(top_features, ["mother_bmi_cat_underweight", "mother_underweight"]):
        recommendations.append("Support maternal nutrition and postpartum health services")

    if _contains_any(top_features, ["sanitation_risk_index", "water_source", "toilet_type", "cooking_fuel"]):
        recommendations.append("Reduce sanitation-related health risks through clean water and safe toilets")

    if _contains_any(top_features, ["birth_weight", "low_birth_weight"]):
        recommendations.append("Use targeted feeding support for low birth weight infants")

    if not recommendations:
        recommendations.append("Continue regular growth monitoring and preventive child health care")

    # Keep the recommendation list unique and preserve order.
    seen = set()
    unique_recommendations: list[str] = []
    for recommendation in recommendations:
        if recommendation not in seen:
            seen.add(recommendation)
            unique_recommendations.append(recommendation)

    return unique_recommendations[:5]
