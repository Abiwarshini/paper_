"""Risk scoring utilities for child malnutrition probability outputs."""

from __future__ import annotations


def probability_to_risk_score(probability: float) -> int:
    """Convert a model probability to a 0-100 risk score."""
    score = int(round(float(probability) * 100))
    return max(0, min(score, 100))


def risk_level_from_score(score: int) -> str:
    """Map a risk score into a clinical severity bucket."""
    if score <= 20:
        return "Low"
    if score <= 50:
        return "Moderate"
    if score <= 80:
        return "High"
    return "Critical"


def risk_profile(probability: float) -> dict[str, object]:
    """Return a dictionary containing score and level for a probability."""
    score = probability_to_risk_score(probability)
    level = risk_level_from_score(score)
    return {
        "risk_score": score,
        "risk_level": level,
    }
