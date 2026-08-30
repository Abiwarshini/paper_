"""Streamlit app for the existing child malnutrition prediction pipeline."""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd

from dashboard_utils import get_dashboard_field_options, get_dashboard_numeric_fields, predict_stunting

RISK_COLORS = {
    "Low": "#2ecc71",
    "Moderate": "#f1c40f",
    "High": "#e67e22",
    "Critical": "#e74c3c",
}


def render_sidebar() -> dict[str, object]:
    st.sidebar.title("Child Nutrition Dashboard")
    st.sidebar.markdown(
        "This dashboard uses your trained XGBoost model and existing preprocessing pipeline to predict child stunting risk from raw household and child inputs."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model details")
    st.sidebar.markdown("- Trained XGBoost model\n- SHAP explainability\n- Risk score + recommendations")
    st.sidebar.markdown("---")
    with st.sidebar.expander("Data entry guidance", expanded=False):
        st.markdown(
            "Enter the child and household attributes below. The app will preprocess the raw inputs exactly as your pipeline did, then run the saved model to return risk and recommendations."
        )
    return {}


def render_prediction_cards(results: dict[str, object]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Prediction", results["prediction"])
    col2.metric("Probability", f"{results['probability']:.2%}")
    col3.metric("Risk score", results["risk_score"])
    risk_level = results["risk_level"]
    col4.markdown(
        f"<div style='padding:16px; border-radius:12px; background:{RISK_COLORS[risk_level]}; color:white; text-align:center;'>"
        f"<strong>{risk_level}</strong>"
        "</div>",
        unsafe_allow_html=True,
    )


def render_shap_section(results: dict[str, object]) -> None:
    st.subheader("SHAP explainability")
    st.write(
        "The plots below explain which engineered features influenced the stunting prediction for the current child profile."
    )
    shap_features = results["shap_top_features"]
    st.write("**Top SHAP features for this prediction:**")
    st.write(", ".join(shap_features))

    shap_plot = px.bar(
        x=[1] * len(shap_features),
        y=shap_features,
        orientation="h",
        labels={"x": "Importance rank", "y": "Feature"},
        title="Top SHAP features for this sample",
    )
    shap_plot.update_layout(showlegend=False, height=320)
    st.plotly_chart(shap_plot, use_container_width=True)


def render_recommendations(results: dict[str, object]) -> None:
    st.subheader("Personalized recommendations")
    for rec in results["recommendations"]:
        st.markdown(f"- {rec}")


def build_form() -> dict[str, object]:
    st.header("Child & Household Information")
    st.markdown("Enter the raw data values used by the NFHS-5 survey variables.")

    options = get_dashboard_field_options()

    with st.form(key="prediction_form"):
        st.subheader("Child demographics")
        col1, col2, col3 = st.columns(3)
        child_sex = col1.selectbox("Child sex", options["child_sex"], index=0)
        child_age_months = col2.number_input("Child age (months)", min_value=0.0, max_value=59.0, value=24.0)
        birth_order = col3.number_input("Birth order", min_value=1, max_value=15, value=1)

        col4, col5, col6 = st.columns(3)
        birth_size = col4.selectbox("Birth size", options["birth_size"], index=2)
        birth_weight = col5.number_input("Birth weight (kg)", min_value=0.5, max_value=6.0, value=2.8, format="%.1f")
        birth_weight_source = col6.selectbox("Birth weight source", options["birth_weight_source"], index=0)

        st.subheader("Maternal & household factors")
        col7, col8, col9 = st.columns(3)
        education = col7.selectbox("Mother education level", options["education"], index=0)
        wealth_quintile = col8.selectbox("Wealth quintile", options["wealth_quintile"], index=2)
        residence = col9.selectbox("Residence type", options["residence"], index=0)

        col10, col11, col12 = st.columns(3)
        gender_hh_head = col10.selectbox("Gender of household head", options["gender_hh_head"], index=0)
        dist_market_proxy = col11.selectbox("Distance to market", options["dist_market_proxy"], index=0)
        hhsize = col12.number_input("Household size", min_value=1, max_value=20, value=5)

        col13, col14, col15 = st.columns(3)
        age_hh_head_proxy = col13.number_input("Head age (proxy)", min_value=15.0, max_value=80.0, value=35.0, format="%.1f")
        wealth_score = col14.number_input("Wealth score", min_value=-5.0, max_value=10.0, value=0.0, format="%.2f")
        anc_visits = col15.number_input("ANC visits", min_value=0, max_value=15, value=3)

        col16, col17, col18 = st.columns(3)
        mother_weight = col16.number_input("Mother weight (kg)", min_value=30.0, max_value=120.0, value=55.0, format="%.1f")
        mother_height = col17.number_input("Mother height (cm)", min_value=120.0, max_value=190.0, value=155.0, format="%.1f")
        mother_bmi = col18.number_input("Mother BMI", min_value=10.0, max_value=40.0, value=22.0, format="%.1f")

        col19, col20, col21 = st.columns(3)
        breastfeeding_duration = col19.number_input("Breastfeeding duration (months)", min_value=0.0, max_value=60.0, value=6.0, format="%.1f")
        mother_marital_status = col20.selectbox("Mother marital status", options["mother_marital_status"], index=1)
        measles_vaccine = col21.selectbox("Measles vaccine doses", options["measles_vaccine"], index=0)

        st.subheader("Child health status")
        col22, col23, col24 = st.columns(3)
        diarrhea_recent = col22.selectbox("Recent diarrhea", options["diarrhea_recent"], index=0)
        fever_recent = col23.selectbox("Recent fever", options["fever_recent"], index=0)
        cough_recent = col24.selectbox("Recent cough", options["cough_recent"], index=0)

        col25, col26, col27 = st.columns(3)
        water_source = col25.selectbox("Water source", options["water_source"], index=0)
        toilet_type = col26.selectbox("Toilet type", options["toilet_type"], index=0)
        cooking_fuel = col27.selectbox("Cooking fuel", options["cooking_fuel"], index=0)

        submit_button = st.form_submit_button("Predict")

    return {
        "education": education,
        "wealth_quintile": wealth_quintile,
        "residence": residence,
        "gender_hh_head": gender_hh_head,
        "dist_market_proxy": dist_market_proxy,
        "child_sex": child_sex,
        "birth_order": birth_order,
        "birth_size": birth_size,
        "birth_weight": birth_weight,
        "birth_weight_source": birth_weight_source,
        "breastfeeding_duration": breastfeeding_duration,
        "measles_vaccine": measles_vaccine,
        "diarrhea_recent": diarrhea_recent,
        "fever_recent": fever_recent,
        "cough_recent": cough_recent,
        "mother_marital_status": mother_marital_status,
        "water_source": water_source,
        "toilet_type": toilet_type,
        "cooking_fuel": cooking_fuel,
        "age_hh_head_proxy": age_hh_head_proxy,
        "hhsize": hhsize,
        "wealth_score": wealth_score,
        "child_age_months": child_age_months,
        "birth_order": birth_order,
        "birth_weight": birth_weight,
        "mother_weight": mother_weight,
        "mother_height": mother_height,
        "mother_bmi": mother_bmi,
        "anc_visits": anc_visits,
        "breastfeeding_duration": breastfeeding_duration,
    }, submit_button


def main() -> None:
    st.set_page_config(page_title="Child Malnutrition Risk Dashboard", layout="wide")
    render_sidebar()

    raw_inputs, submit_button = build_form()

    if submit_button:
        with st.spinner("Running prediction through the saved model and pipeline..."):
            results = predict_stunting(raw_inputs)

        st.markdown("---")
        render_prediction_cards(results)
        st.markdown("---")
        render_shap_section(results)
        st.markdown("---")
        render_recommendations(results)

        st.markdown("---")
        st.write("### Input fields submitted")
        st.json(raw_inputs)


if __name__ == "__main__":
    main()
