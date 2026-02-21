import streamlit as st
import pickle
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Smart Hiring Risk Predictor", layout="wide")

st.title("💼 Smart Hiring Risk Prediction System")
st.markdown("AI-powered candidate evaluation using XGBoost")

# -------------------------------
# Load Model (Cached)
# -------------------------------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("📋 Candidate Details")

cgpa = st.sidebar.slider("CGPA", 5.0, 10.0, 7.0)
technical_score = st.sidebar.slider("Technical Score", 40, 100, 70)
aptitude_score = st.sidebar.slider("Aptitude Score", 40, 100, 70)
communication_score = st.sidebar.slider("Communication Score", 1, 10, 5)
internship_count = st.sidebar.slider("Internship Count", 0, 5, 1)
project_count = st.sidebar.slider("Project Count", 1, 8, 3)
experience_months = st.sidebar.slider("Experience (Months)", 0, 36, 6)
leadership_score = st.sidebar.slider("Leadership Score", 1, 10, 5)
cultural_fit_score = st.sidebar.slider("Cultural Fit Score", 1, 10, 5)

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.3,
    max_value=0.9,
    value=0.5,
    step=0.05
)

predict_button = st.sidebar.button("🚀 Predict")

# -------------------------------
# Prediction Logic
# -------------------------------
if predict_button:

    input_data = np.array([[cgpa, technical_score, aptitude_score,
                            communication_score, internship_count,
                            project_count, experience_months,
                            leadership_score, cultural_fit_score]])

    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= threshold else 0

    # Risk Category
    if probability >= 0.75:
        risk_category = "🟢 Low Risk - Highly Recommended"
    elif probability >= 0.50:
        risk_category = "🟡 Moderate Risk - Needs Review"
    else:
        risk_category = "🔴 High Risk - Not Recommended"

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Success Probability", f"{probability*100:.2f}%")

    with col2:
        st.info(f"Risk Category: {risk_category}")

    st.divider()

    # -------------------------------
    # SHAP Explanation
    # -------------------------------
    st.subheader("🔎 Candidate-Level SHAP Explanation")

    explainer = shap.TreeExplainer(model)

    feature_names = ["cgpa", "technical_score", "aptitude_score",
                     "communication_score", "internship_count",
                     "project_count", "experience_months",
                     "leadership_score", "cultural_fit_score"]

    input_df = pd.DataFrame(input_data, columns=feature_names)

    shap_values = explainer.shap_values(input_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(values=shap_values[0],
                         base_values=explainer.expected_value,
                         data=input_df.iloc[0],
                         feature_names=feature_names),
        show=False
    )

    st.pyplot(fig)

    st.divider()

    # -------------------------------
    # Global Feature Importance
    # -------------------------------
    st.subheader("📊 Global Feature Importance")

    importance = model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=True)

    fig2, ax2 = plt.subplots()
    ax2.barh(importance_df["Feature"], importance_df["Importance"])
    ax2.set_xlabel("Importance Score")
    ax2.set_title("XGBoost Feature Importance")

    st.pyplot(fig2)