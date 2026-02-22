import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="Smart Hiring Risk Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Smart Hiring Risk Predictor")
st.markdown("AI-powered candidate risk evaluation system using XGBoost")

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("Candidate Details")

cgpa = st.sidebar.slider("CGPA", 5.0, 10.0, 7.5)
technical_score = st.sidebar.slider("Technical Score", 40, 100, 70)
aptitude_score = st.sidebar.slider("Aptitude Score", 40, 100, 65)
communication_score = st.sidebar.slider("Communication Score", 1, 10, 7)
internship_count = st.sidebar.slider("Internship Count", 0, 5, 1)
project_count = st.sidebar.slider("Project Count", 1, 8, 3)
experience_months = st.sidebar.slider("Experience (Months)", 0, 36, 12)
leadership_score = st.sidebar.slider("Leadership Score", 1, 10, 6)
cultural_fit_score = st.sidebar.slider("Cultural Fit Score", 1, 10, 7)

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5)

# -------------------------------
# Prepare Input Data (EXACT MATCH)
# -------------------------------
input_data = pd.DataFrame([[ 
    cgpa,
    technical_score,
    aptitude_score,
    communication_score,
    internship_count,
    project_count,
    experience_months,
    leadership_score,
    cultural_fit_score
]], columns=[
    "cgpa",
    "technical_score",
    "aptitude_score",
    "communication_score",
    "internship_count",
    "project_count",
    "experience_months",
    "leadership_score",
    "cultural_fit_score"
])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Hiring Risk"):

    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= threshold else 0

    st.subheader("Prediction Result")
    st.write(f"**Predicted Probability of Success:** {probability:.2f}")

    # Risk Category Logic
    if probability < 0.4:
        st.success("Low Risk Candidate ✅")
    elif 0.4 <= probability < 0.7:
        st.warning("Medium Risk Candidate ⚠️")
    else:
        st.error("High Risk Candidate ❌")

# -------------------------------
# Global Feature Importance
# -------------------------------
st.subheader("📈 Global Feature Importance")

feature_importance = model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": input_data.columns,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.invert_yaxis()
ax.set_xlabel("Importance Score")

st.pyplot(fig)
