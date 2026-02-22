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
technical = st.sidebar.slider("Technical Score", 0, 100, 70)
aptitude = st.sidebar.slider("Aptitude Score", 0, 100, 65)
communication = st.sidebar.slider("Communication Score", 0, 100, 75)
internships = st.sidebar.slider("Internships", 0, 5, 1)
projects = st.sidebar.slider("Projects", 0, 10, 3)
experience = st.sidebar.slider("Experience (Months)", 0, 60, 12)
leadership = st.sidebar.slider("Leadership Score", 0, 100, 60)
cultural_fit = st.sidebar.slider("Cultural Fit Score", 0, 100, 70)

threshold = st.sidebar.slider("Decision Threshold", 0.0, 1.0, 0.5)

# -------------------------------
# Prepare Input Data
# -------------------------------
input_data = pd.DataFrame([{
    "CGPA": cgpa,
    "Technical_Score": technical,
    "Aptitude_Score": aptitude,
    "Communication_Score": communication,
    "Internships": internships,
    "Projects": projects,
    "Experience_Months": experience,
    "Leadership_Score": leadership,
    "Cultural_Fit_Score": cultural_fit
}])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict Hiring Risk"):

    probability = model.predict_proba(input_data)[0][1]
    prediction = 1 if probability >= threshold else 0

    st.subheader("Prediction Result")

    st.write(f"**Predicted Probability of High Risk:** {probability:.2f}")

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
features = input_data.columns

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": feature_importance
}).sort_values(by="Importance", ascending=False)

fig, ax = plt.subplots()
ax.barh(importance_df["Feature"], importance_df["Importance"])
ax.invert_yaxis()
ax.set_xlabel("Importance Score")

st.pyplot(fig)
