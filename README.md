Smart Hiring Risk Predictor
An AI-powered recruitment decision support system that predicts candidate hiring success risk using Machine Learning.
The model is trained using XGBoost on a synthetic recruitment dataset.

Features

- Candidate risk prediction (Binary Classification)
- Adjustable probability threshold
- Risk categorization (Low / Medium / High)
- Global feature importance visualization
- Interactive professional UI built with Streamlit
  
Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Streamlit
  Input Features

- CGPA
- Technical Score
- Aptitude Score
- Communication Score
- Internship Count
- Project Count
- Experience (Months)
- Leadership Score
- Cultural Fit Score

 Model Performance

- Accuracy: ~96%
- ROC-AUC: ~0.99

## 🔧 How to Run Locally


pip install -r requirements.txt
streamlit run app.py
