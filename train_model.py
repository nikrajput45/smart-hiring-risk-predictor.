import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from xgboost import XGBClassifier

# Load dataset
data = pd.read_csv("hiring_data.csv")

# Separate features and target
X = data.drop("success", axis=1)
y = data["success"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.5,
    reg_lambda=1,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:, 1]

# Evaluation
print("Accuracy:", accuracy_score(y_test, pred))
print("ROC-AUC:", roc_auc_score(y_test, prob))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved as model.pkl")