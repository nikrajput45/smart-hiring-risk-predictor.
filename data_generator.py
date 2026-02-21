import numpy as np
import pandas as pd

np.random.seed(42)

n = 1000  # number of candidates

data = pd.DataFrame({
    "cgpa": np.round(np.random.uniform(5.0, 10.0, n), 2),
    "technical_score": np.random.randint(40, 100, n),
    "aptitude_score": np.random.randint(40, 100, n),
    "communication_score": np.random.randint(1, 10, n),
    "internship_count": np.random.randint(0, 5, n),
    "project_count": np.random.randint(1, 8, n),
    "experience_months": np.random.randint(0, 36, n),
    "leadership_score": np.random.randint(1, 10, n),
    "cultural_fit_score": np.random.randint(1, 10, n)
})

# Create target variable based on weighted logic
data["success"] = (
    (data["technical_score"] * 0.3) +
    (data["communication_score"] * 5) +
    (data["cgpa"] * 2) +
    (data["internship_count"] * 3) +
    (data["experience_months"] * 0.5)
)

# Convert into binary classification
data["success"] = np.where(data["success"] > data["success"].median(), 1, 0)

print(data.head())
data.to_csv("hiring_data.csv", index=False)
print("Dataset saved as hiring_data.csv")