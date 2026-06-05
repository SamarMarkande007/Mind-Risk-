import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
df = pd.read_csv("data/depression_anxiety_data.csv")

print("Columns:", df.columns.tolist())
print("Shape:", df.shape)

# Create risk level target
def risk_level(row):
    phq = row["phq_score"]
    gad = row["gad_score"]
    suicidal = row["suicidal"]

    if suicidal or phq >= 15 or gad >= 15:
        return 2      # High

    elif phq >= 10 or gad >= 10:
        return 1      # Moderate

    else:
        return 0      # Low

df["risk_level"] = df.apply(risk_level, axis=1)

print("\nRisk distribution:")
print(df["risk_level"].value_counts().to_dict())

# Gender encoding
df["gender_enc"] = (
    df["gender"]
    .map({
        "Male": 0,
        "Female": 1,
        "M": 0,
        "F": 1
    })
    .fillna(0)
)

# Features
features = [
    "phq_score",
    "gad_score",
    "age",
    "bmi",
    "epworth_score",
    "gender_enc"
]

df_clean = df[features + ["risk_level"]].dropna()

X = df_clean[features]
y = df_clean["risk_level"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Scaling
scaler = StandardScaler()

X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Model
model = GradientBoostingClassifier(
    n_estimators=100,
    max_depth=4,
    random_state=42
)

model.fit(X_train_s, y_train)

y_pred = model.predict(X_test_s)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Low", "Moderate", "High"]
    )
)

# Save artifacts
joblib.dump(model, "models/risk_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(features, "models/features.pkl")

print("\nModel saved!")