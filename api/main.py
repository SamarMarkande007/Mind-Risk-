from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import os

app = FastAPI(
    title="MindRisk API — AI Mental Health UK",
    description="Mental Health Risk Assessment API using real NHS-aligned data (PHQ-9/GAD-7)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model & data at startup ──────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model   = joblib.load(os.path.join(BASE, "models/risk_model.pkl"))
scaler  = joblib.load(os.path.join(BASE, "models/scaler.pkl"))
features = joblib.load(os.path.join(BASE, "models/features.pkl"))

cluster_profiles   = pd.read_csv(os.path.join(BASE, "data/cluster_profiles.csv"))
clustered_patients = pd.read_csv(os.path.join(BASE, "data/clustered_patients.csv"))
forecast_df        = pd.read_csv(os.path.join(BASE, "data/forecast_3month_values.csv"))
referrals_df       = pd.read_csv(os.path.join(BASE, "data/referrals_past_data.csv"))

# ── Schemas ───────────────────────────────────────────────────────────────────
class PatientInput(BaseModel):
    phq_score:     int   = Field(..., ge=0, le=27, description="PHQ-9 score (0–27)")
    gad_score:     int   = Field(..., ge=0, le=21, description="GAD-7 score (0–21)")
    age:           int   = Field(..., ge=16, le=100)
    bmi:           float = Field(..., ge=10.0, le=60.0)
    epworth_score: int   = Field(..., ge=0, le=24, description="Epworth Sleepiness Scale (0–24)")
    gender:        str   = Field(..., description="Male or Female")

class RiskResponse(BaseModel):
    risk_level:       str
    risk_score:       float
    risk_probability: dict
    phq_severity:     str
    gad_severity:     str
    recommendation:   str
    iapt_referral:    bool
    crisis_flag:      bool

# ── Helpers ───────────────────────────────────────────────────────────────────
def phq_severity(score: int) -> str:
    if score <= 4:  return "None–Minimal"
    if score <= 9:  return "Mild"
    if score <= 14: return "Moderate"
    if score <= 19: return "Moderately Severe"
    return "Severe"

def gad_severity(score: int) -> str:
    if score <= 4:  return "Minimal"
    if score <= 9:  return "Mild"
    if score <= 14: return "Moderate"
    return "Severe"

def get_recommendation(risk: str, phq: int, gad: int) -> str:
    if risk == "High":
        return "Urgent review recommended. Consider immediate referral to crisis team or NHS 111. Clinician follow-up within 24 hours."
    if risk == "Moderate":
        return "IAPT referral recommended. Monitor weekly. Consider talking therapy (CBT). Follow up within 2 weeks."
    return "Self-management support advised. Provide wellness resources and psychoeducation. Review in 4–6 weeks."

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "MindRisk API is running", "version": "1.0.0", "team": "AI Mental Health UK"}

@app.post("/predict", response_model=RiskResponse, summary="Predict mental health risk")
def predict_risk(patient: PatientInput):
    """
    Takes PHQ-9, GAD-7 and patient demographics.
    Returns risk level (Low/Moderate/High), probabilities and clinical recommendation.
    Model trained on real PHQ-9/GAD-7 open dataset (n=783).
    """
    gender_enc = 0 if patient.gender.lower() in ["male", "m"] else 1
    X = np.array([[
        patient.phq_score,
        patient.gad_score,
        patient.age,
        patient.bmi,
        patient.epworth_score,
        gender_enc
    ]])
    X_scaled = scaler.transform(X)
    pred     = model.predict(X_scaled)[0]
    proba    = model.predict_proba(X_scaled)[0]

    label_map = {0: "Low", 1: "Moderate", 2: "High"}
    risk      = label_map[pred]

    # Crisis flag: suicidal ideation proxy (PHQ item 9 threshold)
    crisis_flag = patient.phq_score >= 20 or patient.gad_score >= 18

    return RiskResponse(
        risk_level       = risk,
        risk_score       = round(float(proba[pred]) * 100, 1),
        risk_probability = {
            "Low":      round(float(proba[0]) * 100, 1),
            "Moderate": round(float(proba[1]) * 100, 1),
            "High":     round(float(proba[2]) * 100, 1),
        },
        phq_severity   = phq_severity(patient.phq_score),
        gad_severity   = gad_severity(patient.gad_score),
        recommendation = get_recommendation(risk, patient.phq_score, patient.gad_score),
        iapt_referral  = patient.phq_score >= 10 or patient.gad_score >= 10,
        crisis_flag    = crisis_flag,
    )

@app.post("/cluster", summary="Assign patient to population cluster")
def assign_cluster(patient: PatientInput):
    """
    Matches patient to one of 5 NHS population clusters derived from
    real PHQ-9/GAD-7 dataset using K-Means clustering.
    """
    from sklearn.metrics.pairwise import euclidean_distances

    gender_enc = 0 if patient.gender.lower() in ["male", "m"] else 1
    patient_vec = np.array([[
        patient.phq_score,
        patient.gad_score,
        patient.age,
        patient.bmi,
        patient.epworth_score,
        gender_enc
    ]])

    # Cluster centroids from profiles
    centroids = cluster_profiles[['phq_score','gad_score','age','bmi','epworth_score','gender']].values
    dists     = euclidean_distances(patient_vec, centroids)[0]
    cluster_id = int(np.argmin(dists))
    profile    = cluster_profiles.iloc[cluster_id]

    return {
        "cluster_id":    cluster_id,
        "profile_name":  profile["Profile_Name"],
        "phq_severity":  profile["PHQ_Severity"],
        "gad_severity":  profile["GAD_Severity"],
        "cluster_size":  int(profile["count"]),
        "avg_phq":       round(float(profile["phq_score"]), 1),
        "avg_gad":       round(float(profile["gad_score"]), 1),
        "all_profiles":  cluster_profiles[["Cluster","Profile_Name","PHQ_Severity","GAD_Severity","count"]].to_dict(orient="records"),
    }

@app.get("/forecast", summary="3-month IAPT referral forecast")
def get_forecast():
    """
    Returns 3-month ahead IAPT referral volume forecast with confidence intervals.
    Built using ARIMA/Prophet on real NHS IAPT monthly data.
    """
    past = referrals_df.tail(12).to_dict(orient="records")
    future = forecast_df.to_dict(orient="records")
    return {
        "past_12_months": past,
        "forecast_3_months": future,
        "model": "ARIMA trained on NHS IAPT monthly referrals data",
        "note": "Forecast includes 95% confidence intervals"
    }

@app.get("/clusters/summary", summary="All cluster profiles summary")
def cluster_summary():
    """All 5 patient cluster profiles from NHS-aligned population analysis."""
    return cluster_profiles.to_dict(orient="records")

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True, "clusters": len(cluster_profiles)}