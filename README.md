# 🧠 MindRisk — AI Mental Health Risk Screener
**AI Mental Health UK | Viva Demo Project**

## What It Does
- **PHQ-9 + GAD-7 questionnaire** intake form (full 9+7 questions)
- **ML Risk Model** (Gradient Boosting, 95% accuracy) trained on 783 real patients
- **Cluster assignment** — matches patient to 5 NHS population profiles
- **IAPT Referral Forecast** — 3-month forecast with confidence intervals

## Tech Stack
| Layer | Tech |
|-------|------|
| Backend | FastAPI + scikit-learn |
| Frontend | Streamlit + Plotly |
| Model | Gradient Boosting Classifier |
| Data | PHQ-9/GAD-7 dataset (n=783), NHS IAPT referrals |

## Project Structure
```
mindRisk/
├── backend/
│   └── main.py          ← FastAPI app (4 endpoints)
├── frontend/
│   └── app.py           ← Streamlit UI
├── models/
│   ├── risk_model.pkl   ← Trained GB model
│   └── scaler.pkl       ← StandardScaler
├── data/
│   ├── depression_anxiety_data.csv
│   ├── cluster_profiles.csv
│   ├── forecast_3month_values.csv
│   └── referrals_past_data.csv
└── requirements.txt
```

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Risk level + probability + recommendation |
| `/cluster` | POST | Population cluster match |
| `/forecast` | GET | 3-month IAPT forecast |
| `/clusters/summary` | GET | All 5 cluster profiles |
| `/docs` | GET | Swagger UI |

## ▶️ Run Instructions

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Start FastAPI backend (Terminal 1)
```bash
cd backend
uvicorn api.main:app --reload --port 8000
```
API docs: http://localhost:8000/docs

### Step 3 — Start Streamlit frontend (Terminal 2)
```bash
cd frontend
streamlit run app.py
```
App: http://localhost:8501

## Model Performance
- **Algorithm**: Gradient Boosting Classifier
- **Training data**: 783 real patients (PHQ-9/GAD-7 open dataset)
- **Test accuracy**: 95%
- **Features**: PHQ-9 score, GAD-7 score, Age, BMI, Epworth Sleepiness Score, Gender
- **Classes**: Low Risk / Moderate Risk / High Risk

## Links to Sprint Board Tasks
- AIML-2 (Week 2): PHQ/GAD feature engineering ✅
- AIML-4 (Week 3): Model packaging for API ✅
- DS-1 (Week 3): IAPT demand forecasting ✅
- DS-3 (Week 3): Geo/cluster analysis ✅
- FS-3 (Week 2): Chatbot proxy / prediction API ✅