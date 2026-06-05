import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="MindRisk — AI Mental Health UK",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8fafc; }
    .stApp { background-color: #f8fafc; }
    .risk-high    { background:#fee2e2; border-left:5px solid #dc2626; padding:16px; border-radius:8px; }
    .risk-moderate{ background:#fef9c3; border-left:5px solid #d97706; padding:16px; border-radius:8px; }
    .risk-low     { background:#dcfce7; border-left:5px solid #16a34a; padding:16px; border-radius:8px; }
    .crisis-box   { background:#7f1d1d; color:white; padding:16px; border-radius:8px; text-align:center; }
    .metric-card  { background:white; border-radius:10px; padding:16px; box-shadow:0 1px 4px rgba(0,0,0,0.1); }
    .cluster-card { background:white; border-radius:10px; padding:16px; border:1px solid #e2e8f0; margin:4px 0; }
    h1 { color: #0f172a; }
    .nhs-header { background:#003087; color:white; padding:12px 20px; border-radius:8px; margin-bottom:20px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="nhs-header">
    <h2 style="margin:0;color:white;">🧠 MindRisk — AI Mental Health Risk Screener</h2>
    <p style="margin:4px 0 0 0;font-size:13px;opacity:0.85;">AI Mental Health UK · NHS-Aligned · PHQ-9 / GAD-7 Model · Trained on Real Clinical Data</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar — Patient Input Form ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 📋 Patient Intake Form")
    st.caption("All fields required for risk assessment")

    st.markdown("**Demographics**")
    age    = st.slider("Age", 16, 90, 30)
    gender = st.radio("Gender", ["Male", "Female"], horizontal=True)
    bmi    = st.number_input("BMI", min_value=10.0, max_value=60.0, value=24.0, step=0.1)

    st.markdown("---")
    st.markdown("**PHQ-9 — Depression Screen**")
    st.caption("Over the last 2 weeks, how often have you been bothered by the following?")
    phq_items = [
        "Little interest or pleasure in doing things",
        "Feeling down, depressed, or hopeless",
        "Trouble falling or staying asleep",
        "Feeling tired or having little energy",
        "Poor appetite or overeating",
        "Feeling bad about yourself",
        "Trouble concentrating on things",
        "Moving or speaking slowly / restless",
        "Thoughts of self-harm or suicide",
    ]
    phq_scores = []
    for i, item in enumerate(phq_items):
        val = st.select_slider(
            f"Q{i+1}: {item[:40]}…" if len(item) > 40 else f"Q{i+1}: {item}",
            options=[0, 1, 2, 3],
            value=0,
            format_func=lambda x: ["Not at all","Several days","More than half","Nearly every day"][x],
            key=f"phq_{i}"
        )
        phq_scores.append(val)
    phq_total = sum(phq_scores)
    st.info(f"**PHQ-9 Total: {phq_total} / 27**")

    st.markdown("---")
    st.markdown("**GAD-7 — Anxiety Screen**")
    gad_items = [
        "Feeling nervous, anxious or on edge",
        "Not being able to stop or control worrying",
        "Worrying too much about different things",
        "Trouble relaxing",
        "Being so restless that it's hard to sit still",
        "Becoming easily annoyed or irritable",
        "Feeling afraid as if something awful might happen",
    ]
    gad_scores = []
    for i, item in enumerate(gad_items):
        val = st.select_slider(
            f"G{i+1}: {item[:40]}…" if len(item) > 40 else f"G{i+1}: {item}",
            options=[0, 1, 2, 3],
            value=0,
            format_func=lambda x: ["Not at all","Several days","More than half","Nearly every day"][x],
            key=f"gad_{i}"
        )
        gad_scores.append(val)
    gad_total = sum(gad_scores)
    st.info(f"**GAD-7 Total: {gad_total} / 21**")

    st.markdown("---")
    epworth = st.slider("Epworth Sleepiness Score", 0, 24, 6,
                        help="0=No daytime sleepiness, 24=Severe sleepiness")

    assess_btn = st.button("🔍 Run Risk Assessment", use_container_width=True, type="primary")

# ── Main Content ──────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🧪 Risk Assessment", "👥 Population Clusters", "📈 IAPT Forecast"])

# ─ Tab 1: Risk Assessment ─────────────────────────────────────────────────────
with tab1:
    if not assess_btn:
        st.info("👈 Complete the patient intake form on the left and click **Run Risk Assessment**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""<div class="metric-card"><h4>PHQ-9 Scale</h4>
            <p>0–4: None–Minimal<br>5–9: Mild<br>10–14: Moderate<br>15–19: Moderately Severe<br>20–27: Severe</p></div>""",
            unsafe_allow_html=True)
        with col2:
            st.markdown("""<div class="metric-card"><h4>GAD-7 Scale</h4>
            <p>0–4: Minimal<br>5–9: Mild<br>10–14: Moderate<br>15–21: Severe</p></div>""",
            unsafe_allow_html=True)
        with col3:
            st.markdown("""<div class="metric-card"><h4>About This Model</h4>
            <p>Gradient Boosting Classifier<br>Trained on 783 real patients<br>Accuracy: 95%<br>Features: PHQ, GAD, Age, BMI, Sleep</p></div>""",
            unsafe_allow_html=True)
    else:
        payload = {
            "phq_score": phq_total,
            "gad_score": gad_total,
            "age": age,
            "bmi": bmi,
            "epworth_score": epworth,
            "gender": gender
        }

        with st.spinner("Running AI risk model..."):
            try:
                r1 = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
                r2 = requests.post(f"{API_URL}/cluster", json=payload, timeout=10)
                r1.raise_for_status()
                r2.raise_for_status()
                result  = r1.json()
                cluster = r2.json()
            except Exception as e:
                st.error(f"❌ Could not reach FastAPI backend. Make sure it's running: {e}")
                st.code("cd backend && uvicorn main:app --reload")
                st.stop()

        # Crisis alert
        if result["crisis_flag"]:
            st.markdown("""<div class="crisis-box">
            ⚠️ <strong>CRISIS FLAG RAISED</strong> — PHQ-9/GAD-7 scores indicate possible crisis.
            Contact NHS 111 or Samaritans: 116 123. Do NOT leave patient alone.
            </div>""", unsafe_allow_html=True)
            st.markdown("")

        # Risk result banner
        risk = result["risk_level"]
        css_class = {"High": "risk-high", "Moderate": "risk-moderate", "Low": "risk-low"}[risk]
        emoji     = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}[risk]

        st.markdown(f"""<div class="{css_class}">
        <h3>{emoji} Risk Level: <strong>{risk}</strong> &nbsp;|&nbsp; Confidence: {result['risk_score']}%</h3>
        <p><strong>PHQ-9:</strong> {phq_total}/27 — {result['phq_severity']} &nbsp;|&nbsp;
           <strong>GAD-7:</strong> {gad_total}/21 — {result['gad_severity']}</p>
        <p><strong>📋 Recommendation:</strong> {result['recommendation']}</p>
        {"<p><strong>✅ IAPT Referral Indicated</strong></p>" if result['iapt_referral'] else ""}
        </div>""", unsafe_allow_html=True)

        st.markdown("---")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### Risk Probability Distribution")
            proba = result["risk_probability"]
            fig = go.Figure(go.Bar(
                x=list(proba.keys()),
                y=list(proba.values()),
                marker_color=["#16a34a","#d97706","#dc2626"],
                text=[f"{v}%" for v in proba.values()],
                textposition="auto"
            ))
            fig.update_layout(
                yaxis_title="Probability (%)", xaxis_title="Risk Level",
                height=300, plot_bgcolor="white",
                margin=dict(t=20, b=20)
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Patient Summary")
            st.markdown(f"""
            | Field | Value |
            |-------|-------|
            | Age | {age} |
            | Gender | {gender} |
            | BMI | {bmi} |
            | PHQ-9 Score | {phq_total}/27 ({result['phq_severity']}) |
            | GAD-7 Score | {gad_total}/21 ({result['gad_severity']}) |
            | Sleepiness (Epworth) | {epworth}/24 |
            | IAPT Referral | {"Yes ✅" if result['iapt_referral'] else "No"} |
            | Crisis Flag | {"YES ⚠️" if result['crisis_flag'] else "No"} |
            """)

        # Cluster match
        st.markdown("---")
        st.markdown(f"#### 👥 Matched Population Cluster: **{cluster['profile_name']}**")
        st.caption(f"This patient's profile closely matches Cluster {cluster['cluster_id']} "
                   f"(n={cluster['cluster_size']} patients) — Avg PHQ: {cluster['avg_phq']}, Avg GAD: {cluster['avg_gad']}")

# ─ Tab 2: Population Clusters ─────────────────────────────────────────────────
with tab2:
    st.markdown("### 👥 Population Cluster Analysis")
    st.caption("K-Means clustering applied to 783 real patients from PHQ-9/GAD-7 dataset · 5 distinct clinical profiles identified")

    try:
        r = requests.get(f"{API_URL}/clusters/summary", timeout=5)
        profiles = r.json()
        df_profiles = pd.DataFrame(profiles)

        col1, col2 = st.columns([1, 1])

        with col1:
            fig = px.scatter(
                df_profiles,
                x="phq_score", y="gad_score",
                size="count", color="Profile_Name",
                hover_data=["PHQ_Severity","GAD_Severity","count"],
                title="Cluster Centroids — PHQ-9 vs GAD-7",
                labels={"phq_score": "Avg PHQ-9 Score", "gad_score": "Avg GAD-7 Score"}
            )
            fig.update_layout(height=400, plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = go.Figure(go.Bar(
                x=df_profiles["Profile_Name"],
                y=df_profiles["count"],
                marker_color=["#16a34a","#3b82f6","#d97706","#8b5cf6","#dc2626"],
                text=df_profiles["count"],
                textposition="auto"
            ))
            fig2.update_layout(
                title="Cluster Sizes", xaxis_tickangle=-20,
                yaxis_title="Patients", height=400,
                plot_bgcolor="white", margin=dict(b=80)
            )
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Cluster Profile Details")
        st.dataframe(
            df_profiles[["Cluster","Profile_Name","phq_score","gad_score","age","bmi","PHQ_Severity","GAD_Severity","count"]].rename(columns={
                "phq_score":"Avg PHQ","gad_score":"Avg GAD","age":"Avg Age","bmi":"Avg BMI","count":"Patients"
            }),
            use_container_width=True, hide_index=True
        )
    except Exception as e:
        st.error(f"Could not load cluster data. Ensure API is running. {e}")

# ─ Tab 3: IAPT Forecast ───────────────────────────────────────────────────────
with tab3:
    st.markdown("### 📈 IAPT Referral Demand Forecast")
    st.caption("ARIMA model trained on NHS IAPT monthly referral data · 3-month forecast with 95% confidence intervals")

    try:
        r = requests.get(f"{API_URL}/forecast", timeout=5)
        data = r.json()

        past_df   = pd.DataFrame(data["past_12_months"])
        future_df = pd.DataFrame(data["forecast_3_months"])
        future_df["Month"] = pd.to_datetime(future_df["Month"])

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=past_df["Month"], y=past_df["Referrals_Received"],
            mode="lines+markers", name="Historical Referrals",
            line=dict(color="#3b82f6", width=2),
            marker=dict(size=6)
        ))

        fig.add_trace(go.Scatter(
            x=future_df["Month"].dt.strftime("%b-%y"),
            y=future_df["Forecast"],
            mode="lines+markers", name="Forecast",
            line=dict(color="#dc2626", width=2, dash="dash"),
            marker=dict(size=8, symbol="diamond")
        ))

        fig.add_trace(go.Scatter(
            x=list(future_df["Month"].dt.strftime("%b-%y")) + list(future_df["Month"].dt.strftime("%b-%y"))[::-1],
            y=list(future_df["CI_Upper"]) + list(future_df["CI_Lower"])[::-1],
            fill="toself", fillcolor="rgba(220,38,38,0.1)",
            line=dict(color="rgba(255,255,255,0)"),
            name="95% Confidence Interval"
        ))

        fig.update_layout(
            title="NHS IAPT Monthly Referrals — Historical + 3-Month Forecast",
            xaxis_title="Month", yaxis_title="Referrals",
            height=450, plot_bgcolor="white",
            legend=dict(orientation="h", y=-0.2)
        )
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        for i, (col, row) in enumerate(zip([col1, col2, col3], future_df.iterrows())):
            _, r = row
            with col:
                st.metric(
                    label=r["Month"].strftime("%B %Y"),
                    value=f"{int(r['Forecast']):,}",
                    delta=f"CI: {int(r['CI_Lower']):,} – {int(r['CI_Upper']):,}"
                )

        st.markdown("---")
        st.markdown("#### Historical IAPT Referral Data (Last 12 months)")
        st.dataframe(past_df, use_container_width=True, hide_index=True)

    except Exception as e:
        st.error(f"Could not load forecast data. Ensure API is running. {e}")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("🏥 MindRisk v1.0 · AI Mental Health UK · NHS-aligned · For clinical decision support only · Not a substitute for professional judgement")