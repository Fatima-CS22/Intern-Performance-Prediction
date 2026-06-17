import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(
    page_title="InternIQ · Performance Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=Inter:wght@300;400;500;600&display=swap');

/* Reset & base */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0A0F1E !important;
    color: #F0F2FF !important;
    font-family: 'Inter', sans-serif;
}

[data-testid="stSidebar"] {
    background: #0D1426 !important;
    border-right: 1px solid rgba(108,99,255,0.2);
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* Headings */
h1,h2,h3,h4 { font-family: 'Syne', sans-serif !important; }

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600;
    color: #8B92B8 !important;
    border-bottom: 2px solid transparent !important;
    background: transparent !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #6C63FF !important;
    border-bottom: 2px solid #6C63FF !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(108,99,255,0.08);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 12px;
    padding: 16px 20px;
}
[data-testid="stMetricLabel"] { color: #8B92B8 !important; font-size: 12px !important; }
[data-testid="stMetricValue"] { color: #F0F2FF !important; font-family: 'Syne', sans-serif !important; }

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #8B5CF6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 12px 28px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(108,99,255,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(108,99,255,0.55) !important;
}

/* Sliders */
[data-testid="stSlider"] > div > div > div { background: #6C63FF !important; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(108,99,255,0.06) !important;
    border: 1.5px dashed rgba(108,99,255,0.4) !important;
    border-radius: 12px !important;
}

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }

/* Radio */
[data-testid="stRadio"] label { color: #B0B8D8 !important; }

/* Selectbox */
[data-testid="stSelectbox"] > div { background: #131929 !important; border-color: rgba(108,99,255,0.3) !important; }

/* Download button */
.stDownloadButton > button {
    background: rgba(0,212,170,0.12) !important;
    color: #00D4AA !important;
    border: 1px solid rgba(0,212,170,0.4) !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0A0F1E; }
::-webkit-scrollbar-thumb { background: #6C63FF; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    base = os.path.dirname(__file__)
    rf       = joblib.load(os.path.join(base, 'random_forest_model.pkl'))
    xgb_m    = joblib.load(os.path.join(base, 'xgboost_model.pkl'))
    features = joblib.load(os.path.join(base, 'features.pkl'))
    metrics  = joblib.load(os.path.join(base, 'model_metrics.pkl'))
    return rf, xgb_m, features, metrics

rf_model, xgb_model, FEATURES, METRICS = load_models()

# ── Helpers ───────────────────────────────────────────────────
def engineer_features(df):
    df = df.copy()
    for col in ['Completion_Time', 'Feedback_Rating', 'Attendance']:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    df['Completion_Efficiency']       = df['Attendance'] / (df['Completion_Time'] + 1)
    df['Overall_Feedback_Attendance'] = df['Feedback_Rating'] * df['Attendance'] / 100
    df['Feedback_per_Hour']           = df['Feedback_Rating'] / (df['Completion_Time'] + 1)
    return df

CAT_CONFIG = {
    "🏆 Excel":    {"color": "#00D4AA", "glow": "rgba(0,212,170,0.35)",   "bg": "rgba(0,212,170,0.10)"},
    "✅ Good":     {"color": "#6C63FF", "glow": "rgba(108,99,255,0.35)",  "bg": "rgba(108,99,255,0.10)"},
    "⚠️ Average":  {"color": "#FFB547", "glow": "rgba(255,181,71,0.35)",  "bg": "rgba(255,181,71,0.10)"},
    "🔴 Struggle": {"color": "#FF4757", "glow": "rgba(255,71,87,0.35)",   "bg": "rgba(255,71,87,0.10)"},
}

def categorize(score):
    if score >= 80:   return "🏆 Excel"
    elif score >= 60: return "✅ Good"
    elif score >= 45: return "⚠️ Average"
    else:             return "🔴 Struggle"

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#B0B8D8"),
    margin=dict(t=40, b=20, l=20, r=20),
)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:20px 0 10px'>
        <div style='font-family:Syne,sans-serif;font-size:22px;font-weight:800;
                    background:linear-gradient(135deg,#6C63FF,#00D4AA);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent'>
            ⚡ InternIQ
        </div>
        <div style='color:#8B92B8;font-size:12px;margin-top:4px'>Performance Analytics</div>
    </div>
    <hr style='border-color:rgba(108,99,255,0.2);margin:10px 0 20px'>
    """, unsafe_allow_html=True)

    model_choice = st.radio("**Active Model**", ["XGBoost", "Random Forest"],
                            help="Switch between trained models")
    active_model = xgb_model if model_choice == "XGBoost" else rf_model
    m = METRICS['xgb'] if model_choice == "XGBoost" else METRICS['rf']

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<div style='color:#8B92B8;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px'>Model Metrics</div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    col1.metric("R² Score", f"{m['r2']:.3f}")
    col2.metric("RMSE", f"{m['rmse']:.2f}")
    st.metric("MAE", f"{m['mae']:.2f}", delta="Mean Absolute Error")

    st.markdown("""
    <hr style='border-color:rgba(108,99,255,0.2);margin:20px 0'>
    <div style='color:#8B92B8;font-size:11px'>
        📌 Models trained on <b style='color:#F0F2FF'>10,000</b> intern records<br><br>
        🔬 Features: Completion Time · Feedback · Attendance + 3 engineered
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div style='padding:30px 0 10px'>
    <div style='font-family:Syne,sans-serif;font-size:36px;font-weight:800;
                background:linear-gradient(135deg,#F0F2FF 40%,#6C63FF);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                line-height:1.1'>
        Intern Performance<br>Prediction Dashboard
    </div>
    <div style='color:#8B92B8;margin-top:10px;font-size:14px'>
        ML-powered analytics · Random Forest & XGBoost · Real-time predictions
    </div>
</div>
<hr style='border-color:rgba(108,99,255,0.15);margin:10px 0 30px'>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📁  Batch Analysis", "🔍  Single Prediction"])

# ══════════════════════════════════════════════════════════════
# TAB 1 — Batch
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Upload Intern Dataset")
    st.markdown("<div style='color:#8B92B8;font-size:13px;margin-bottom:16px'>CSV must contain: <code>Intern_ID</code>, <code>Completion_Time</code>, <code>Feedback_Rating</code>, <code>Attendance</code></div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    if uploaded:
        df = pd.read_csv(uploaded)
        df = engineer_features(df)
        df['Predicted_Score']    = active_model.predict(df[FEATURES])
        df['Predicted_Category'] = df['Predicted_Score'].apply(categorize)

        total = len(df)
        cats  = df['Predicted_Category'].value_counts()

        # ── KPI strip ─────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        k0, k1, k2, k3, k4 = st.columns(5)
        kpis = [
            ("Total Interns",  total,                          "#6C63FF"),
            ("🏆 Excel",       cats.get("🏆 Excel",    0),     "#00D4AA"),
            ("✅ Good",        cats.get("✅ Good",     0),      "#6C63FF"),
            ("⚠️ Average",     cats.get("⚠️ Average",  0),     "#FFB547"),
            ("🔴 Struggle",    cats.get("🔴 Struggle", 0),     "#FF4757"),
        ]
        for col, (label, val, color) in zip([k0,k1,k2,k3,k4], kpis):
            col.markdown(f"""
            <div style='background:rgba(255,255,255,0.03);border:1px solid {color}33;
                        border-radius:12px;padding:16px;text-align:center'>
                <div style='color:#8B92B8;font-size:11px;text-transform:uppercase;
                            letter-spacing:0.8px;margin-bottom:6px'>{label}</div>
                <div style='font-family:Syne,sans-serif;font-size:28px;font-weight:800;
                            color:{color}'>{val}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        # ── Charts ────────────────────────────────────────────
        ch1, ch2 = st.columns(2)

        with ch1:
            st.markdown("#### Category Breakdown")
            order  = ["🏆 Excel","✅ Good","⚠️ Average","🔴 Struggle"]
            colors = ["#00D4AA","#6C63FF","#FFB547","#FF4757"]
            counts_df = df['Predicted_Category'].value_counts().reindex(order, fill_value=0).reset_index()
            counts_df.columns = ['Category','Count']
            fig = px.bar(counts_df, x='Category', y='Count',
                         color='Category', text='Count',
                         color_discrete_sequence=colors)
            fig.update_traces(textposition='outside', textfont_color="#F0F2FF",
                              marker_line_width=0, width=0.55)
            fig.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                              xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""),
                              yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title=""))
            st.plotly_chart(fig, use_container_width=True)

        with ch2:
            st.markdown("#### Score Distribution")
            fig2 = px.histogram(df, x='Predicted_Score', nbins=35,
                                color_discrete_sequence=['#6C63FF'])
            fig2.update_traces(marker_line_width=0, opacity=0.85)
            fig2.add_vline(x=df['Predicted_Score'].mean(), line_dash="dash",
                           line_color="#00D4AA", annotation_text="Mean",
                           annotation_font_color="#00D4AA")
            fig2.update_layout(**PLOTLY_LAYOUT, height=320,
                               xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Predicted Score"),
                               yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Count"))
            st.plotly_chart(fig2, use_container_width=True)

        # ── Scatter ───────────────────────────────────────────
        if 'Performance_Score' in df.columns:
            st.markdown("#### Actual vs Predicted Score")
            fig3 = px.scatter(df, x='Performance_Score', y='Predicted_Score',
                              color='Predicted_Category',
                              color_discrete_map={k: v['color'] for k,v in CAT_CONFIG.items()},
                              opacity=0.6, hover_data=['Intern_ID'])
            min_v = min(df['Performance_Score'].min(), df['Predicted_Score'].min())
            max_v = max(df['Performance_Score'].max(), df['Predicted_Score'].max())
            fig3.add_trace(go.Scatter(x=[min_v,max_v], y=[min_v,max_v],
                                      mode='lines', line=dict(color='#8B92B8', dash='dash', width=1),
                                      name='Perfect fit', showlegend=True))
            fig3.update_layout(**PLOTLY_LAYOUT, height=370,
                               xaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Actual Score"),
                               yaxis=dict(gridcolor="rgba(255,255,255,0.05)", title="Predicted Score"))
            st.plotly_chart(fig3, use_container_width=True)

        # ── Feature importance ────────────────────────────────
        st.markdown("#### Feature Importance")
        feat_labels = ['Completion Time','Feedback Rating','Attendance',
                       'Completion Efficiency','Feedback × Attendance','Feedback/Hour']
        imp_df = pd.DataFrame({'Feature': feat_labels,
                               'Importance': active_model.feature_importances_})\
                   .sort_values('Importance')
        fig4 = px.bar(imp_df, x='Importance', y='Feature', orientation='h',
                      color='Importance', color_continuous_scale=[[0,'#2D1B69'],[1,'#6C63FF']])
        fig4.update_layout(**PLOTLY_LAYOUT, height=300, coloraxis_showscale=False,
                           xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
                           yaxis=dict(gridcolor="rgba(255,255,255,0.05)"))
        st.plotly_chart(fig4, use_container_width=True)

        # ── Tables ────────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("""<div style='color:#00D4AA;font-family:Syne,sans-serif;
                           font-weight:700;font-size:15px;margin-bottom:10px'>
                           🏆 Top Performers — Excel</div>""", unsafe_allow_html=True)
            top = df[df['Predicted_Category']=="🏆 Excel"]\
                    [['Intern_ID','Attendance','Feedback_Rating','Predicted_Score']]\
                    .sort_values('Predicted_Score', ascending=False).head(10)\
                    .reset_index(drop=True)
            top['Predicted_Score'] = top['Predicted_Score'].round(1)
            st.dataframe(top, use_container_width=True)

        with t2:
            st.markdown("""<div style='color:#FF4757;font-family:Syne,sans-serif;
                           font-weight:700;font-size:15px;margin-bottom:10px'>
                           🔴 Needs Attention — Struggle</div>""", unsafe_allow_html=True)
            bot = df[df['Predicted_Category']=="🔴 Struggle"]\
                    [['Intern_ID','Attendance','Feedback_Rating','Predicted_Score']]\
                    .sort_values('Predicted_Score').head(10)\
                    .reset_index(drop=True)
            bot['Predicted_Score'] = bot['Predicted_Score'].round(1)
            st.dataframe(bot, use_container_width=True)

        # ── Download ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        out_cols = ['Intern_ID','Completion_Time','Feedback_Rating',
                    'Attendance','Predicted_Score','Predicted_Category']
        csv_bytes = df[out_cols].to_csv(index=False).encode('utf-8')
        st.download_button("⬇️  Download Full Predictions CSV", csv_bytes,
                           "intern_predictions.csv", "text/csv", use_container_width=True)
    else:
        # Empty state
        st.markdown("""
        <div style='text-align:center;padding:80px 40px;
                    background:rgba(108,99,255,0.04);border:1.5px dashed rgba(108,99,255,0.25);
                    border-radius:16px;margin-top:20px'>
            <div style='font-size:48px;margin-bottom:16px'>📂</div>
            <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:700;
                        color:#F0F2FF;margin-bottom:8px'>Drop your CSV above</div>
            <div style='color:#8B92B8;font-size:13px;max-width:400px;margin:0 auto'>
                Upload the intern dataset to run batch predictions across all interns instantly.
                Results include category labels, score distributions, and a downloadable report.
            </div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 2 — Single Prediction
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Predict a Single Intern")
    st.markdown("<div style='color:#8B92B8;font-size:13px;margin-bottom:24px'>Adjust the sliders to match the intern's profile and get an instant prediction.</div>", unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("<div style='color:#8B92B8;font-size:12px;margin-bottom:4px'>⏱ COMPLETION TIME (hours)</div>", unsafe_allow_html=True)
        completion_time = st.slider("", 0.5, 10.0, 4.0, 0.1, key="ct", label_visibility="collapsed")
    with s2:
        st.markdown("<div style='color:#8B92B8;font-size:12px;margin-bottom:4px'>⭐ FEEDBACK RATING (1–5)</div>", unsafe_allow_html=True)
        feedback_rating = st.slider("", 1.0, 5.0, 3.5, 0.1, key="fr", label_visibility="collapsed")
    with s3:
        st.markdown("<div style='color:#8B92B8;font-size:12px;margin-bottom:4px'>📅 ATTENDANCE (%)</div>", unsafe_allow_html=True)
        attendance = st.slider("", 0.0, 100.0, 75.0, 0.5, key="att", label_visibility="collapsed")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  Run Prediction", use_container_width=True)

    if predict_btn:
        eff   = attendance / (completion_time + 1)
        ofa   = feedback_rating * attendance / 100
        fph   = feedback_rating / (completion_time + 1)
        inp   = np.array([[completion_time, feedback_rating, attendance, eff, ofa, fph]])
        score = float(active_model.predict(inp)[0])
        cat   = categorize(score)
        cfg   = CAT_CONFIG[cat]

        st.markdown("<br>", unsafe_allow_html=True)
        res1, res2 = st.columns([1, 1.4])

        with res1:
            # Score card with glow
            st.markdown(f"""
            <div style='background:{cfg["bg"]};border:1.5px solid {cfg["color"]}55;
                        border-radius:16px;padding:36px 28px;text-align:center;
                        box-shadow:0 0 40px {cfg["glow"]};'>
                <div style='font-family:Syne,sans-serif;font-size:14px;font-weight:600;
                            color:{cfg["color"]};letter-spacing:2px;text-transform:uppercase;
                            margin-bottom:12px'>Prediction Result</div>
                <div style='font-family:Syne,sans-serif;font-size:64px;font-weight:800;
                            color:{cfg["color"]};line-height:1;margin-bottom:8px'>
                    {score:.1f}
                </div>
                <div style='color:#8B92B8;font-size:12px;margin-bottom:20px'>out of 100</div>
                <div style='background:{cfg["color"]}22;border:1px solid {cfg["color"]}66;
                            border-radius:30px;padding:10px 24px;display:inline-block;
                            font-family:Syne,sans-serif;font-weight:700;
                            color:{cfg["color"]};font-size:18px'>
                    {cat}
                </div>
            </div>""", unsafe_allow_html=True)

        with res2:
            # Gauge chart
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number=dict(font=dict(family="Syne", color=cfg["color"], size=36)),
                gauge=dict(
                    axis=dict(range=[0,100], tickcolor="#8B92B8",
                              tickfont=dict(color="#8B92B8", size=11)),
                    bar=dict(color=cfg["color"], thickness=0.25),
                    bgcolor="rgba(0,0,0,0)",
                    bordercolor="rgba(0,0,0,0)",
                    steps=[
                        dict(range=[0,45],  color="rgba(255,71,87,0.15)"),
                        dict(range=[45,60], color="rgba(255,181,71,0.12)"),
                        dict(range=[60,80], color="rgba(108,99,255,0.12)"),
                        dict(range=[80,100],color="rgba(0,212,170,0.15)"),
                    ],
                    threshold=dict(line=dict(color=cfg["color"], width=3), value=score)
                )
            ))
            fig_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", color="#B0B8D8"),
                height=260, margin=dict(t=20,b=0,l=30,r=30)
            )
            st.plotly_chart(fig_g, use_container_width=True)

        # Input breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Input & Derived Features")
        feat_rows = [
            ("⏱ Completion Time",          f"{completion_time} hrs",  "Raw input"),
            ("⭐ Feedback Rating",           f"{feedback_rating} / 5", "Raw input"),
            ("📅 Attendance",               f"{attendance}%",          "Raw input"),
            ("📐 Completion Efficiency",    f"{eff:.3f}",              "Attendance ÷ (Time + 1)"),
            ("🔗 Feedback × Attendance",    f"{ofa:.3f}",              "Rating × Attendance ÷ 100"),
            ("⚡ Feedback per Hour",         f"{fph:.3f}",             "Rating ÷ (Time + 1)"),
        ]
        cols = st.columns(3)
        for i, (label, val, formula) in enumerate(feat_rows):
            with cols[i % 3]:
                st.markdown(f"""
                <div style='background:rgba(255,255,255,0.03);border:1px solid rgba(108,99,255,0.2);
                            border-radius:10px;padding:14px;margin-bottom:12px'>
                    <div style='color:#8B92B8;font-size:11px;margin-bottom:4px'>{label}</div>
                    <div style='font-family:Syne,sans-serif;font-size:20px;font-weight:700;
                                color:#F0F2FF'>{val}</div>
                    <div style='color:#6C63FF;font-size:10px;margin-top:4px'>{formula}</div>
                </div>""", unsafe_allow_html=True)

        # Recommendation
        recs = {
            "🏆 Excel":    ("This intern is on track to be a top performer.", "#00D4AA",
                            "Consider assigning them to high-impact projects and fast-tracking their responsibilities."),
            "✅ Good":     ("This intern is performing well above average.", "#6C63FF",
                            "Provide stretch assignments to help them move toward excellence."),
            "⚠️ Average":  ("This intern is meeting baseline expectations.", "#FFB547",
                            "Schedule a feedback session and identify specific areas for improvement."),
            "🔴 Struggle": ("This intern needs immediate support.", "#FF4757",
                            "Arrange mentorship and review workload — early intervention is key."),
        }
        headline, color, advice = recs[cat]
        st.markdown(f"""
        <div style='background:rgba(255,255,255,0.02);border-left:3px solid {color};
                    border-radius:0 10px 10px 0;padding:16px 20px;margin-top:8px'>
            <div style='font-family:Syne,sans-serif;font-weight:700;
                        color:{color};margin-bottom:6px'>{headline}</div>
            <div style='color:#B0B8D8;font-size:13px'>{advice}</div>
        </div>""", unsafe_allow_html=True)