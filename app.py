"""
Streamlit Frontend — Multivariate Weather Forecasting with Attention
====================================================================
Pages:
  1. 🏠 Home             — Project overview & live metric cards
  2. 📊 EDA Explorer     — Interactive EDA visualizations
  3. 🔮 Live Forecast    — Real-time model inference on any test window
  4. 🧠 Attention Maps   — Temporal attention heatmaps & case studies
  5. 🏆 Model Comparison — Side-by-side benchmark table + radar chart
  6. 🔬 Ablation Study   — Hyperparameter tuning results
"""

import sys
import warnings
import pickle
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ── Project Root ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.attention import TemporalAdditiveAttention  # noqa: E402  (registers custom layer)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Weather Forecasting · Attention",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Colour Palette ────────────────────────────────────────────────────────────
ACCENT   = "#6C63FF"
ACCENT2  = "#43CFCA"
ACCENT3  = "#FF6B6B"
ACCENT4  = "#FFA552"
VAR_COLORS = {
    "T (degC)": "#FF6B6B",
    "p (mbar)": "#6C63FF",
    "rh (%)":   "#43CFCA",
    "wv (m/s)": "#FFA552",
}
MODEL_COLORS = {
    "Baseline_LSTM":     "#6C63FF",
    "Stacked_GRU":       "#43CFCA",
    "GRU_with_Attention":"#FFA552",
}

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background: #0d1117;
    color: #e6edf3;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13161d 0%, #0d1117 100%);
    border-right: 1px solid rgba(108,99,255,0.2);
}

.metric-card {
    background: linear-gradient(135deg, rgba(108,99,255,0.12) 0%, rgba(67,207,202,0.08) 100%);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 16px;
    padding: 20px 24px;
    text-align: center;
    transition: transform .2s, box-shadow .2s;
    min-height: 110px;
}
.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(108,99,255,0.2);
}
.metric-label { font-size: 12px; color: #8b949e; font-weight: 500; letter-spacing: .6px; text-transform: uppercase; margin-bottom: 8px; }
.metric-value { font-size: 28px; font-weight: 700; color: #e6edf3; }
.metric-unit  { font-size: 12px; color: #8b949e; margin-top: 4px; }

.hero {
    background: linear-gradient(135deg, #13161d 0%, #1a1f2e 50%, #13161d 100%);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 24px;
    padding: 48px;
    margin-bottom: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%; left: -50%; width: 200%; height: 200%;
    background: radial-gradient(circle at 30% 50%, rgba(108,99,255,0.08) 0%, transparent 60%),
                radial-gradient(circle at 70% 50%, rgba(67,207,202,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero h1 { font-size: 3rem; font-weight: 700; background: linear-gradient(90deg, #6C63FF, #43CFCA); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0 0 12px 0; }
.hero p  { font-size: 1.1rem; color: #8b949e; max-width: 700px; margin: 0 auto; line-height: 1.8; }

.section-header {
    font-size: 1.6rem; font-weight: 700;
    background: linear-gradient(90deg, #6C63FF, #43CFCA);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 4px;
}

.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 12px;
    font-weight: 600;
    margin: 2px;
}
.badge-purple { background: rgba(108,99,255,0.2); border: 1px solid rgba(108,99,255,0.4); color: #a5a0ff; }
.badge-teal   { background: rgba(67,207,202,0.2); border: 1px solid rgba(67,207,202,0.4); color: #6ee7e3; }
.badge-red    { background: rgba(255,107,107,0.2); border: 1px solid rgba(255,107,107,0.4); color: #ff9b9b; }
.badge-orange { background: rgba(255,165,82,0.2);  border: 1px solid rgba(255,165,82,0.4);  color: #ffc07a; }

hr { border: none; border-top: 1px solid rgba(108,99,255,0.15); margin: 24px 0; }

.forecast-box {
    background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(67,207,202,0.07));
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 16px;
    padding: 24px;
}

.stButton>button {
    background: linear-gradient(135deg, #6C63FF, #43CFCA) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    transition: opacity .2s !important;
}
.stButton>button:hover { opacity: .85 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Data & Model Loaders (Cached)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_dataset():
    return pd.read_csv(ROOT / "Dataset" / "jena_climate_hourly_cleaned.csv",
                       index_col=0, parse_dates=True)

@st.cache_data(show_spinner=False)
def load_metrics():
    return pd.read_csv(ROOT / "results" / "metrics" / "model_comparison.csv")

@st.cache_data(show_spinner=False)
def load_tuning():
    return pd.read_csv(ROOT / "results" / "metrics" / "tuning_experiments.csv")

@st.cache_resource(show_spinner=False)
def load_scaler():
    with open(ROOT / "results" / "scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_target_scaler():
    with open(ROOT / "results" / "target_scaler.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource(show_spinner=False)
def load_model(name: str):
    return joblib.load(ROOT / "results" / "models" / f"{name}.joblib")

def run_inference(model_name: str, window: np.ndarray):
    model = load_model(model_name)
    x = window[np.newaxis, :, :].astype(np.float32)
    out = model(x)
    if isinstance(out, (list, tuple)):
        preds_scaled, attn = out[0].numpy(), out[1].numpy()
    else:
        preds_scaled, attn = out.numpy(), None
    preds_phys = load_target_scaler().inverse_transform(preds_scaled)[0]
    return preds_phys, attn

TARGET_COLS   = ["T (degC)", "p (mbar)", "rh (%)", "wv (m/s)"]
TARGET_UNITS  = {"T (degC)": "°C", "p (mbar)": "mbar", "rh (%)": "%", "wv (m/s)": "m/s"}
TARGET_ICONS  = {"T (degC)": "🌡️", "p (mbar)": "🧭", "rh (%)": "💧", "wv (m/s)": "💨"}
CYCLICAL_COLS = ["Day_sin", "Day_cos", "Year_sin", "Year_cos"]
ALL_FEATURES  = TARGET_COLS + CYCLICAL_COLS
MODEL_NAMES   = ["Baseline_LSTM", "Stacked_GRU", "GRU_with_Attention"]
MODEL_LABELS  = {
    "Baseline_LSTM":     "Baseline LSTM",
    "Stacked_GRU":       "Stacked GRU",
    "GRU_with_Attention":"GRU + Attention ✨"
}
FIGURES_DIR = ROOT / "figures"

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar Navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 16px 0 8px'>
        <div style='font-size:2.8rem'>🌦️</div>
        <div style='font-size:1rem; font-weight:700; color:#a5a0ff; letter-spacing:.5px'>Weather Forecasting</div>
        <div style='font-size:.75rem; color:#8b949e'>with Temporal Attention</div>
    </div>
    <hr style='border-color:rgba(108,99,255,0.2); margin:8px 0 16px'>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        options=["🏠 Home", "📊 EDA Explorer", "🔮 Live Forecast",
                 "🧠 Attention Maps", "🏆 Model Comparison", "🔬 Ablation Study"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='border-color:rgba(108,99,255,0.2)'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:.72rem; color:#8b949e; padding: 0 4px'>
        <b style='color:#a5a0ff'>Dataset:</b> Jena Climate (2009–2016)<br>
        <b style='color:#a5a0ff'>Models:</b> LSTM · GRU · GRU+Attention<br>
        <b style='color:#a5a0ff'>Targets:</b> T · p · rh · wv<br>
        <b style='color:#a5a0ff'>Framework:</b> TensorFlow 2.16
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("""
    <div class='hero'>
        <h1>🌦️ Weather Forecasting<br>with Temporal Attention</h1>
        <p>An end-to-end deep learning system for <strong>joint multivariate forecasting</strong> of
        Temperature, Pressure, Humidity &amp; Wind Velocity using the Jena Climate Dataset.
        Benchmarks Baseline LSTM vs Stacked GRU vs GRU + Bahdanau Additive Attention.</p>
        <div style='margin-top:20px'>
            <span class='badge badge-purple'>TensorFlow 2.16</span>
            <span class='badge badge-teal'>Keras 3</span>
            <span class='badge badge-red'>Scikit-Learn</span>
            <span class='badge badge-orange'>Plotly</span>
            <span class='badge badge-purple'>70,129 Records</span>
            <span class='badge badge-teal'>Zero Data Leakage</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    metrics_df = load_metrics()

    st.markdown("<div class='section-header'>📈 Best Model Performance (GRU + Attention)</div>", unsafe_allow_html=True)
    st.markdown("<div style='color:#8b949e; margin-bottom:16px'>Test set · Physical units · Zero-leakage evaluation</div>", unsafe_allow_html=True)

    attn_df = metrics_df[metrics_df["Model"] == "GRU_with_Attention"].reset_index(drop=True)
    cols = st.columns(4)
    for i, row in attn_df.iterrows():
        var = row["Target Variable"]
        with cols[i % 4]:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{TARGET_ICONS.get(var,'')} {var}</div>
                <div class='metric-value'>R² {row['Test_R2_Score']:.4f}</div>
                <div class='metric-unit'>MAE {row['Test_MAE']:.4f} {row['Unit']} · RMSE {row['Test_RMSE']:.4f} {row['Unit']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.markdown("<div class='section-header'>🧠 Architecture Benchmark</div>", unsafe_allow_html=True)
        arch_data = {
            "Architecture": ["Baseline LSTM", "Stacked GRU", "GRU + Attention"],
            "Test Mean RMSE": [1.165, 1.16, 1.158],
            "Params": [32292, 24804, 25892],
        }
        arch_df = pd.DataFrame(arch_data)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=arch_df["Architecture"], y=arch_df["Test Mean RMSE"],
            marker=dict(color=list(MODEL_COLORS.values()), line=dict(width=0)),
            text=arch_df["Test Mean RMSE"].round(4), textposition="outside",
            textfont=dict(color="#e6edf3", size=12),
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3", family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.05)"),
            yaxis=dict(title="Test Mean RMSE", gridcolor="rgba(255,255,255,0.05)"),
            margin=dict(l=0, r=0, t=20, b=0), height=280,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("<div class='section-header'>📐 Dataset Statistics</div>", unsafe_allow_html=True)
        stats = {
            "Total Records": "70,129",
            "Date Range": "2009 – 2016",
            "Sampling Rate": "1 hour",
            "Train / Val / Test": "70% / 15% / 15%",
            "Input Features": "8 (4 targets + 4 cyclical)",
            "Lookback Window": "72 hours (3 days)",
            "Forecast Horizon": "t+1 hour",
        }
        for k, v in stats.items():
            col_a, col_b = st.columns([2, 3])
            col_a.markdown(f"<span style='color:#8b949e; font-size:.85rem'>{k}</span>", unsafe_allow_html=True)
            col_b.markdown(f"<span style='color:#e6edf3; font-weight:600; font-size:.85rem'>{v}</span>", unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔄 Pipeline Steps</div>", unsafe_allow_html=True)
    steps = [
        ("1️⃣", "Data Cleaning", "Sensor anomaly removal, hourly resampling, linear gap-fill"),
        ("2️⃣", "Feature Engineering", "Cyclical Fourier embeddings (Day/Year sin+cos)"),
        ("3️⃣", "Zero-Leakage Split", "70/15/15 chronological — scaler fit on train only"),
        ("4️⃣", "Sequence Generation", "Sliding-window tf.data.Dataset (72h → t+1h)"),
        ("5️⃣", "Model Training", "LSTM · Stacked GRU · GRU+Attention + EarlyStopping"),
        ("6️⃣", "Evaluation", "Per-variable MAE, RMSE, R² in physical units"),
        ("7️⃣", "Ablation Study", "Window, depth, attention mechanism comparisons"),
        ("8️⃣", "Serialization", ".keras · .joblib · .pkl with forward-pass verification"),
    ]
    cols_p = st.columns(4)
    for idx, (num, title, desc) in enumerate(steps):
        with cols_p[idx % 4]:
            st.markdown(f"""
            <div class='metric-card' style='text-align:left; min-height:130px; margin-bottom:8px'>
                <div style='font-size:1.3rem'>{num}</div>
                <div style='font-weight:600; color:#a5a0ff; margin:6px 0 4px; font-size:.9rem'>{title}</div>
                <div style='color:#8b949e; font-size:.78rem; line-height:1.5'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — EDA EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 EDA Explorer":
    st.markdown("<div class='section-header'>📊 Exploratory Data Analysis</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e'>Interactive exploration of the Jena Climate Dataset (2009–2016)</p>", unsafe_allow_html=True)

    try:
        df = load_dataset()

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📅 Yearly Trends", "🌡️ Correlations", "🕐 Diurnal", "⚗️ Phase Coupling", "📦 Distributions"
        ])

        with tab1:
            col_sel = st.selectbox("Select variable:", TARGET_COLS, key="eda_yearly_var")
            rolling = st.slider("Rolling average window (hours):", 24, 720, 168, 24, key="eda_roll")
            ts = df[col_sel].copy()
            ts_roll = ts.rolling(rolling, center=True).mean()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=ts, mode="lines", name="Raw",
                line=dict(color=VAR_COLORS[col_sel], width=0.6), opacity=0.3))
            fig.add_trace(go.Scatter(x=df.index, y=ts_roll, mode="lines",
                name=f"{rolling}h Rolling Mean", line=dict(color=VAR_COLORS[col_sel], width=2.5)))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
                font=dict(color="#e6edf3", family="Inter"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Date"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=f"{col_sel} ({TARGET_UNITS[col_sel]})"),
                legend=dict(bgcolor="rgba(0,0,0,0)"), height=450, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            corr = df[TARGET_COLS].corr()
            fig = go.Figure(go.Heatmap(
                z=corr.values, x=corr.columns, y=corr.index,
                colorscale=[[0,"#FF6B6B"],[0.5,"#13161d"],[1,"#6C63FF"]],
                text=corr.round(3).values, texttemplate="%{text}",
                textfont=dict(size=14, color="white"), zmin=-1, zmax=1))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#e6edf3", family="Inter"), height=420, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.info("💡 Strong negative T–rh correlation (r ≈ −0.57): inverse Clausius-Clapeyron relationship.")

        with tab3:
            diurnal = df[TARGET_COLS].copy()
            diurnal["hour"] = df.index.hour
            hourly_mean = diurnal.groupby("hour").mean()
            hourly_std  = diurnal.groupby("hour").std()
            fig = make_subplots(rows=2, cols=2, subplot_titles=TARGET_COLS,
                vertical_spacing=0.12, horizontal_spacing=0.08)
            positions = [(1,1),(1,2),(2,1),(2,2)]
            for (r, c), col in zip(positions, TARGET_COLS):
                mean_v = hourly_mean[col]
                std_v  = hourly_std[col]
                hrs_x  = hourly_mean.index
                fill_color = VAR_COLORS[col] + "28"
                fig.add_trace(go.Scatter(
                    x=list(hrs_x) + list(hrs_x[::-1]),
                    y=list(mean_v + std_v) + list((mean_v - std_v)[::-1]),
                    fill="toself", fillcolor=fill_color,
                    line=dict(width=0), showlegend=False, mode="lines"), row=r, col=c)
                fig.add_trace(go.Scatter(
                    x=hrs_x, y=mean_v, mode="lines", name=col,
                    line=dict(color=VAR_COLORS[col], width=2.5)), row=r, col=c)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
                font=dict(color="#e6edf3", family="Inter"), height=520,
                margin=dict(l=0,r=0,t=40,b=0), legend=dict(bgcolor="rgba(0,0,0,0)"))
            for row in range(1,3):
                for col in range(1,3):
                    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)",
                        title_text="Hour" if row==2 else "", row=row, col=col)
                    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=row, col=col)
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            season_map = {1:"Winter", 2:"Spring", 3:"Summer", 4:"Autumn"}
            df_plot = df[TARGET_COLS].copy()
            df_plot["season"] = ((df.index.month % 12 // 3) + 1).map(season_map)
            x_var = st.selectbox("X axis:", TARGET_COLS, index=0, key="phase_x")
            y_var = st.selectbox("Y axis:", TARGET_COLS, index=2, key="phase_y")
            season_colors = {"Winter":"#6C63FF","Spring":"#43CFCA","Summer":"#FF6B6B","Autumn":"#FFA552"}
            sample = df_plot.sample(min(5000, len(df_plot)), random_state=42)
            fig = go.Figure()
            for season, grp in sample.groupby("season"):
                fig.add_trace(go.Scatter(
                    x=grp[x_var], y=grp[y_var], mode="markers", name=season,
                    marker=dict(color=season_colors.get(season, ACCENT), size=4, opacity=0.55)))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
                font=dict(color="#e6edf3", family="Inter"),
                xaxis=dict(title=f"{x_var} ({TARGET_UNITS[x_var]})", gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(title=f"{y_var} ({TARGET_UNITS[y_var]})", gridcolor="rgba(255,255,255,0.06)"),
                legend=dict(bgcolor="rgba(0,0,0,0)"), height=460, margin=dict(l=0,r=0,t=20,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with tab5:
            fig = go.Figure()
            for col in TARGET_COLS:
                for s_id, s_name in zip([1,2,3,4], ["Winter","Spring","Summer","Autumn"]):
                    mask = (df.index.month % 12 // 3 + 1) == s_id
                    fig.add_trace(go.Box(
                        y=df.loc[mask, col], name=f"{col}<br>{s_name}",
                        marker_color=VAR_COLORS[col], boxmean=True,
                        line_width=1.5, marker_size=3))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
                font=dict(color="#e6edf3", family="Inter"),
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
                height=500, margin=dict(l=0,r=0,t=20,b=0), legend_visible=False)
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"EDA failed: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — LIVE FORECAST
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Live Forecast":
    st.markdown("<div class='section-header'>🔮 Live Forecast</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e'>Select a time window from the test set and run real-time inference</p>", unsafe_allow_html=True)

    try:
        df = load_dataset()
        from src.features import add_cyclical_time_features
        df_feat = add_cyclical_time_features(df)
        scaler = load_scaler()

        n = len(df_feat)
        test_start_idx = int(n * 0.85)
        test_df = df_feat.iloc[test_start_idx:]

        col_ctrl, col_out = st.columns([1, 2], gap="large")

        with col_ctrl:
            st.markdown("#### ⚙️ Controls")
            model_choice = st.selectbox("Select model:", MODEL_NAMES,
                format_func=lambda x: MODEL_LABELS[x], key="fc_model")
            max_idx = len(test_df) - 74
            window_idx = st.slider("Test window start:", 0, max_idx, max_idx // 2)
            run_btn = st.button("🚀 Run Forecast")

            ts_start = test_df.index[window_idx]
            ts_target = test_df.index[window_idx + 72]
            st.markdown(f"""
            <div class='forecast-box' style='margin-top:12px'>
                <div style='color:#8b949e; font-size:.78rem; margin-bottom:6px'>Selected Window</div>
                <div style='font-size:.85rem'>📅 <b>From:</b> {ts_start.strftime('%Y-%m-%d %H:%M')}</div>
                <div style='font-size:.85rem'>⏱️ <b>Input:</b> 72 hours of history</div>
                <div style='font-size:.85rem'>🎯 <b>Predicts:</b> {ts_target.strftime('%Y-%m-%d %H:%M')}</div>
            </div>
            """, unsafe_allow_html=True)

        with col_out:
            window_raw    = df_feat.iloc[test_start_idx + window_idx:
                                         test_start_idx + window_idx + 72][ALL_FEATURES].values
            window_scaled = scaler.transform(window_raw)
            actual_raw    = test_df.iloc[window_idx + 72][TARGET_COLS].values

            with st.spinner("Running inference…"):
                preds_phys, _ = run_inference(model_choice, window_scaled)

            st.markdown("#### 📊 Forecast Results")
            res_cols = st.columns(4)
            for i, var in enumerate(TARGET_COLS):
                pred_val = preds_phys[i]
                act_val  = actual_raw[i]
                delta    = pred_val - act_val
                clr      = "#43CFCA" if abs(delta) < 0.5 else "#FF6B6B"
                with res_cols[i]:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <div class='metric-label'>{TARGET_ICONS[var]} {var}</div>
                        <div class='metric-value' style='color:{VAR_COLORS[var]}'>{pred_val:.2f}</div>
                        <div class='metric-unit'>{TARGET_UNITS[var]}</div>
                        <div style='margin-top:8px; font-size:.78rem; color:{clr}'>
                            Actual: {act_val:.2f} · Δ {delta:+.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Plot historical window
            st.markdown("<br>", unsafe_allow_html=True)
            fig = make_subplots(rows=2, cols=2,
                subplot_titles=[f"{v} ({TARGET_UNITS[v]})" for v in TARGET_COLS],
                vertical_spacing=0.14, horizontal_spacing=0.08)
            hrs_rel = list(range(-72, 0))
            positions = [(1,1),(1,2),(2,1),(2,2)]
            for (r, c), var in zip(positions, TARGET_COLS):
                idx_v = TARGET_COLS.index(var)
                hist  = test_df.iloc[window_idx: window_idx + 72][var].values
                fig.add_trace(go.Scatter(x=hrs_rel, y=hist, mode="lines",
                    line=dict(color=VAR_COLORS[var], width=2), showlegend=False), row=r, col=c)
                fig.add_trace(go.Scatter(x=[0], y=[actual_raw[idx_v]], mode="markers",
                    marker=dict(color="white", size=10, symbol="circle"), showlegend=False), row=r, col=c)
                fig.add_trace(go.Scatter(x=[0], y=[preds_phys[idx_v]], mode="markers",
                    marker=dict(color=VAR_COLORS[var], size=14, symbol="star"), showlegend=False), row=r, col=c)
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
                font=dict(color="#e6edf3", family="Inter"),
                height=460, margin=dict(l=0,r=0,t=40,b=0))
            for row in range(1, 3):
                for col in range(1, 3):
                    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)",
                        title_text="Hours relative to forecast" if row==2 else "", row=row, col=col)
                    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", row=row, col=col)
            st.plotly_chart(fig, use_container_width=True)
            st.caption("⭐ Star = Forecast | ⚪ Circle = Actual value at t+1h")

    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — ATTENTION MAPS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 Attention Maps":
    st.markdown("<div class='section-header'>🧠 Temporal Attention Maps</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e'>Visualize which past hours the GRU+Attention model focuses on during forecast</p>", unsafe_allow_html=True)

    try:
        df = load_dataset()
        from src.features import add_cyclical_time_features
        df_feat = add_cyclical_time_features(df)
        scaler  = load_scaler()

        n = len(df_feat)
        test_start_idx = int(n * 0.85)
        test_df = df_feat.iloc[test_start_idx:]

        col_ctrl, col_attn = st.columns([1, 2], gap="large")

        with col_ctrl:
            st.markdown("#### ⚙️ Controls")
            max_idx   = len(test_df) - 74
            window_idx = st.slider("Window index:", 0, max_idx, max_idx // 3, key="attn_win")

            ts_start  = test_df.index[window_idx]
            ts_target = test_df.index[window_idx + 72]
            month_n   = ts_target.month
            season_n  = "Winter" if month_n in [12,1,2] else "Spring" if month_n in [3,4,5] else "Summer" if month_n in [6,7,8] else "Autumn"
            st.markdown(f"""
            <div class='forecast-box' style='margin-top:12px'>
                <div style='color:#8b949e; font-size:.78rem; margin-bottom:6px'>Window Info</div>
                <div style='font-size:.85rem'>📅 {ts_start.strftime('%b %d %Y %H:%M')}</div>
                <div style='font-size:.85rem'>🎯 Predicts: {ts_target.strftime('%b %d %Y %H:%M')}</div>
                <div style='font-size:.85rem; color:#FFA552; margin-top:4px'>Season: <b>{season_n}</b></div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div style='margin-top:16px; padding:14px; background:rgba(108,99,255,0.08);
                border-radius:12px; border:1px solid rgba(108,99,255,0.2);
                font-size:.78rem; color:#8b949e; line-height:1.7'>
                <b style='color:#a5a0ff'>Reading the chart:</b><br>
                Peak at <b>t−24h, −48h, −72h</b> → diurnal alignment.<br>
                Peak near <b>t−1h to t−6h</b> → rapid frontal shift.
            </div>
            """, unsafe_allow_html=True)

        with col_attn:
            window_raw    = df_feat.iloc[test_start_idx + window_idx:
                                         test_start_idx + window_idx + 72][ALL_FEATURES].values
            window_scaled = scaler.transform(window_raw)

            with st.spinner("Extracting attention weights…"):
                preds_phys, attn_weights = run_inference("GRU_with_Attention", window_scaled)

            if attn_weights is not None:
                attn_1d = attn_weights[0, :, 0]
                hrs_rel = list(range(-72, 0))

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=hrs_rel, y=attn_1d,
                    marker=dict(color=attn_1d,
                        colorscale=[[0,"#13161d"],[0.4,"#6C63FF"],[1,"#43CFCA"]],
                        cmin=float(attn_1d.min()), cmax=float(attn_1d.max())),
                ))
                for marker_h in [-24, -48, -72]:
                    fig.add_vline(x=marker_h, line_dash="dash",
                        line_color="rgba(255,165,82,0.6)", line_width=1.5,
                        annotation_text=f"t{marker_h}h",
                        annotation_font_color="#FFA552",
                        annotation_position="top")
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
                    font=dict(color="#e6edf3", family="Inter"),
                    xaxis=dict(title="Hours before forecast", gridcolor="rgba(255,255,255,0.06)"),
                    yaxis=dict(title="Softmax Weight α", gridcolor="rgba(255,255,255,0.06)"),
                    height=300, margin=dict(l=0,r=0,t=20,b=0))
                st.plotly_chart(fig, use_container_width=True)

                # 2D heatmap: 3 days × 24 hours
                st.markdown("#### 🌡️ Attention Intensity by Day × Hour")
                attn_2d = attn_1d.reshape(3, 24)
                fig2 = go.Figure(go.Heatmap(
                    z=attn_2d,
                    x=[f"{h:02d}:00" for h in range(24)],
                    y=["Day −3 (t−72h to t−48h)", "Day −2 (t−48h to t−24h)", "Day −1 (t−24h to t−0h)"],
                    colorscale=[[0,"#0d1117"],[0.5,"#6C63FF"],[1,"#43CFCA"]]))
                fig2.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#e6edf3", family="Inter"),
                    height=220, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig2, use_container_width=True)

                # Forecast result
                st.markdown("#### 🎯 Forecast at this Window")
                fc_cols = st.columns(4)
                for i, var in enumerate(TARGET_COLS):
                    actual = test_df.iloc[window_idx + 72][var]
                    with fc_cols[i]:
                        st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>{TARGET_ICONS[var]} {var}</div>
                            <div class='metric-value' style='color:{VAR_COLORS[var]}; font-size:1.5rem'>{preds_phys[i]:.2f}</div>
                            <div class='metric-unit'>{TARGET_UNITS[var]}</div>
                            <div style='font-size:.75rem; color:#8b949e; margin-top:4px'>actual: {actual:.2f}</div>
                        </div>
                        """, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<div class='section-header'>📌 Pre-Computed Case Studies</div>", unsafe_allow_html=True)
        case_img = FIGURES_DIR / "08_attention_case_studies.png"
        if case_img.exists():
            st.image(str(case_img),
                caption="4 meteorological regimes: Diurnal cycle · Frontal shift · Turbulent gust · Stagnant high-pressure",
                use_column_width=True)

    except Exception as e:
        st.error(f"Attention visualization failed: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🏆 Model Comparison":
    st.markdown("<div class='section-header'>🏆 Model Benchmark</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e'>Side-by-side evaluation on the held-out test set (Oct 2015 – Jan 2017)</p>", unsafe_allow_html=True)

    metrics_df = load_metrics()

    tab1, tab2, tab3 = st.tabs(["📋 Metrics Table", "📈 RMSE Comparison", "🕸️ R² Radar"])

    with tab1:
        styled = metrics_df.copy()
        styled.columns = ["Model", "Variable", "Unit", "Val RMSE", "Test MAE", "Test RMSE", "R²"]
        styled["Model"] = styled["Model"].map(lambda x: MODEL_LABELS.get(x, x))
        st.dataframe(styled, use_container_width=True, hide_index=True,
            column_config={
                "R²":        st.column_config.ProgressColumn("R²", min_value=0, max_value=1, format="%.4f"),
                "Test RMSE": st.column_config.NumberColumn("Test RMSE", format="%.4f"),
                "Test MAE":  st.column_config.NumberColumn("Test MAE",  format="%.4f"),
                "Val RMSE":  st.column_config.NumberColumn("Val RMSE",  format="%.4f"),
            })

    with tab2:
        metric_sel = st.radio("Metric:", ["Test_RMSE", "Test_MAE", "Val_RMSE", "Test_R2_Score"],
            horizontal=True, key="cmp_metric")
        fig = go.Figure()
        for model_name in MODEL_NAMES:
            sub = metrics_df[metrics_df["Model"] == model_name]
            fig.add_trace(go.Bar(
                name=MODEL_LABELS[model_name],
                x=sub["Target Variable"], y=sub[metric_sel],
                marker_color=MODEL_COLORS[model_name],
                text=sub[metric_sel].round(4), textposition="outside",
                textfont=dict(color="#e6edf3")))
        fig.update_layout(
            barmode="group",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
            font=dict(color="#e6edf3", family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=metric_sel.replace("_"," ")),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=420, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        radar_vars = TARGET_COLS
        fig = go.Figure()
        for model_name in MODEL_NAMES:
            sub = metrics_df[metrics_df["Model"] == model_name]
            r2_vals = [sub[sub["Target Variable"] == v]["Test_R2_Score"].values[0] for v in radar_vars]
            r2_vals_closed = r2_vals + [r2_vals[0]]
            fig.add_trace(go.Scatterpolar(
                r=r2_vals_closed, theta=radar_vars + [radar_vars[0]],
                name=MODEL_LABELS[model_name], fill="toself",
                line_color=MODEL_COLORS[model_name],
                fillcolor=MODEL_COLORS[model_name] + "28"))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0.8, 1.0],
                    gridcolor="rgba(255,255,255,0.1)", color="#8b949e"),
                angularaxis=dict(gridcolor="rgba(255,255,255,0.1)", color="#8b949e"),
                bgcolor="rgba(13,17,23,1)"),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e6edf3", family="Inter"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            height=450, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.info("📌 GRU+Attention leads on Pressure (R²=0.9974) and Temperature (R²=0.9930) — the two most physically regular variables.")

    st.markdown("<hr>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        img = FIGURES_DIR / "06_training_loss_curves.png"
        if img.exists():
            st.image(str(img), caption="Training & Validation Loss Convergence", use_column_width=True)
    with col_b:
        img = FIGURES_DIR / "07_actual_vs_predicted.png"
        if img.exists():
            st.image(str(img), caption="Actual vs Predicted — 7-Day Test Window", use_column_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — ABLATION STUDY
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔬 Ablation Study":
    st.markdown("<div class='section-header'>🔬 Hyperparameter Tuning & Ablation</div>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8b949e'>Systematic comparison of lookback windows, recurrent depth, and attention scoring mechanisms</p>", unsafe_allow_html=True)

    tune_df = load_tuning()

    categories = ["All"] + sorted(tune_df["Category"].unique().tolist())
    cat = st.selectbox("Filter by category:", categories, key="abl_cat")
    df_filt = tune_df if cat == "All" else tune_df[tune_df["Category"] == cat]

    tab1, tab2, tab3 = st.tabs(["📋 Results Table", "📊 Val RMSE", "⏱️ Speed vs Accuracy"])

    with tab1:
        display_cols = ["Experiment", "Category", "Window_Length", "GRU_Layers",
                        "Scoring_Function", "Val_Mean_RMSE", "Test_Mean_RMSE", "Training_Time_s"]
        st.dataframe(df_filt[display_cols].rename(columns={
            "Window_Length": "Window (h)", "GRU_Layers": "Layers",
            "Scoring_Function": "Scoring", "Val_Mean_RMSE": "Val RMSE",
            "Test_Mean_RMSE": "Test RMSE", "Training_Time_s": "Train (s)"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Val RMSE":  st.column_config.NumberColumn(format="%.4f"),
                "Test RMSE": st.column_config.NumberColumn(format="%.4f"),
                "Train (s)": st.column_config.NumberColumn(format="%.1f"),
            })

    with tab2:
        colors = [ACCENT4 if "Attention" in r else ACCENT for r in df_filt["Experiment"]]
        fig = go.Figure(go.Bar(
            x=df_filt["Experiment"], y=df_filt["Val_Mean_RMSE"],
            marker=dict(color=colors, line=dict(width=0)),
            text=df_filt["Val_Mean_RMSE"].round(4), textposition="outside",
            textfont=dict(color="#e6edf3", size=11)))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
            font=dict(color="#e6edf3", family="Inter"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.06)", tickangle=-25),
            yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Val Mean RMSE"),
            height=400, margin=dict(l=0,r=0,t=20,b=80))
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = go.Figure()
        for _, row in df_filt.iterrows():
            is_attn = "Attention" in row["Experiment"]
            fig.add_trace(go.Scatter(
                x=[row["Training_Time_s"]], y=[row["Val_Mean_RMSE"]],
                mode="markers+text",
                text=[row["Experiment"].replace("GRU_","")],
                textposition="top center",
                textfont=dict(size=10, color="#8b949e"),
                marker=dict(size=14,
                    color=ACCENT4 if is_attn else ACCENT,
                    line=dict(color="#e6edf3", width=1.5)),
                showlegend=False))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,17,23,1)",
            font=dict(color="#e6edf3", family="Inter"),
            xaxis=dict(title="Training Time (seconds)", gridcolor="rgba(255,255,255,0.06)"),
            yaxis=dict(title="Val Mean RMSE", gridcolor="rgba(255,255,255,0.06)"),
            height=420, margin=dict(l=0,r=0,t=20,b=0))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    img = FIGURES_DIR / "09_hyperparameter_tuning_ablation.png"
    if img.exists():
        st.image(str(img), caption="Hyperparameter Tuning & Ablation Study — Full Results", use_column_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>🔑 Key Findings</div>", unsafe_allow_html=True)
    findings_cols = st.columns(3)
    findings = [
        ("⚡ Attention Scoring", "**Bahdanau additive** (Val RMSE 1.575) vs Luong dot-product (3.388). Nonlinear projection is critical for multi-sensor weather alignment.", ACCENT),
        ("📐 Optimal Lookback", "**72-hour window** captures 3 full diurnal cycles (24h, 48h, 72h), giving attention ideal historical context.", ACCENT2),
        ("🧱 Recurrent Depth", "**2-layer GRU** (64→32) beats 3-layer (64→32→16). Extra depth causes over-parameterization on noisy atmospheric signals.", ACCENT4),
    ]
    for col_w, (title, desc, color) in zip(findings_cols, findings):
        with col_w:
            st.markdown(f"""
            <div style='background:rgba(255,255,255,0.03); border:1px solid {color}44;
                border-radius:14px; padding:20px; height:180px'>
                <div style='font-weight:700; color:{color}; margin-bottom:10px; font-size:.95rem'>{title}</div>
                <div style='color:#8b949e; font-size:.82rem; line-height:1.7'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)
