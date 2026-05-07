import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import datetime
from utils import load_data
from model import prepare_features, train_model, predict

st.set_page_config(page_title="AgriFlow AI", layout="wide", page_icon="")

# ─────────────────────────── CSS ───────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── PAGE ── */
body { background: #eef4ee; color: #1a2e1a; }
.block-container { padding-top: 1rem !important; padding-bottom: 2.5rem !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #d8e8d8; }
::-webkit-scrollbar-thumb { background: #2e8b57; border-radius: 3px; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: #12271e !important;
    border-right: 4px solid #2e8b57;
}
[data-testid="stSidebar"] * { color: #c8e6c9 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(180deg, #3da868 0%, #2e8b57 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border: 2px solid #4caf7d !important;
    border-bottom: 4px solid #1a5c38 !important;
    border-radius: 10px !important;
    padding: 10px !important;
    letter-spacing: 0.5px;
    transition: all 0.15s ease;
    box-shadow: 0 4px 12px rgba(46,139,87,0.35);
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(180deg, #4dbf78 0%, #3da868 100%) !important;
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(46,139,87,0.45) !important;
}
[data-testid="stSidebar"] .stButton > button:active {
    transform: translateY(1px);
    border-bottom-width: 2px !important;
}

/* ── HEADER ── */
.agri-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 28px;
    background: linear-gradient(135deg, #12271e 0%, #1a3828 60%, #1e4530 100%);
    border-radius: 18px;
    margin-bottom: 24px;
    border: 2px solid #2e8b57;
    border-bottom: 5px solid #1a5c38;
    box-shadow: 0 6px 0 #0e1f16, 0 10px 32px rgba(14,31,22,0.35);
}
.agri-logo { display: flex; align-items: center; gap: 14px; }
.agri-logo-icon {
    width: 48px; height: 48px;
    background: linear-gradient(145deg, #3da868, #2e8b57);
    border-radius: 14px;
    border: 2px solid #4caf7d;
    border-bottom: 3px solid #1a5c38;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800; color: #fff; letter-spacing: -0.5px;
    box-shadow: 0 3px 0 #1a5c38, 0 4px 12px rgba(46,139,87,0.4);
}
.agri-logo-title { font-size: 22px; font-weight: 800; color: #ffffff; letter-spacing: -0.5px; }
.agri-logo-title span { color: #4caf7d; }
.agri-logo-sub   { font-size: 11px; color: #5a9a6a; letter-spacing: 0.5px; margin-top: 2px; }
.header-right    { display: flex; align-items: center; gap: 12px; }

/* Date widget */
.date-widget {
    background: linear-gradient(160deg, #1e4530 0%, #163828 100%);
    border: 2px solid #2e8b57;
    border-bottom: 4px solid #1a5c38;
    border-radius: 14px;
    padding: 10px 22px;
    text-align: center;
    min-width: 80px;
    box-shadow: 0 4px 0 #0e1f16;
}
.date-widget-day   { font-size: 28px; font-weight: 800; color: #4caf7d; line-height: 1; }
.date-widget-month { font-size: 10px; color: #5a9a6a; text-transform: uppercase; letter-spacing: 1.5px; margin-top: 3px; }

/* ── MAIN CARD ── */
.main-card {
    background: linear-gradient(160deg, #ffffff 0%, #f6fbf6 100%);
    border-radius: 20px;
    border: 2px solid #7cbf8a;
    border-top: 6px solid #2e8b57;
    border-bottom: 4px solid #4caf7d;
    padding: 28px 32px;
    margin-bottom: 24px;
    box-shadow: 0 6px 0 #b8d8b8, 0 12px 36px rgba(26,56,40,0.1);
}
.main-card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    flex-wrap: wrap;
    gap: 20px;
}
.main-card-left h4 {
    font-size: 11px; color: #5a8a6a; margin-bottom: 6px;
    font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px;
}
.main-card-left h1 {
    font-size: 54px; font-weight: 800; color: #12271e;
    margin: 0; line-height: 1; letter-spacing: -2px;
}
.main-card-left h1 span { font-size: 20px; font-weight: 500; color: #7aaa8a; letter-spacing: 0; }
.change-down { color: #b71c1c; font-size: 13px; font-weight: 700; margin-top: 10px; }
.change-up   { color: #1b5e20; font-size: 13px; font-weight: 700; margin-top: 10px; }
.pill {
    display: inline-block; padding: 3px 12px;
    background: linear-gradient(135deg, #e8f5e9, #d0f0d8);
    border: 1px solid #81c784;
    border-bottom: 2px solid #4caf50;
    border-radius: 20px;
    font-size: 11px; color: #1b5e20; font-weight: 700;
}

.main-stats { display: flex; gap: 8px; flex-wrap: wrap; align-items: stretch; }
.stat-item {
    text-align: center;
    background: linear-gradient(160deg, #f1f9f1, #e8f5e9);
    border: 2px solid #81c784;
    border-top: 3px solid #4caf50;
    border-bottom: 3px solid #388e3c;
    border-radius: 12px;
    padding: 12px 16px;
    min-width: 90px;
    box-shadow: 0 3px 0 #b8d8c8;
}
.stat-label { font-size: 9px; text-transform: uppercase; letter-spacing: 1.2px; color: #5a8a6a; margin-bottom: 5px; font-weight: 800; }
.stat-value { font-size: 14px; font-weight: 800; color: #12271e; }

/* ── SECTION TITLE ── */
.section-title {
    font-size: 11px; font-weight: 800; color: #1e5c38;
    margin: 28px 0 14px 0;
    display: flex; align-items: center; gap: 10px;
    text-transform: uppercase; letter-spacing: 1.5px;
}
.section-title::before {
    content: '';
    width: 5px; height: 18px;
    background: linear-gradient(180deg, #4caf7d, #2e8b57);
    border-radius: 3px;
    flex-shrink: 0;
}
.section-title::after {
    content: ''; flex: 1; height: 2px;
    background: linear-gradient(90deg, #81c784, #c8e6c9, #eef4ee);
    margin-left: 4px;
    border-radius: 1px;
}

/* ── FORECAST CARDS ── */
.forecast-card {
    border-radius: 18px;
    padding: 28px 22px;
    text-align: center;
    height: 100%;
    transition: transform 0.18s, box-shadow 0.18s;
}
.forecast-card:hover { transform: translateY(-4px); }

.fc-green {
    background: linear-gradient(160deg, #ffffff 0%, #f2faf5 100%);
    border: 2px solid #2e8b57;
    border-top: 5px solid #1b5e20;
    border-bottom: 4px solid #2e8b57;
    box-shadow: 0 5px 0 #a5d6b5, 0 8px 24px rgba(46,139,87,0.15);
}
.fc-amber {
    background: linear-gradient(160deg, #ffffff 0%, #fffdf0 100%);
    border: 2px solid #f9a825;
    border-top: 5px solid #e65100;
    border-bottom: 4px solid #f9a825;
    box-shadow: 0 5px 0 #ffe082, 0 8px 24px rgba(249,168,37,0.15);
}
.fc-red {
    background: linear-gradient(160deg, #ffffff 0%, #fff8f8 100%);
    border: 2px solid #e53935;
    border-top: 5px solid #b71c1c;
    border-bottom: 4px solid #e53935;
    box-shadow: 0 5px 0 #ffcdd2, 0 8px 24px rgba(229,57,53,0.15);
}

.fc-label        { font-size: 9px; text-transform: uppercase; letter-spacing: 2px; color: #7a9a7a; margin-bottom: 18px; font-weight: 800; }
.fc-status-green { font-size: 19px; font-weight: 800; color: #1b5e20; letter-spacing: 1px; }
.fc-status-amber { font-size: 19px; font-weight: 800; color: #e65100; letter-spacing: 1px; }
.fc-status-red   { font-size: 19px; font-weight: 800; color: #b71c1c; letter-spacing: 1px; }

.fc-conf-bar-bg    { width: 100%; height: 7px; background: #e8f5e9; border-radius: 4px; margin: 16px 0 8px 0; border: 1px solid #c8e6c9; overflow: hidden; }
.fc-conf-bar-green { height: 7px; border-radius: 4px; background: linear-gradient(90deg, #81c784, #2e8b57, #1b5e20); }
.fc-conf-bar-amber { height: 7px; border-radius: 4px; background: linear-gradient(90deg, #ffe082, #ffa000, #e65100); }
.fc-conf  { font-size: 11px; color: #5a8a6a; font-weight: 600; }
.fc-yield { font-size: 13px; color: #2e5a3e; margin-top: 10px; font-weight: 700; }

/* ── METRIC CARDS ── */
.metric-card {
    background: linear-gradient(160deg, #ffffff 0%, #f6fbf6 100%);
    border: 2px solid #81c784;
    border-top: 4px solid #2e8b57;
    border-bottom: 3px solid #388e3c;
    border-radius: 14px;
    padding: 18px 12px;
    text-align: center;
    transition: transform 0.18s, box-shadow 0.18s, border-color 0.18s;
    box-shadow: 0 3px 0 #b8d8b8;
}
.metric-card:hover {
    border-color: #2e8b57;
    border-top-color: #1b5e20;
    transform: translateY(-3px);
    box-shadow: 0 6px 0 #81c784, 0 10px 24px rgba(46,139,87,0.15);
}
.mc-label    { font-size: 9px; text-transform: uppercase; letter-spacing: 1.2px; color: #5a8a6a; margin-bottom: 10px; font-weight: 800; }
.mc-value-green  { font-size: 21px; font-weight: 800; color: #1b5e20; }
.mc-value-amber  { font-size: 21px; font-weight: 800; color: #e65100; }
.mc-value-blue   { font-size: 21px; font-weight: 800; color: #0d47a1; }
.mc-value-purple { font-size: 21px; font-weight: 800; color: #4a148c; }
.mc-value-white  { font-size: 21px; font-weight: 800; color: #12271e; }
.mc-value-red    { font-size: 21px; font-weight: 800; color: #b71c1c; }
.mc-sub { font-size: 10px; color: #7a9a7a; margin-top: 5px; font-weight: 600; }

/* ── CHART WRAPPER ── */
.chart-outer {
    background: linear-gradient(160deg, #ffffff 0%, #f9fdf9 100%);
    border: 2px solid #81c784;
    border-top: 5px solid #2e8b57;
    border-bottom: 3px solid #4caf50;
    border-radius: 18px;
    padding: 6px;
    box-shadow: 0 5px 0 #b8d8b8, 0 8px 24px rgba(26,56,40,0.08);
    margin-bottom: 6px;
}

/* ── ALERT CARDS ── */
.alert-card {
    border-radius: 14px;
    padding: 16px 18px;
    margin-bottom: 12px;
    display: flex;
    align-items: flex-start;
    gap: 14px;
    border-left: 6px solid;
    border-top: 2px solid;
    border-right: 2px solid;
    border-bottom: 3px solid;
}
.alert-high {
    background: linear-gradient(135deg, #fff5f5 0%, #ffebee 100%);
    border-left-color: #b71c1c;
    border-top-color: #ef9a9a;
    border-right-color: #ef9a9a;
    border-bottom-color: #e53935;
}
.alert-medium {
    background: linear-gradient(135deg, #fffde7 0%, #fff8e1 100%);
    border-left-color: #e65100;
    border-top-color: #ffe082;
    border-right-color: #ffe082;
    border-bottom-color: #ffa000;
}
.alert-low {
    background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
    border-left-color: #1b5e20;
    border-top-color: #a5d6a7;
    border-right-color: #a5d6a7;
    border-bottom-color: #388e3c;
}
.alert-dot-high   { width: 11px; height: 11px; border-radius: 50%; background: #b71c1c; border: 2px solid #ef5350; margin-top: 3px; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(183,28,28,0.15); }
.alert-dot-medium { width: 11px; height: 11px; border-radius: 50%; background: #e65100; border: 2px solid #ff9800; margin-top: 3px; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(230,81,0,0.15); }
.alert-dot-low    { width: 11px; height: 11px; border-radius: 50%; background: #1b5e20; border: 2px solid #4caf50; margin-top: 3px; flex-shrink: 0; box-shadow: 0 0 0 3px rgba(27,94,32,0.15); }
.alert-title { font-size: 13px; font-weight: 800; color: #12271e; margin-bottom: 4px; }
.alert-body  { font-size: 12px; color: #3a5a3a; line-height: 1.65; }

/* ── INSIGHT CARD ── */
.insight-card {
    background: linear-gradient(160deg, #ffffff 0%, #f2faf5 100%);
    border: 2px solid #4caf50;
    border-top: 6px solid #2e8b57;
    border-bottom: 3px solid #388e3c;
    border-radius: 18px;
    padding: 24px 26px;
    height: 100%;
    box-shadow: 0 5px 0 #a5d6a7, 0 8px 24px rgba(46,139,87,0.1);
}
.insight-title {
    font-size: 11px; font-weight: 800; color: #1b5e20;
    margin-bottom: 16px;
    text-transform: uppercase; letter-spacing: 1.5px;
    padding-bottom: 12px;
    border-bottom: 3px solid #81c784;
}
.insight-item  {
    font-size: 12px; color: #2e4a2e; line-height: 1.7; padding: 8px 0;
    border-bottom: 1px solid #c8e6c9;
    display: flex; justify-content: space-between; gap: 10px;
}
.insight-item:last-child { border-bottom: none; }
.insight-key { color: #5a7a5a; font-weight: 600; }
.insight-val { font-weight: 800; text-align: right; }

/* ── COMPARE TABLE ── */
.compare-table {
    width: 100%; border-collapse: separate; border-spacing: 0;
    border-radius: 14px; overflow: hidden; font-size: 13px;
    border: 2px solid #81c784; margin-top: 14px;
    box-shadow: 0 3px 0 #b8d8b8;
}
.compare-table th {
    background: linear-gradient(135deg, #2e8b57, #1b5e20);
    color: #ffffff;
    font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
    padding: 14px 16px; text-align: left; font-weight: 800;
    border-bottom: 3px solid #1b5e20;
}
.compare-table td {
    padding: 11px 16px; border-bottom: 1px solid #e8f5e9;
    color: #12271e; background: #fff; font-weight: 600;
}
.compare-table tr:hover td { background: linear-gradient(90deg, #f1f9f1, #ffffff); }
.compare-table tr:last-child td { border-bottom: none; }

/* ── SIDEBAR COMPONENTS ── */
.coverage-box {
    background: linear-gradient(160deg, #1a3828, #12271e);
    border: 2px solid #2e8b57;
    border-top: 3px solid #4caf7d;
    border-radius: 12px; padding: 14px 16px;
    font-size: 12px; color: #7aaa8a;
    margin-top: 14px; line-height: 2;
    box-shadow: 0 3px 0 #0e1f16;
}
.coverage-title {
    font-size: 9px; text-transform: uppercase; letter-spacing: 1.5px;
    color: #4caf7d; font-weight: 800; margin-bottom: 8px;
}
.sidebar-divider { height: 1px; background: linear-gradient(90deg, transparent, #2e8b57, transparent); margin: 18px 0; }
.sidebar-help    { font-size: 11px; color: #4a6a4a; line-height: 1.8; margin-top: 14px; }

/* ── DONUT WRAPPER ── */
.donut-wrap {
    background: linear-gradient(160deg, #ffffff, #f6fbf6);
    border: 2px solid #81c784;
    border-top: 4px solid #2e8b57;
    border-bottom: 3px solid #4caf50;
    border-radius: 14px;
    padding: 4px;
    margin-bottom: 10px;
    box-shadow: 0 4px 0 #b8d8b8;
}

/* ── CHART SECTION LABELS ── */
.trade-label {
    font-size: 11px; color: #1b5e20; text-transform: uppercase;
    letter-spacing: 1px; margin-bottom: 8px; font-weight: 800;
    padding: 6px 12px;
    background: linear-gradient(135deg, #e8f5e9, #d0f0d8);
    border: 2px solid #81c784;
    border-bottom: 3px solid #4caf50;
    border-radius: 8px;
    display: inline-block;
    box-shadow: 0 2px 0 #a5d6a7;
}

/* ── SPLASH ── */
.splash-wrap { text-align: center; padding: 80px 0 60px; }
.splash-icon {
    width: 68px; height: 68px;
    background: linear-gradient(145deg, #3da868, #2e8b57);
    border-radius: 22px;
    border: 3px solid #4caf7d;
    border-bottom: 6px solid #1a5c38;
    margin: 0 auto 24px;
    display: flex; align-items: center; justify-content: center;
    font-size: 18px; font-weight: 800; color: #fff;
    box-shadow: 0 6px 0 #a5d6a7, 0 10px 28px rgba(46,139,87,0.25);
}
.splash-title { font-size: 26px; font-weight: 800; color: #12271e; margin-bottom: 10px; }
.splash-sub   { font-size: 14px; color: #5a8a6a; font-weight: 500; }

/* ── FOOTER ── */
.agri-footer {
    margin-top: 36px; padding: 18px 26px;
    background: linear-gradient(135deg, #ffffff 0%, #f6fbf6 100%);
    border: 2px solid #81c784;
    border-top: 5px solid #2e8b57;
    border-bottom: 3px solid #4caf50;
    border-radius: 16px;
    display: flex; align-items: center; justify-content: space-between;
    flex-wrap: wrap; gap: 10px;
    box-shadow: 0 5px 0 #b8d8b8;
}
.footer-brand { display: flex; align-items: center; gap: 10px; }
.footer-icon {
    width: 30px; height: 30px;
    background: linear-gradient(145deg, #3da868, #2e8b57);
    border-radius: 9px;
    border: 2px solid #4caf7d;
    border-bottom: 3px solid #1a5c38;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 800; color: #fff;
    box-shadow: 0 3px 0 #1a5c38;
}
.footer-name { font-size: 12px; color: #2e5a3e; font-weight: 700; }
.footer-meta { font-size: 11px; color: #7a9a7a; font-weight: 500; }

.js-plotly-plot .plotly .modebar { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────── LOAD DATA ───────────────────────────
df = load_data()

now       = datetime.datetime.now()
now_day   = now.strftime("%d")
now_month = now.strftime("%b %Y")
now_full  = now.strftime("%A, %d %B %Y")

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:22px;
                padding-bottom:16px;border-bottom:2px solid #2e8b57;'>
        <div style='width:38px;height:38px;background:#2e8b57;border-radius:10px;
                    border:2px solid #4caf7d;border-bottom:3px solid #1a5c38;
                    display:flex;align-items:center;justify-content:center;
                    font-size:12px;font-weight:800;color:#fff;
                    box-shadow:0 3px 0 #1a5c38;'>AF</div>
        <div>
            <div style='font-size:15px;font-weight:800;color:#fff;letter-spacing:-0.3px;'>AgriFlow AI</div>
            <div style='font-size:10px;color:#5a8a6a;letter-spacing:0.5px;'>Agricultural Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='font-size:9px;text-transform:uppercase;letter-spacing:1.5px;"
                "color:#4caf7d;font-weight:800;margin-bottom:10px;'>CONTROL PANEL</div>",
                unsafe_allow_html=True)

    country = st.selectbox("Country", sorted(df["Country"].unique()))
    crop    = st.selectbox("Crop", df[df["Country"] == country]["Crop"].unique())

    year_min = int(df["Year"].min())
    year_max = int(df["Year"].max())
    period   = st.selectbox("Analysis Period", [
        f"Full Period ({year_min}–{year_max})",
        f"Last 5 Years ({year_max-4}–{year_max})",
        f"Last 3 Years ({year_max-2}–{year_max})",
    ])

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size:9px;text-transform:uppercase;letter-spacing:1.5px;"
                "color:#4caf7d;font-weight:800;margin-bottom:8px;'>COMPARISON</div>",
                unsafe_allow_html=True)
    compare_enabled = st.toggle("Enable Crop Comparison", value=False)
    if compare_enabled:
        compare_crop = st.selectbox("Compare with Crop",
            [c for c in df[df["Country"] == country]["Crop"].unique() if c != crop])
    else:
        compare_crop = None

    st.markdown("<div class='sidebar-divider'></div>", unsafe_allow_html=True)
    run = st.button("Run Analysis", use_container_width=True, type="primary")

    n_countries = df["Country"].nunique()
    n_crops     = df["Crop"].nunique()
    n_records   = len(df)
    n_combos    = df.groupby(["Country","Crop"]).ngroups

    st.markdown(f"""
    <div class="coverage-box">
        <div class="coverage-title">DATA COVERAGE</div>
        <b style="color:#c8e6c9">{n_countries}</b> countries &nbsp;·&nbsp;
        <b style="color:#c8e6c9">{n_crops}</b> crops<br>
        <b style="color:#c8e6c9">{year_min}–{year_max}</b> &nbsp;·&nbsp;
        <b style="color:#c8e6c9">{n_records:,}</b> records<br>
        <b style="color:#c8e6c9">{n_combos:,}</b> unique combinations
    </div>
    <div class="sidebar-help">
        Select a country and crop, then click
        <b style="color:#4caf7d">Run Analysis</b>
        to get yield trends, food security forecasts,
        risk alerts, and AI-powered recommendations.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────── SPLASH ───────────────────────────
if not run:
    st.markdown(f"""
    <div class="agri-header">
        <div class="agri-logo">
            <div class="agri-logo-icon">AF</div>
            <div>
                <div class="agri-logo-title">AgriFlow <span>AI</span></div>
                <div class="agri-logo-sub">Agricultural Intelligence Platform</div>
            </div>
        </div>
        <div class="header-right">
            <div class="date-widget">
                <div class="date-widget-day">{now_day}</div>
                <div class="date-widget-month">{now_month}</div>
            </div>
        </div>
    </div>
    <div class="splash-wrap">
        <div class="splash-icon">AF</div>
        <div class="splash-title">Welcome to AgriFlow AI</div>
        <div class="splash-sub">Select a country and crop from the sidebar,
        then click <b>Run Analysis</b></div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────── FILTER ───────────────────────────
data = df[(df["Country"] == country) & (df["Crop"] == crop)].sort_values("Year")

if len(data) < 3:
    st.error("Not enough data for this combination. Please select another.")
    st.stop()

if "Last 5" in period:
    data = data[data["Year"] >= year_max - 4]
elif "Last 3" in period:
    data = data[data["Year"] >= year_max - 2]

# ─────────────────────────── MODEL ───────────────────────────
data_model      = prepare_features(data)
model, features = train_model(data_model)
pred, prob      = predict(model, data_model, features)

latest = data.iloc[-1]
prev   = data.iloc[-2] if len(data) >= 2 else latest

yield_change     = latest["Yield"] - prev["Yield"]
yield_change_pct = (yield_change / prev["Yield"] * 100) if prev["Yield"] != 0 else 0
arrow            = "▼" if yield_change < 0 else "▲"
change_class     = "change-down" if yield_change < 0 else "change-up"

# ─────────────────────────── HEADER ───────────────────────────
st.markdown(f"""
<div class="agri-header">
    <div class="agri-logo">
        <div class="agri-logo-icon">AF</div>
        <div>
            <div class="agri-logo-title">AgriFlow <span>AI</span></div>
            <div class="agri-logo-sub">Agricultural Intelligence Platform</div>
        </div>
    </div>
    <div class="header-right">
        <div class="date-widget">
            <div class="date-widget-day">{now_day}</div>
            <div class="date-widget-month">{now_month}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── MAIN CARD ───────────────────────────
prod_val    = f"{latest['Production']/1000:.1f}K MT" if latest['Production'] > 1000 else f"{latest['Production']:.0f} MT"
rain_val    = f"{latest['Rainfall']:.0f} mm"
temp_val    = f"{latest['Temperature']:.1f} C"
fsi_val     = latest['FSI']
year_val    = int(latest["Year"])
price_val   = f"${latest['Price']:.0f}/MT"        if 'Price'        in latest.index else "N/A"
trade_val   = f"${latest['TradeBalance']:.0f}M"    if 'TradeBalance' in latest.index else "N/A"
drought_val = latest['DroughtRisk']                if 'DroughtRisk'  in latest.index else "N/A"
fs_status   = latest['FSStatus']                  if 'FSStatus'     in latest.index \
              else ("Food Secure" if fsi_val > 60 else "At Risk")
fs_color    = "#1e8449" if "Secure"   in str(fs_status) \
              else "#c87a00" if "Moderate" in str(fs_status) else "#c0392b"
trade_color = "#1e8449" if latest.get('TradeBalance', 0) >= 0 else "#c0392b"

st.markdown(f"""
<div class="main-card">
    <div class="main-card-top">
        <div class="main-card-left">
            <h4>Current Analysis — {country} / {crop} ({year_val})</h4>
            <h1>{latest['Yield']:.2f} <span>MT/ha</span></h1>
            <div class="{change_class}">
                {arrow} {abs(yield_change_pct):.2f}% vs prev year &nbsp;
                <span class="pill">{country}</span> &nbsp;
                <span class="pill">{crop}</span>
            </div>
        </div>
        <div class="main-stats">
            <div class="stat-item">
                <div class="stat-label">PRICE</div>
                <div class="stat-value">{price_val}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">PRODUCTION</div>
                <div class="stat-value">{prod_val}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">TRADE BALANCE</div>
                <div class="stat-value" style="color:{trade_color}">{trade_val}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">FSI STATUS</div>
                <div class="stat-value" style="color:{fs_color};">{fs_status}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">DROUGHT RISK</div>
                <div class="stat-value">{drought_val}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">RAINFALL</div>
                <div class="stat-value">{rain_val}</div>
            </div>
            <div class="stat-item">
                <div class="stat-label">TEMPERATURE</div>
                <div class="stat-value">{temp_val}</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────── FORECAST CARDS ───────────────────────────
st.markdown('<div class="section-title">Forecast Summary</div>', unsafe_allow_html=True)

def forecast_label(p): return "IMPROVING" if p == 1 else "DECLINING"
def forecast_color(p): return "green"     if p == 1 else "amber"

short_label  = forecast_label(pred)
short_color  = forecast_color(pred)
medium_pred  = 1 if prob > 0.4  else 0
long_pred    = 1 if prob > 0.35 else 0
medium_label = "STABLE" if abs(prob - 0.5) < 0.15 else forecast_label(medium_pred)
long_label   = "STABLE" if abs(prob - 0.5) < 0.2  else forecast_label(long_pred)
medium_color = long_color = "amber"

short_conf  = prob * 100
medium_conf = max(30, prob * 85)
long_conf   = max(30, prob * 82)

est_yield      = latest["Yield"] * (1.03 if pred == 1 else 0.97)
est_yield_med  = latest["Yield"] * 1.005
est_yield_long = latest["Yield"] * 1.01

col1, col2, col3 = st.columns(3)

for col, term, label, color, conf, est in [
    (col1, "SHORT-TERM",  short_label,  short_color,  short_conf,  est_yield),
    (col2, "MEDIUM-TERM", medium_label, medium_color, medium_conf, est_yield_med),
    (col3, "LONG-TERM",   long_label,   long_color,   long_conf,   est_yield_long),
]:
    with col:
        st.markdown(f"""
        <div class="forecast-card fc-{color}">
            <div class="fc-label">{term}</div>
            <div class="fc-status-{color}">{label}</div>
            <div class="fc-conf-bar-bg">
                <div class="fc-conf-bar-{color}" style="width:{conf:.0f}%"></div>
            </div>
            <div class="fc-conf">Confidence: <b style="color:#1a3828">{conf:.1f}%</b></div>
            <div class="fc-yield">Est. {est:.2f} MT/ha</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────── METRICS ───────────────────────────
st.markdown('<div class="section-title">Key Metrics</div>', unsafe_allow_html=True)

avg_yield     = data["Yield"].mean()
profit_margin = data["ProfitMargin"].mean() if "ProfitMargin" in data.columns else 0
loss_rate     = data["LossRate"].mean()     if "LossRate"     in data.columns else 0
avg_fsi       = data["FSI"].mean()
mechanization = latest["Mechanization"]     if "Mechanization" in latest.index else 0
irrigation    = latest["Irrigation"]        if "Irrigation"    in latest.index else 0
soil_health   = latest["SoilHealth"]        if "SoilHealth"    in latest.index else 0
avg_temp      = data["Temperature"].mean()

metric_data = [
    ("AVG YIELD",     f"{avg_yield:.2f} MT/ha", "green",  f"Peak: {data['Yield'].max():.2f}"),
    ("PROFIT MARGIN", f"{profit_margin:.1f}%",  "green",  "Avg over period"),
    ("LOSS RATE",     f"{loss_rate:.1f}%",       "amber",  "Post-harvest"),
    ("AVG FSI",       f"{avg_fsi:.1f}",          "blue",   "Food Security"),
    ("MECHANIZATION", f"{mechanization:.0f}%",   "purple", "Farm coverage"),
    ("IRRIGATION",    f"{irrigation:.0f}%",      "blue",   "Irrigated area"),
    ("SOIL HEALTH",   f"{soil_health:.0f}/100",  "green",  "Composite score"),
    ("AVG TEMP",      f"{avg_temp:.1f} C",       "white",  f"Min: {data['Temperature'].min():.1f}"),
]

cols = st.columns(8)
for i, (label, value, color, sub) in enumerate(metric_data):
    with cols[i]:
        st.markdown(f"""
        <div class="metric-card">
            <div class="mc-label">{label}</div>
            <div class="mc-value-{color}">{value}</div>
            <div class="mc-sub">{sub}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────── YIELD CHART ───────────────────────────
st.markdown('<div class="section-title">Yield Performance</div>', unsafe_allow_html=True)

CHART_BG   = "rgba(0,0,0,0)"
GRID_COLOR = "#e8f5e9"
TICK_COLOR = "#6a9a7a"
HOVER_BG   = "#ffffff"

chart_col, side_col = st.columns([3, 1])

with chart_col:
    st.markdown('<div class="chart-outer">', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data["Year"], y=data["Yield"],
        mode="lines+markers", name="Yield (MT/ha)",
        line=dict(color="#2e8b57", width=3),
        marker=dict(size=8, color="#2e8b57", line=dict(color="#ffffff", width=2.5)),
        fill="tozeroy", fillcolor="rgba(76,175,125,0.12)"
    ))
    if len(data) >= 3:
        roll = data["Yield"].rolling(3, center=True).mean()
        fig.add_trace(go.Scatter(
            x=data["Year"], y=roll, mode="lines", name="3-Year Avg",
            line=dict(color="#1565c0", width=2, dash="dot"), opacity=0.9
        ))
    fig.add_trace(go.Scatter(
        x=data["Year"], y=data["FSI"], mode="lines", name="FSI Score",
        line=dict(color="#6a1b9a", width=2, dash="dash"),
        yaxis="y2", opacity=0.8
    ))
    fig.update_layout(
        template="plotly_white", paper_bgcolor=CHART_BG, plot_bgcolor="#fafffe",
        height=320, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", x=0, y=1.12,
                    font=dict(size=11, color=TICK_COLOR), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR), zeroline=False,
                   linecolor="#a5d6a7", linewidth=2),
        yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR), zeroline=False,
                   title=dict(text="MT/ha", font=dict(color=TICK_COLOR, size=10)),
                   linecolor="#a5d6a7", linewidth=2),
        yaxis2=dict(overlaying="y", side="right", zeroline=False, gridcolor="rgba(0,0,0,0)",
                    tickfont=dict(color="#6a1b9a", size=9),
                    title=dict(text="FSI", font=dict(color="#6a1b9a", size=10))),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#fff", bordercolor="#2e8b57",
                        font=dict(color="#1a3828", size=12))
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with side_col:
    for val, label, color, bg in [
        (mechanization, "MECH",  "#1565c0", "#e3f2fd"),
        (irrigation,    "IRRIG", "#2e8b57", "#e8f5e9")
    ]:
        st.markdown('<div class="donut-wrap">', unsafe_allow_html=True)
        fig_d = go.Figure()
        fig_d.add_trace(go.Pie(
            values=[val, 100 - val], labels=[label, "Other"],
            hole=0.68, marker=dict(colors=[color, "#e8f5e9"],
                                   line=dict(color=["#ffffff","#ffffff"], width=2)),
            textinfo="none", hoverinfo="label+percent"
        ))
        fig_d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=150, margin=dict(l=0, r=0, t=0, b=0),
            annotations=[dict(
                text=f"<b>{val:.0f}%</b><br><span style='font-size:9px;color:#6a9a7a'>{label}</span>",
                showarrow=False, font=dict(color=color, size=15)
            )]
        )
        st.plotly_chart(fig_d, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────── CLIMATE & PRODUCTION ───────────────────────────
st.markdown('<div class="section-title">Climate & Production</div>', unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    st.markdown('<div class="chart-outer">', unsafe_allow_html=True)
    fig_clim = go.Figure()
    fig_clim.add_trace(go.Bar(
        x=data["Year"], y=data["Rainfall"], name="Rainfall (mm)",
        marker=dict(color="#1565c0", line=dict(color="#0d47a1", width=1))
    ))
    fig_clim.add_trace(go.Scatter(
        x=data["Year"], y=data["Temperature"], mode="lines+markers", name="Temp (C)",
        line=dict(color="#c0392b", width=2.5),
        marker=dict(size=6, color="#c0392b", line=dict(color="#fff", width=2)),
        yaxis="y2"
    ))
    fig_clim.update_layout(
        template="plotly_white", paper_bgcolor=CHART_BG, plot_bgcolor="#fafffe",
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", x=0, y=1.15,
                    font=dict(size=10, color=TICK_COLOR), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR), zeroline=False,
                   linecolor="#a5d6a7", linewidth=2),
        yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR),
                   title=dict(text="Rainfall mm", font=dict(color=TICK_COLOR, size=10)),
                   linecolor="#a5d6a7", linewidth=2),
        yaxis2=dict(overlaying="y", side="right", tickfont=dict(color="#c0392b", size=9),
                    title=dict(text="Temp C", font=dict(color="#c0392b", size=10)),
                    gridcolor="rgba(0,0,0,0)"),
        hovermode="x unified", bargap=0.25,
        hoverlabel=dict(bgcolor="#fff", bordercolor="#2e8b57", font=dict(color="#1a3828"))
    )
    st.plotly_chart(fig_clim, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with c2:
    st.markdown('<div class="chart-outer">', unsafe_allow_html=True)
    fig_prod = go.Figure()
    fig_prod.add_trace(go.Bar(
        x=data["Year"], y=data["Production"], name="Production (MT)",
        marker=dict(color="#2e8b57", line=dict(color="#1a5c38", width=1))
    ))
    if "ProfitMargin" in data.columns:
        fig_prod.add_trace(go.Scatter(
            x=data["Year"], y=data["ProfitMargin"], mode="lines+markers",
            name="Profit Margin %",
            line=dict(color="#c87a00", width=2.5),
            marker=dict(size=6, color="#c87a00", line=dict(color="#fff", width=2)),
            yaxis="y2"
        ))
    fig_prod.update_layout(
        template="plotly_white", paper_bgcolor=CHART_BG, plot_bgcolor="#fafffe",
        height=260, margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", x=0, y=1.15,
                    font=dict(size=10, color=TICK_COLOR), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR), zeroline=False,
                   linecolor="#a5d6a7", linewidth=2),
        yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR),
                   title=dict(text="Production MT", font=dict(color=TICK_COLOR, size=10)),
                   linecolor="#a5d6a7", linewidth=2),
        yaxis2=dict(overlaying="y", side="right", tickfont=dict(color="#c87a00", size=9),
                    title=dict(text="Margin %", font=dict(color="#c87a00", size=10)),
                    gridcolor="rgba(0,0,0,0)"),
        hovermode="x unified", bargap=0.25,
        hoverlabel=dict(bgcolor="#fff", bordercolor="#2e8b57", font=dict(color="#1a3828"))
    )
    st.plotly_chart(fig_prod, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────── CROP COMPARISON ───────────────────────────
if compare_enabled and compare_crop:
    st.markdown('<div class="section-title">Crop Comparison</div>', unsafe_allow_html=True)

    data_b = df[(df["Country"] == country) & (df["Crop"] == compare_crop)].sort_values("Year")
    if "Last 5" in period:
        data_b = data_b[data_b["Year"] >= year_max - 4]
    elif "Last 3" in period:
        data_b = data_b[data_b["Year"] >= year_max - 2]

    if len(data_b) >= 2:
        st.markdown('<div class="chart-outer">', unsafe_allow_html=True)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=data["Year"], y=data["Yield"], mode="lines+markers", name=crop,
            line=dict(color="#2e8b57", width=3),
            marker=dict(size=7, color="#2e8b57", line=dict(color="#fff", width=2))
        ))
        fig_cmp.add_trace(go.Scatter(
            x=data_b["Year"], y=data_b["Yield"], mode="lines+markers", name=compare_crop,
            line=dict(color="#c0392b", width=3),
            marker=dict(size=7, color="#c0392b", line=dict(color="#fff", width=2))
        ))
        fig_cmp.update_layout(
            template="plotly_white", paper_bgcolor=CHART_BG, plot_bgcolor="#fafffe",
            height=240, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(orientation="h", x=0, y=1.15,
                        font=dict(size=11, color=TICK_COLOR), bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR), zeroline=False,
                       linecolor="#a5d6a7", linewidth=2),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR),
                       title=dict(text="MT/ha", font=dict(color=TICK_COLOR, size=10)),
                       linecolor="#a5d6a7", linewidth=2),
            hovermode="x unified",
            hoverlabel=dict(bgcolor="#fff", bordercolor="#2e8b57", font=dict(color="#1a3828"))
        )
        st.plotly_chart(fig_cmp, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        lat_b = data_b.iloc[-1]
        rows  = [
            ("Yield (MT/ha)", f"{latest['Yield']:.2f}",               f"{lat_b['Yield']:.2f}"),
            ("FSI Score",     f"{latest['FSI']:.1f}",                 f"{lat_b['FSI']:.1f}"),
            ("Rainfall (mm)", f"{latest['Rainfall']:.0f}",            f"{lat_b['Rainfall']:.0f}"),
            ("Profit Margin", f"{latest.get('ProfitMargin',0):.1f}%", f"{lat_b.get('ProfitMargin',0):.1f}%"),
        ]
        rows_html = "".join(
            f"<tr><td>{r[0]}</td>"
            f"<td style='color:#1e8449;font-weight:700'>{r[1]}</td>"
            f"<td style='color:#c0392b;font-weight:700'>{r[2]}</td></tr>"
            for r in rows
        )
        st.markdown(f"""
        <table class="compare-table">
          <thead><tr><th>Metric</th><th>{crop}</th><th>{compare_crop}</th></tr></thead>
          <tbody>{rows_html}</tbody>
        </table>
        """, unsafe_allow_html=True)

# ─────────────────────────── RISK ALERTS & INSIGHTS ───────────────────────────
st.markdown('<div class="section-title">Risk Alerts & Insights</div>', unsafe_allow_html=True)

alert_col, insight_col = st.columns([1, 1])

with alert_col:
    alerts = []
    dr = str(latest.get("DroughtRisk", "")).lower()

    if "high" in dr or "extreme" in dr:
        alerts.append(("high", "High Drought Risk",
            f"Current drought risk is classified as <b>{latest['DroughtRisk']}</b>. "
            "Consider drought-resistant varieties and water-saving irrigation."))
    elif "moderate" in dr or "medium" in dr:
        alerts.append(("medium", "Moderate Drought Risk",
            f"Drought risk is <b>{latest['DroughtRisk']}</b>. "
            "Monitor soil moisture and plan contingency irrigation."))

    if len(data) >= 2:
        fsi_trend = (data["FSI"].iloc[-1] - data["FSI"].iloc[-3]
                     if len(data) >= 3 else data["FSI"].iloc[-1] - data["FSI"].iloc[-2])
        if fsi_trend < -5:
            alerts.append(("high", "FSI Declining",
                f"Food Security Index dropped by <b>{abs(fsi_trend):.1f} pts</b>. "
                "Policy intervention may be needed."))
        elif fsi_trend < 0:
            alerts.append(("medium", "FSI Under Pressure",
                f"Food Security Index decreased by <b>{abs(fsi_trend):.1f} pts</b>. "
                "Closely monitor supply chain."))

    if loss_rate > 15:
        alerts.append(("high", "High Post-Harvest Loss",
            f"Loss rate of <b>{loss_rate:.1f}%</b> is above threshold. "
            "Invest in cold-chain and storage infrastructure."))
    elif loss_rate > 8:
        alerts.append(("medium", "Elevated Loss Rate",
            f"Loss rate of <b>{loss_rate:.1f}%</b>. "
            "Explore improved storage and transport options."))

    if pred == 1 and prob > 0.65:
        alerts.append(("low", "Strong Positive Forecast",
            f"Model predicts <b>improving FSI</b> with <b>{prob*100:.0f}% confidence</b>. "
            "Favorable conditions ahead."))

    if not alerts:
        alerts.append(("low", "All Indicators Normal",
            "No critical risk factors detected. Continue standard monitoring."))

    for level, title, body in alerts:
        st.markdown(f"""
        <div class="alert-card alert-{level}">
            <div class="alert-dot-{level}"></div>
            <div>
                <div class="alert-title">{title}</div>
                <div class="alert-body">{body}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

with insight_col:
    yield_vs_avg    = ((latest["Yield"] - avg_yield) / avg_yield * 100) if avg_yield else 0
    best_year       = data.loc[data["Yield"].idxmax(), "Year"]
    worst_year      = data.loc[data["Yield"].idxmin(), "Year"]
    rain_yield_corr = data[["Rainfall", "Yield"]].corr().iloc[0, 1] if len(data) >= 3 else 0
    corr_str        = "Positive" if rain_yield_corr > 0.3 \
                      else "Negative" if rain_yield_corr < -0.3 else "Weak"
    dir_color       = "#1e8449" if yield_vs_avg >= 0 else "#c0392b"
    dir_word        = "above" if yield_vs_avg >= 0 else "below"

    insights = [
        ("Current yield vs average",
         f"<span style='color:{dir_color};font-weight:700'>{abs(yield_vs_avg):.1f}% {dir_word}</span>"),
        ("Best performance year",
         f"<span style='color:#1e8449;font-weight:700'>{int(best_year)}</span>"
         f"<span style='color:#3a5a3a'> — {data['Yield'].max():.2f} MT/ha</span>"),
        ("Weakest performance year",
         f"<span style='color:#c0392b;font-weight:700'>{int(worst_year)}</span>"
         f"<span style='color:#3a5a3a'> — {data['Yield'].min():.2f} MT/ha</span>"),
        ("Rainfall-yield correlation",
         f"<span style='color:#1565c0;font-weight:700'>{corr_str}</span>"
         f"<span style='color:#3a5a3a'> (r = {rain_yield_corr:.2f})</span>"),
        ("Model confidence",
         f"<span style='color:#6a1b9a;font-weight:700'>{prob*100:.1f}%</span>"
         f" — {'High' if prob>0.7 else 'Moderate' if prob>0.5 else 'Low'} reliability"),
        ("Years of data analyzed",
         f"<span style='font-weight:700;color:#1a3828'>{len(data)}</span>"),
    ]

    rows_html = "".join(
        f"<div class='insight-item'>"
        f"<span class='insight-key'>{k}</span>"
        f"<span class='insight-val'>{v}</span>"
        f"</div>"
        for k, v in insights
    )
    st.markdown(f"""
    <div class="insight-card">
        <div class="insight-title">AI Insights — {country} / {crop}</div>
        {rows_html}
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────── TRADE & ECONOMICS ───────────────────────────
if "TradeBalance" in data.columns and data["TradeBalance"].sum() != 0:
    st.markdown('<div class="section-title">Trade & Economics</div>', unsafe_allow_html=True)

    t1, t2, t3 = st.columns(3)

    with t1:
        st.markdown('<div class="trade-label">Trade Balance</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-outer">', unsafe_allow_html=True)
        colors_tb = ["#2e8b57" if v >= 0 else "#c0392b" for v in data["TradeBalance"]]
        fig_trade = go.Figure(go.Bar(
            x=data["Year"], y=data["TradeBalance"],
            marker=dict(color=colors_tb, line=dict(
                color=["#1a5c38" if v >= 0 else "#7b241c" for v in data["TradeBalance"]],
                width=1.5))
        ))
        fig_trade.update_layout(
            template="plotly_white", paper_bgcolor=CHART_BG, plot_bgcolor="#fafffe",
            height=200, margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR), zeroline=False,
                       linecolor="#a5d6a7", linewidth=2),
            yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR),
                       title=dict(text="USD M", font=dict(color=TICK_COLOR, size=10)),
                       zeroline=True, zerolinecolor="#a5d6a7", zerolinewidth=2,
                       linecolor="#a5d6a7", linewidth=2),
            showlegend=False,
            hoverlabel=dict(bgcolor="#fff", bordercolor="#2e8b57", font=dict(color="#1a3828"))
        )
        st.plotly_chart(fig_trade, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with t2:
        if "Price" in data.columns and data["Price"].sum() != 0:
            st.markdown('<div class="trade-label">Price Trend ($/MT)</div>', unsafe_allow_html=True)
            st.markdown('<div class="chart-outer">', unsafe_allow_html=True)
            fig_price = go.Figure(go.Scatter(
                x=data["Year"], y=data["Price"],
                mode="lines+markers", fill="tozeroy",
                line=dict(color="#c87a00", width=3),
                marker=dict(size=7, color="#c87a00", line=dict(color="#fff", width=2)),
                fillcolor="rgba(200,122,0,0.1)"
            ))
            fig_price.update_layout(
                template="plotly_white", paper_bgcolor=CHART_BG, plot_bgcolor="#fafffe",
                height=200, margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR), zeroline=False,
                           linecolor="#a5d6a7", linewidth=2),
                yaxis=dict(gridcolor=GRID_COLOR, tickfont=dict(color=TICK_COLOR),
                           title=dict(text="$/MT", font=dict(color=TICK_COLOR, size=10)),
                           linecolor="#a5d6a7", linewidth=2),
                showlegend=False,
                hoverlabel=dict(bgcolor="#fff", bordercolor="#2e8b57", font=dict(color="#1a3828"))
            )
            st.plotly_chart(fig_price, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        st.markdown('<div class="trade-label">Profit Margin Gauge</div>', unsafe_allow_html=True)
        st.markdown('<div class="chart-outer">', unsafe_allow_html=True)
        pm       = profit_margin
        color_pm = "#1e8449" if pm > 30 else "#c87a00" if pm > 15 else "#c0392b"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pm,
            number=dict(suffix="%", font=dict(color=color_pm, size=30, family="Inter")),
            gauge=dict(
                axis=dict(range=[0, 80], tickcolor=TICK_COLOR,
                          tickfont=dict(color=TICK_COLOR)),
                bar=dict(color=color_pm, thickness=0.28),
                bgcolor="#f1f8f1",
                borderwidth=2,
                bordercolor="#a5d6a7",
                steps=[
                    dict(range=[0, 15],  color="#ffebee"),
                    dict(range=[15, 30], color="#fff8e1"),
                    dict(range=[30, 80], color="#e8f5e9"),
                ],
                threshold=dict(
                    line=dict(color=color_pm, width=3),
                    thickness=0.8, value=pm
                )
            )
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", height=200,
            margin=dict(l=20, r=20, t=30, b=10),
            font=dict(color="#1a3828", family="Inter")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────── FOOTER ───────────────────────────
st.markdown(f"""
<div class="agri-footer">
    <div class="footer-brand">
        <div class="footer-icon">AF</div>
        <span class="footer-name">AgriFlow AI — Agricultural Intelligence Platform</span>
    </div>
    <span class="footer-meta">{now_full} &nbsp;·&nbsp; Powered by RandomForest · Plotly · Streamlit</span>
</div>
""", unsafe_allow_html=True)