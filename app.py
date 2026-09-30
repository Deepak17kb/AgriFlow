import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from model import predict, prepare_features, train_model
from utils import load_data

st.set_page_config(page_title="AgriFlow AI", layout="wide", page_icon=":material/eco:")

MIN_RECORDS = 3  # the forecast needs at least two year-to-year outcomes to learn from

# Chart colors. Pairs that share a chart (crop A/B, surplus/deficit) pass the
# color-vision-deficiency separation check; green/orange does not, so they never share one.
GREEN, BLUE, VIOLET, ORANGE, RED = "#1a7f4e", "#2a78d6", "#4a3aa7", "#eb6834", "#e34948"
TREND = "#98a2b3"
INK, INK_2, MUTED = "#101828", "#475467", "#667085"
GRID, AXIS, BORDER = "#eef0f3", "#d0d5dd", "#e4e7ec"
FONT = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

_SVG = ('<svg viewBox="0 0 16 16" width="{s}" height="{s}" fill="none" stroke="currentColor" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{p}</svg>')
_PATHS = {
    "leaf":  '<path d="M3.5 12.5c0-5.5 3.5-9 9-9 0 5.5-3.5 9-9 9z"/><path d="M3.5 12.5l5-5"/>',
    "up":    '<path d="M2.5 11l4-4 2.5 2.5 4.5-5"/><path d="M10 4.5h3.5V8"/>',
    "down":  '<path d="M2.5 5l4 4 2.5-2.5 4.5 5"/><path d="M10 11.5h3.5V8"/>',
    "flat":  '<path d="M2.5 8h11"/><path d="M10.5 5l3 3-3 3"/>',
    "alert": '<path d="M8 2.5l6 10.5H2L8 2.5z"/><path d="M8 6.5v3"/><path d="M8 11.5h.01"/>',
    "check": '<circle cx="8" cy="8" r="6"/><path d="M5.5 8.2l1.8 1.8 3.2-3.5"/>',
}


def icon(name, size=16):
    return _SVG.format(s=size, p=_PATHS[name])


def html(markup):
    """Render trusted HTML. Lines are stripped so Markdown never reads indentation as a code block."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()),
                unsafe_allow_html=True)


# ─────────────────────────── STYLE ───────────────────────────
st.html("""
<style>
:root {
  --bg:#f6f7f9; --surface:#ffffff; --border:#e4e7ec; --ink:#101828; --ink-2:#475467;
  --muted:#667085; --subtle:#98a2b3; --fill:#f2f4f7;
  --accent:#1a7f4e; --accent-soft:#e9f5ee;
  --good:#067647; --good-bg:#ecfdf3; --good-line:#abefc6; --good-dot:#0ca30c;
  --warn:#b54708; --warn-bg:#fffaeb; --warn-line:#fedf89; --warn-dot:#f0a30a;
  --bad:#b42318;  --bad-bg:#fef3f2;  --bad-line:#fecdca;  --bad-dot:#d03b3b;
  --shadow:0 1px 2px rgba(16,24,40,.05);
}
[data-testid="stMainBlockContainer"] { padding-top:2.25rem; padding-bottom:3rem; max-width:1400px; }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stDecoration"] { display:none; }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p { font-size:12.5px; font-weight:600; color:var(--ink-2); }

/* Cards: st.container(key="card-…") */
[class*="st-key-card"] {
  background:var(--surface); border:1px solid var(--border); border-radius:12px;
  padding:18px 20px 10px; box-shadow:var(--shadow);
}
/* Cards alone in a column fill its height, so side-by-side cards line up */
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> [class*="st-key-card"]) { flex:1 1 auto; }
.af-card-title { font-size:14px; font-weight:600; color:var(--ink); }
.af-card-sub   { font-size:12.5px; color:var(--muted); margin-top:2px; }

/* Page header */
.af-head { display:flex; justify-content:space-between; align-items:flex-end; gap:16px; flex-wrap:wrap; margin-bottom:8px; }
.af-eyebrow { font-size:12px; font-weight:600; color:var(--accent); letter-spacing:.05em; text-transform:uppercase; margin-bottom:6px; }
.af-title { font-size:30px; font-weight:700; color:var(--ink); letter-spacing:-.02em; line-height:1.15; }
.af-title span { color:var(--subtle); font-weight:500; }
.af-meta { font-size:13px; color:var(--muted); margin-top:6px; }
.af-chips { display:flex; gap:8px; flex-wrap:wrap; }
.af-chip {
  display:inline-flex; align-items:center; gap:7px; padding:5px 11px; border-radius:999px;
  font-size:12.5px; color:var(--ink-2); background:var(--surface); border:1px solid var(--border);
}
.af-chip i { width:7px; height:7px; border-radius:50%; background:var(--subtle); }
.af-chip b { font-weight:600; }
.af-chip.good { background:var(--good-bg); border-color:var(--good-line); color:var(--good); }
.af-chip.warn { background:var(--warn-bg); border-color:var(--warn-line); color:var(--warn); }
.af-chip.bad  { background:var(--bad-bg);  border-color:var(--bad-line);  color:var(--bad); }
.af-chip.good i { background:var(--good-dot); }
.af-chip.warn i { background:var(--warn-dot); }
.af-chip.bad i  { background:var(--bad-dot); }

/* KPI strip: the 1px grid gap over a border-colored background draws the dividers */
.af-kpis {
  display:grid; grid-template-columns:repeat(5, minmax(0,1fr)); gap:1px; background:var(--border);
  border:1px solid var(--border); border-radius:12px; overflow:hidden; box-shadow:var(--shadow);
}
.af-kpi { background:var(--surface); padding:16px 20px; }
.af-kpi-label { font-size:12.5px; font-weight:500; color:var(--muted); }
.af-kpi-value { font-size:26px; font-weight:650; color:var(--ink); letter-spacing:-.02em; margin-top:6px; line-height:1.2; white-space:nowrap; }
.af-kpi-value small { font-size:13px; font-weight:500; color:var(--muted); margin-left:4px; letter-spacing:0; }
.af-delta { font-size:12.5px; font-weight:600; margin-top:6px; }
.af-delta span { color:var(--muted); font-weight:400; }
.af-delta.good { color:var(--good); } .af-delta.bad { color:var(--bad); } .af-delta.neutral { color:var(--ink-2); }
@media (max-width:1100px) {
  .af-kpis { grid-template-columns:repeat(6, minmax(0,1fr)); }
  .af-kpi { grid-column:span 2; } .af-kpi:nth-child(n+4) { grid-column:span 3; }
}
@media (max-width:640px) {
  .af-kpis { grid-template-columns:1fr 1fr; }
  .af-kpi, .af-kpi:nth-child(n+4) { grid-column:span 1; } .af-kpi:last-child { grid-column:span 2; }
}

/* Outlook */
.af-section { font-size:15px; font-weight:600; color:var(--ink); margin:18px 0 0; }
.af-section span { font-size:13px; font-weight:400; color:var(--muted); margin-left:8px; }
.af-outlook { display:grid; grid-template-columns:repeat(3, minmax(0,1fr)); gap:16px; }
.af-panel { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:18px 20px; box-shadow:var(--shadow); }
.af-panel-label { display:flex; justify-content:space-between; align-items:center; gap:8px; font-size:12.5px; font-weight:500; color:var(--muted); }
.af-tag { font-size:11px; font-weight:600; color:var(--ink-2); background:var(--fill); border-radius:6px; padding:2px 7px; white-space:nowrap; }
.af-outlook-value { display:flex; align-items:center; gap:10px; font-size:21px; font-weight:650; color:var(--ink); letter-spacing:-.01em; margin-top:12px; }
.af-outlook-value small { font-size:13px; font-weight:500; color:var(--muted); letter-spacing:0; }
.ic { flex:none; display:inline-flex; align-items:center; justify-content:center; width:28px; height:28px; border-radius:8px; }
.ic.good { background:var(--good-bg); color:var(--good); }
.ic.warn { background:var(--warn-bg); color:var(--warn); }
.ic.bad  { background:var(--bad-bg);  color:var(--bad); }
.ic.neutral { background:var(--fill); color:var(--ink-2); }
.af-bar { height:6px; border-radius:999px; background:var(--fill); margin-top:14px; overflow:hidden; }
.af-bar span { display:block; height:100%; border-radius:999px; background:var(--accent); }
.af-panel-foot { font-size:12.5px; color:var(--muted); margin-top:10px; line-height:1.5; }
.af-panel-foot b { color:var(--ink-2); font-weight:600; }
@media (max-width:900px) { .af-outlook { grid-template-columns:1fr; } }

/* Alerts */
.af-alert { display:flex; gap:12px; padding:12px 0; border-top:1px solid var(--border); }
.af-alert:first-child { border-top:none; padding-top:6px; }
.af-alert-title { display:flex; align-items:center; gap:8px; flex-wrap:wrap; font-size:13.5px; font-weight:600; color:var(--ink); }
.af-sev { font-size:11px; font-weight:600; padding:1px 8px; border-radius:999px; }
.af-sev.good { background:var(--good-bg); color:var(--good); }
.af-sev.warn { background:var(--warn-bg); color:var(--warn); }
.af-sev.bad  { background:var(--bad-bg);  color:var(--bad); }
.af-alert-body { font-size:13px; color:var(--ink-2); line-height:1.55; margin-top:3px; }

/* Key-value rows */
.af-row { display:flex; justify-content:space-between; gap:12px; padding:10px 0; border-top:1px solid var(--border); font-size:13px; }
.af-row:first-child { border-top:none; padding-top:6px; }
.af-row .k { color:var(--muted); }
.af-row .v { color:var(--ink); font-weight:600; text-align:right; }
.af-row .v small { color:var(--muted); font-weight:400; }

/* Meters */
.af-meters { display:grid; grid-template-columns:repeat(4, minmax(0,1fr)); gap:28px; padding:6px 0 8px; }
.af-meter-top { display:flex; justify-content:space-between; align-items:baseline; gap:8px; }
.af-meter-label { font-size:12.5px; font-weight:500; color:var(--muted); }
.af-meter-value { font-size:18px; font-weight:650; color:var(--ink); }
.af-meter-value small { font-size:12px; font-weight:500; color:var(--muted); margin-left:2px; }
.af-meter .af-bar { margin-top:10px; }
.af-meter-note { font-size:12px; color:var(--muted); margin-top:8px; }
@media (max-width:900px) { .af-meters { grid-template-columns:repeat(2, minmax(0,1fr)); } }

/* Comparison table */
.af-table { width:100%; border-collapse:collapse; font-size:13px; margin:4px 0 8px; }
.af-table th { font-size:12px; font-weight:600; color:var(--muted); text-align:left; padding:10px 12px; border-bottom:1px solid var(--border); background:#f9fafb; }
.af-table td { padding:10px 12px; border-bottom:1px solid var(--border); color:var(--ink); font-variant-numeric:tabular-nums; }
.af-table td:first-child { color:var(--ink-2); }
.af-table tr:last-child td { border-bottom:none; }
.af-swatch { display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:6px; }

/* Sidebar */
.af-brand { display:flex; align-items:center; gap:10px; padding-bottom:18px; margin-bottom:6px; border-bottom:1px solid var(--border); }
.af-mark { width:34px; height:34px; border-radius:9px; background:var(--accent); color:#fff; display:flex; align-items:center; justify-content:center; }
.af-brand-name { font-size:15px; font-weight:700; color:var(--ink); letter-spacing:-.01em; }
.af-brand-sub { font-size:12px; color:var(--muted); }
.af-coverage { font-size:12.5px; color:var(--muted); line-height:1.8; padding-top:14px; border-top:1px solid var(--border); }
.af-coverage b { color:var(--ink-2); font-weight:600; }

/* Empty state */
.af-empty { max-width:580px; margin:9vh auto 0; text-align:center; }
.af-empty-icon { width:48px; height:48px; border-radius:12px; background:var(--accent-soft); color:var(--accent); display:flex; align-items:center; justify-content:center; margin:0 auto 18px; }
.af-empty-title { font-size:24px; font-weight:650; color:var(--ink); letter-spacing:-.015em; }
.af-empty-sub { font-size:14px; color:var(--muted); line-height:1.6; margin-top:8px; }
.af-empty-stats { display:grid; grid-template-columns:repeat(4, 1fr); gap:1px; background:var(--border); border:1px solid var(--border); border-radius:12px; overflow:hidden; margin-top:28px; box-shadow:var(--shadow); }
.af-empty-stats div { background:var(--surface); padding:14px 8px; }
.af-empty-stats b { display:block; font-size:18px; font-weight:650; color:var(--ink); }
.af-empty-stats span { font-size:12px; color:var(--muted); }

.af-footer { display:flex; justify-content:space-between; flex-wrap:wrap; gap:8px; font-size:12px; color:var(--muted); border-top:1px solid var(--border); padding-top:16px; margin-top:24px; }
</style>
""")


# ─────────────────────────── DATA ───────────────────────────
@st.cache_resource(show_spinner=False)
def get_data():
    return load_data()  # shared and read-only: every view filters into a new frame


@st.cache_data(show_spinner=False)
def get_catalog():
    """Country -> crops with enough records to analyse."""
    counts = get_data().groupby(["Country", "Crop"]).size()
    counts = counts[counts >= MIN_RECORDS]
    return {country: sorted(g.index.get_level_values("Crop"))
            for country, g in counts.groupby(level="Country")}


def pair_records(country, crop, start_year=None):
    df = get_data()
    data = df[(df["Country"] == country) & (df["Crop"] == crop)].sort_values("Year")
    return data if start_year is None else data[data["Year"] >= start_year]


@st.cache_data(show_spinner=False)
def forecast(country, crop, start_year):
    data_model = prepare_features(pair_records(country, crop, start_year))
    model, features = train_model(data_model)
    pred, p_up = predict(model, data_model, features)
    return pred, p_up, model is not None, int(data_model["target"].notna().sum())


def yield_trend(data, horizons=(1, 3)):
    """Linear yield trend and projections, each with a ±1 standard-error range."""
    x = data["Year"].to_numpy(float)
    y = data["Yield"].to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1)
    resid = y - (slope * x + intercept)
    s = np.sqrt((resid ** 2).sum() / max(len(x) - 2, 1))
    sxx = ((x - x.mean()) ** 2).sum()
    projections = []
    for h in horizons:
        x0 = x[-1] + h
        mid = slope * x0 + intercept
        se = s * np.sqrt(1 + 1 / len(x) + (x0 - x.mean()) ** 2 / sxx)
        projections.append((int(x0), max(mid, 0.0), max(mid - se, 0.0), max(mid + se, 0.0)))
    return slope, intercept, projections


# ─────────────────────────── FORMATTING ───────────────────────────
def compact(v):
    a = abs(v)
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e3:
        return f"{v / 1e3:.1f}K"
    return f"{v:,.0f}"


def money_m(v):
    """USD millions for Markdown; '&#36;' keeps '$' pairs from being read as LaTeX."""
    sign, a = ("−" if v < 0 else ""), abs(v)
    return f"{sign}&#36;{a / 1000:.2f}B" if a >= 1000 else f"{sign}&#36;{a:,.0f}M"


def tone_for(value, table):
    return next((tone for key, tone in table if key in str(value).lower()), "neutral")


FS_TONES = [("food insecure", "bad"), ("moderately insecure", "warn"), ("moderately secure", "neutral"),
            ("secure", "good")]
DROUGHT_TONES = [("critical", "bad"), ("extreme", "bad"), ("high", "bad"), ("moderate", "warn"),
                 ("medium", "warn"), ("low", "good")]


def correlation_label(r):
    if not np.isfinite(r):
        return "n/a"
    strength = "Strong" if abs(r) >= 0.7 else "Moderate" if abs(r) >= 0.4 else "Weak" if abs(r) >= 0.2 else "No clear"
    direction = "" if strength == "No clear" else (" positive" if r > 0 else " negative")
    return f"{strength}{direction}"


# ─────────────────────────── CHART HELPERS ───────────────────────────
def render_chart(fig, years=None, height=230, legend=False, y_range=None, zero_line=False,
                 hovermode="x unified"):
    fig.update_layout(
        height=height,
        margin=dict(l=4, r=8, t=32 if legend else 6, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=12, color=INK_2),
        showlegend=legend,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom",
                    font=dict(size=12, color=INK_2), bgcolor="rgba(0,0,0,0)",
                    itemclick=False, itemdoubleclick=False),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor="#ffffff", bordercolor=BORDER, font=dict(family=FONT, size=12, color=INK)),
        bargap=0.35, barcornerradius=4,
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=AXIS, ticks="", zeroline=False,
                     tickfont=dict(color=MUTED, size=11), fixedrange=True, automargin=True)
    if years is not None:
        span = int(years.max() - years.min())
        fig.update_xaxes(tickformat="d", dtick=1 if span <= 8 else 2)
    fig.update_yaxes(gridcolor=GRID, showline=False, ticks="", tickfont=dict(color=MUTED, size=11),
                     zeroline=zero_line, zerolinecolor=AXIS, zerolinewidth=1, range=y_range,
                     separatethousands=True, fixedrange=True, automargin=True)
    st.plotly_chart(fig, theme=None, config={"displayModeBar": False})


def line(x, y, name, color, unit, fmt=".2f", symbol="circle"):
    return go.Scatter(
        x=x, y=y, name=name, mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=8, color=color, symbol=symbol, line=dict(color="#ffffff", width=2)),
        hovertemplate=f"%{{y:{fmt}}} {unit}<extra>{name}</extra>",
    )


def bars(x, y, name, color, unit, fmt=",.0f"):
    return go.Bar(x=x, y=y, name=name, marker=dict(color=color),
                  hovertemplate=f"%{{y:{fmt}}} {unit}<extra>{name}</extra>")


def card(key, title, subtitle=None):
    box = st.container(key=f"card-{key}")
    with box:
        sub = f'<div class="af-card-sub">{subtitle}</div>' if subtitle else ""
        html(f'<div><div class="af-card-title">{title}</div>{sub}</div>')
    return box


# ─────────────────────────── SIDEBAR ───────────────────────────
df = get_data()
catalog = get_catalog()
countries = sorted(catalog)

# Deep links: ?country=India&crop=Wheat opens straight into that analysis.
if "country" not in st.session_state:
    q_country, q_crop = st.query_params.get("country"), st.query_params.get("crop")
    if q_country in catalog:
        st.session_state.country = q_country
        if q_crop in catalog[q_country]:
            st.session_state.crop = q_crop
            st.session_state.ready = True
    else:
        st.session_state.country = "India" if "India" in catalog else countries[0]

with st.sidebar:
    html(f"""
    <div class="af-brand">
        <div class="af-mark">{icon("leaf", 18)}</div>
        <div>
            <div class="af-brand-name">AgriFlow AI</div>
            <div class="af-brand-sub">Crop &amp; food security analytics</div>
        </div>
    </div>
    """)

    country = st.selectbox("Country", countries, key="country")

    crops = catalog[country]
    if st.session_state.get("crop") not in crops:
        st.session_state.crop = crops[0]
    crop = st.selectbox("Crop", crops, key="crop")

    pair_all = pair_records(country, crop)
    first_year, last_year = int(pair_all["Year"].min()), int(pair_all["Year"].max())
    periods = {"All years": first_year}
    for label, span in [("Last 5 yrs", 5), ("Last 3 yrs", 3)]:
        start = last_year - span + 1
        if start > first_year and (pair_all["Year"] >= start).sum() >= MIN_RECORDS:
            periods[label] = start
    if st.session_state.get("period") not in periods:
        st.session_state.period = "All years"
    period = st.segmented_control(
        "Period", list(periods), key="period", required=True,
        help=f"Shorter periods appear when they hold at least {MIN_RECORDS} records.")
    start_year = periods.get(period, first_year)
    n_period = int((pair_all["Year"] >= start_year).sum())
    st.caption(f"{start_year}–{last_year} · {n_period} records")

    others = [c for c in crops if c != crop]
    compare_crop = None
    if st.toggle("Compare with another crop", key="compare_on", disabled=not others):
        if st.session_state.get("compare_crop") not in others:
            st.session_state.compare_crop = others[0]
        compare_crop = st.selectbox("Comparison crop", others, key="compare_crop")

    if st.button("Run analysis", type="primary", icon=":material/insights:", width="stretch"):
        st.session_state.ready = True

    html(f"""
    <div class="af-coverage">
        <b>{df["Country"].nunique()}</b> countries · <b>{df["Crop"].nunique()}</b> crops<br>
        <b>{int(df["Year"].min())}–{int(df["Year"].max())}</b> · <b>{len(df):,}</b> records<br>
        <b>{sum(len(v) for v in catalog.values()):,}</b> country–crop pairs with {MIN_RECORDS}+ records
    </div>
    """)

now = datetime.datetime.now()

# ─────────────────────────── EMPTY STATE ───────────────────────────
if not st.session_state.get("ready"):
    html(f"""
    <div class="af-empty">
        <div class="af-empty-icon">{icon("leaf", 22)}</div>
        <div class="af-empty-title">Analyse a crop</div>
        <div class="af-empty-sub">Choose a country and crop in the sidebar, then select
        <b>Run analysis</b> for yield trends, the food security outlook and risk alerts.</div>
        <div class="af-empty-stats">
            <div><b>{df["Country"].nunique()}</b><span>countries</span></div>
            <div><b>{df["Crop"].nunique()}</b><span>crops</span></div>
            <div><b>{int(df["Year"].max()) - int(df["Year"].min()) + 1}</b><span>years</span></div>
            <div><b>{len(df):,}</b><span>records</span></div>
        </div>
    </div>
    """)
    st.stop()

st.query_params.from_dict({"country": country, "crop": crop})

# ─────────────────────────── ANALYSIS ───────────────────────────
data = pair_records(country, crop, start_year)
latest, prev = data.iloc[-1], data.iloc[-2]
year_val, prev_year = int(latest["Year"]), int(prev["Year"])
years = data["Year"]

pred, p_up, used_model, n_outcomes = forecast(country, crop, start_year)
slope, intercept, projections = yield_trend(data)

avg_yield = data["Yield"].mean()
loss_rate = data["LossRate"].mean()
profit_margin = data["ProfitMargin"].mean()

# ─────────────────────────── HEADER ───────────────────────────
fs_status = str(latest["FSStatus"])
drought = str(latest["DroughtRisk"])
region = latest.get("region")
eyebrow = f"{region} · Crop intelligence" if isinstance(region, str) else "Crop intelligence"

html(f"""
<div class="af-head">
    <div>
        <div class="af-eyebrow">{eyebrow}</div>
        <div class="af-title">{crop} <span>in</span> {country}</div>
        <div class="af-meta">{start_year}–{year_val} · {len(data)} annual records · latest record {year_val}</div>
    </div>
    <div class="af-chips">
        <span class="af-chip {tone_for(fs_status, FS_TONES)}"><i></i>{fs_status}</span>
        <span class="af-chip {tone_for(drought, DROUGHT_TONES)}"><i></i>Drought risk <b>{drought}</b></span>
    </div>
</div>
""")


# ─────────────────────────── KPI STRIP ───────────────────────────
def delta_html(change, text, good_when="up"):
    if not np.isfinite(change) or change == 0:
        return f'<div class="af-delta neutral">No change <span>vs {prev_year}</span></div>'
    arrow = "▲" if change > 0 else "▼"
    tone = "neutral" if good_when is None else ("good" if (change > 0) == (good_when == "up") else "bad")
    return f'<div class="af-delta {tone}">{arrow} {text} <span>vs {prev_year}</span></div>'


def pct_delta(cur, old, good_when="up"):
    change = (cur - old) / abs(old) * 100 if old else float("nan")
    return delta_html(change, f"{abs(change):.1f}%", good_when)


kpis = [
    ("Yield", f"{latest['Yield']:.2f}", "MT/ha", pct_delta(latest["Yield"], prev["Yield"])),
    ("Food security index", f"{latest['FSI']:.0f}", "/ 100",
     delta_html(latest["FSI"] - prev["FSI"], f"{abs(latest['FSI'] - prev['FSI']):.0f} pts")),
    ("Production", compact(latest["Production"]), "MT", pct_delta(latest["Production"], prev["Production"])),
    ("Price", f"&#36;{latest['Price']:,.0f}", "/ MT", pct_delta(latest["Price"], prev["Price"], None)),
    ("Trade balance", money_m(latest["TradeBalance"]), "",
     delta_html(latest["TradeBalance"] - prev["TradeBalance"],
                money_m(abs(latest["TradeBalance"] - prev["TradeBalance"])))),
]
html('<div class="af-kpis">' + "".join(
    f'<div class="af-kpi"><div class="af-kpi-label">{label}</div>'
    f'<div class="af-kpi-value">{value}<small>{unit}</small></div>{delta}</div>'
    for label, value, unit, delta in kpis) + "</div>")

# ─────────────────────────── OUTLOOK ───────────────────────────
if p_up >= 0.6:
    fsi_tone, fsi_icon, fsi_text = "good", "up", "Likely to improve"
elif p_up <= 0.4:
    fsi_tone, fsi_icon, fsi_text = "bad", "down", "Likely to decline"
else:
    fsi_tone, fsi_icon, fsi_text = "neutral", "flat", "Too close to call"

method = (f"RandomForest trained on <b>{n_outcomes}</b> year-to-year changes" if used_model
          else f"History shows one outcome only, so this is the smoothed share of past rises ({n_outcomes} changes)")

(y1, p1, lo1, hi1), (y3, p3, lo3, hi3) = projections
trend_word = "rising" if slope > 0 else "falling" if slope < 0 else "flat"


def projection_panel(label, year, mid, lo, hi, horizon):
    tone = "good" if mid > latest["Yield"] else "bad" if mid < latest["Yield"] else "neutral"
    arrow = "up" if tone == "good" else "down" if tone == "bad" else "flat"
    change = (mid - latest["Yield"]) / latest["Yield"] * 100 if latest["Yield"] else 0
    return f"""
    <div class="af-panel">
        <div class="af-panel-label">{label}<span class="af-tag">{horizon}</span></div>
        <div class="af-outlook-value"><span class="ic {tone}">{icon(arrow)}</span>
            {mid:.2f}<small>MT/ha · {change:+.1f}%</small></div>
        <div class="af-panel-foot">Likely range <b>{lo:.2f}–{hi:.2f}</b> MT/ha by {year}.<br>
        Linear trend, {trend_word} <b>{slope:+.3f}</b> MT/ha per year.</div>
    </div>"""


html(f'<div class="af-section">Outlook<span>Forecasts from the {start_year}–{year_val} records</span></div>')
html(f"""
<div class="af-outlook">
    <div class="af-panel">
        <div class="af-panel-label">Food security index<span class="af-tag">Next record</span></div>
        <div class="af-outlook-value"><span class="ic {fsi_tone}">{icon(fsi_icon)}</span>{fsi_text}</div>
        <div class="af-bar"><span style="width:{p_up * 100:.0f}%"></span></div>
        <div class="af-panel-foot"><b>{p_up * 100:.0f}%</b> chance the index rises from {latest['FSI']:.0f}.<br>{method}.</div>
    </div>
    {projection_panel("Yield projection", y1, p1, lo1, hi1, str(y1))}
    {projection_panel("Yield projection", y3, p3, lo3, hi3, str(y3))}
</div>
""")

# ─────────────────────────── ALERTS & INSIGHTS ───────────────────────────
alerts = []
drought_level = drought.lower()
if drought_level in ("critical", "extreme", "high"):
    alerts.append(("bad", f"{drought} drought risk", "High",
                   "Prioritise drought-tolerant varieties and water-saving irrigation."))
elif drought_level in ("moderate", "medium"):
    alerts.append(("warn", "Moderate drought risk", "Medium",
                   "Monitor soil moisture and plan contingency irrigation."))

fsi_change = data["FSI"].iloc[-1] - data["FSI"].iloc[-3 if len(data) >= 3 else -2]
if fsi_change < -5:
    alerts.append(("bad", "Food security index falling", "High",
                   f"Down <b>{abs(fsi_change):.0f} pts</b> over the last records. Policy support may be needed."))
elif fsi_change < 0:
    alerts.append(("warn", "Food security index under pressure", "Medium",
                   f"Down <b>{abs(fsi_change):.0f} pts</b> over the last records. Watch supply closely."))

if loss_rate > 15:
    alerts.append(("bad", "High post-harvest loss", "High",
                   f"Average loss of <b>{loss_rate:.1f}%</b>. Invest in cold chain and storage."))
elif loss_rate > 8:
    alerts.append(("warn", "Elevated post-harvest loss", "Medium",
                   f"Average loss of <b>{loss_rate:.1f}%</b>. Review storage and transport."))

if p_up >= 0.65:
    alerts.append(("good", "Positive food security outlook", "Info",
                   f"<b>{p_up * 100:.0f}%</b> chance the index rises by the next record."))
elif p_up <= 0.35:
    alerts.append(("warn", "Negative food security outlook", "Medium",
                   f"Only <b>{p_up * 100:.0f}%</b> chance the index rises by the next record."))

if not alerts:
    alerts.append(("good", "All indicators normal", "Info",
                   "No critical risk factors detected. Continue standard monitoring."))

yield_vs_avg = (latest["Yield"] - avg_yield) / avg_yield * 100 if avg_yield else 0
best = data.loc[data["Yield"].idxmax()]
worst = data.loc[data["Yield"].idxmin()]
rain_r = data["Rainfall"].corr(data["Yield"]) if data["Rainfall"].nunique() > 1 else float("nan")
r_text = f" <small>(r = {rain_r:.2f})</small>" if np.isfinite(rain_r) else ""

insights = [
    ("Latest yield vs average", f"{abs(yield_vs_avg):.1f}% {'above' if yield_vs_avg >= 0 else 'below'}"),
    ("Best year", f"{int(best['Year'])} <small>· {best['Yield']:.2f} MT/ha</small>"),
    ("Weakest year", f"{int(worst['Year'])} <small>· {worst['Yield']:.2f} MT/ha</small>"),
    ("Rainfall–yield link", f"{correlation_label(rain_r)}{r_text}"),
    ("Average profit margin", f"{profit_margin:.1f}%"),
    ("Records analysed", f"{len(data)} <small>· {start_year}–{year_val}</small>"),
]


def rows_html(pairs):
    return "".join(f'<div class="af-row"><span class="k">{k}</span><span class="v">{v}</span></div>'
                   for k, v in pairs)


# ─────────────────────────── TABS ───────────────────────────
st.write("")
tab_overview, tab_climate, tab_econ, tab_compare, tab_data = st.tabs(
    ["Overview", "Climate", "Economics", "Compare", "Data"])

with tab_overview:
    left, right = st.columns([2, 1], gap="medium")
    with left:
        with card("yield", "Yield", "MT per hectare, with linear trend and projections"):
            fit_x = [years.iloc[0], year_val]
            fig = go.Figure()
            fig.add_scatter(x=fit_x, y=[slope * x + intercept for x in fit_x], name="Trend", mode="lines",
                            line=dict(color=TREND, width=1.5), hoverinfo="skip")
            fig.add_scatter(x=[year_val, y3], y=[slope * year_val + intercept, slope * y3 + intercept],
                            mode="lines", line=dict(color=TREND, width=1.5, dash="dot"),
                            hoverinfo="skip", showlegend=False)
            fig.add_scatter(x=[y1, y3], y=[p1, p3], name="Projection", mode="markers",
                            marker=dict(size=9, color="#ffffff", line=dict(color=INK_2, width=2)),
                            error_y=dict(type="data", symmetric=False, array=[hi1 - p1, hi3 - p3],
                                         arrayminus=[p1 - lo1, p3 - lo3], color=TREND, thickness=1.5, width=5),
                            hovertemplate="%{y:.2f} MT/ha<extra>Projection</extra>")
            fig.add_trace(line(years, data["Yield"], "Yield", GREEN, "MT/ha"))
            render_chart(fig, years=np.array([years.iloc[0], y3]), height=290, legend=True)

    with right:
        with card("alerts", "Risk alerts", f"{len(alerts)} active for the latest record"):
            html("".join(
                f'<div class="af-alert"><span class="ic {tone}">{icon("check" if tone == "good" else "alert")}</span>'
                f'<div><div class="af-alert-title">{title}<span class="af-sev {tone}">{sev}</span></div>'
                f'<div class="af-alert-body">{body}</div></div></div>'
                for tone, title, sev, body in alerts))

    left, right = st.columns([2, 1], gap="medium")
    with left:
        with card("fsi", "Food security index", "Score out of 100"):
            fig = go.Figure(line(years, data["FSI"], "FSI", VIOLET, "", fmt=".0f"))
            render_chart(fig, years=years, height=250, y_range=[0, 100])
    with right:
        with card("insights", "Insights", f"{crop} in {country}, {start_year}–{year_val}"):
            html(rows_html(insights))

    meters = [
        ("Mechanization", latest["Mechanization"], "%", "Share of farm work mechanized", latest["Mechanization"]),
        ("Irrigation", latest["Irrigation"], "%", "Share of area irrigated", latest["Irrigation"]),
        ("Soil health", latest["SoilHealth"], "/ 100", "Composite soil score", latest["SoilHealth"]),
        ("Post-harvest loss", latest["LossRate"], "%", "Lower is better", min(latest["LossRate"] * 2.5, 100)),
    ]
    with card("ops", "Farm operations", f"Latest record, {year_val}"):
        html('<div class="af-meters">' + "".join(
            f'<div class="af-meter"><div class="af-meter-top"><span class="af-meter-label">{label}</span>'
            f'<span class="af-meter-value">{value:.0f}<small>{unit}</small></span></div>'
            f'<div class="af-bar"><span style="width:{width:.0f}%"></span></div>'
            f'<div class="af-meter-note">{note}</div></div>'
            for label, value, unit, note, width in meters) + "</div>")

with tab_climate:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with card("rain", "Annual rainfall", "Millimetres"):
            render_chart(go.Figure(bars(years, data["Rainfall"], "Rainfall", BLUE, "mm")), years=years)
    with c2:
        with card("temp", "Average temperature", "Degrees Celsius"):
            render_chart(go.Figure(line(years, data["Temperature"], "Temperature", ORANGE, "°C", fmt=".1f")),
                         years=years)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with card("scatter", "Rainfall vs yield",
                  "Each point is one year" + (f" · r = {rain_r:.2f}" if np.isfinite(rain_r) else "")):
            fig = go.Figure()
            if np.isfinite(rain_r):
                m, b = np.polyfit(data["Rainfall"], data["Yield"], 1)
                rx = np.array([data["Rainfall"].min(), data["Rainfall"].max()])
                fig.add_scatter(x=rx, y=m * rx + b, mode="lines", line=dict(color=TREND, width=1.5),
                                hoverinfo="skip")
            fig.add_scatter(x=data["Rainfall"], y=data["Yield"], mode="markers", customdata=years,
                            marker=dict(size=10, color=GREEN, line=dict(color="#ffffff", width=2)),
                            hovertemplate="%{customdata}<br>Rainfall %{x:,.0f} mm<br>Yield %{y:.2f} MT/ha<extra></extra>")
            fig.update_xaxes(title=dict(text="Rainfall (mm)", font=dict(size=11, color=MUTED)))
            render_chart(fig, height=250, hovermode="closest")
    with c2:
        with card("profile", "Climate profile", f"Latest record, {year_val}"):
            profile = [("Drought risk", drought)]
            for col, label in [("climate_zone", "Climate zone"),
                               ("climate_vulnerability", "Climate vulnerability score"),
                               ("flood_risk_score", "Flood risk score"),
                               ("drought_freq_per10yr", "Droughts per decade")]:
                if col in latest.index:
                    profile.append((label, latest[col]))
            profile.append(("Average rainfall", f"{data['Rainfall'].mean():,.0f} <small>mm</small>"))
            profile.append(("Average temperature", f"{data['Temperature'].mean():.1f} <small>°C</small>"))
            html(rows_html(profile))

with tab_econ:
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with card("prod", "Production", "Metric tonnes"):
            render_chart(go.Figure(bars(years, data["Production"], "Production", GREEN, "MT")), years=years)
    with c2:
        with card("price", "Price", "USD per metric tonne"):
            render_chart(go.Figure(line(years, data["Price"], "Price", GREEN, "USD/MT", fmt=",.0f")),
                         years=years)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with card("margin", "Profit margin", "Percent of revenue"):
            render_chart(go.Figure(bars(years, data["ProfitMargin"], "Profit margin", GREEN, "%", fmt=".1f")),
                         years=years, zero_line=True)
    with c2:
        with card("trade", "Trade balance", "Exports minus imports, USD millions"):
            surplus = data["TradeBalance"].where(data["TradeBalance"] >= 0)
            deficit = data["TradeBalance"].where(data["TradeBalance"] < 0)
            fig = go.Figure([bars(years, surplus, "Surplus", BLUE, "USD M"),
                             bars(years, deficit, "Deficit", RED, "USD M")])
            fig.update_layout(barmode="relative")
            render_chart(fig, years=years, legend=True, zero_line=True)

with tab_compare:
    if not compare_crop:
        st.info("Turn on **Compare with another crop** in the sidebar to compare yields.",
                icon=":material/compare_arrows:")
    else:
        data_b = pair_records(country, compare_crop, start_year)
        data_b = data_b[data_b["Year"] <= year_val]
        if len(data_b) < 2:
            st.info(f"{compare_crop} has fewer than 2 records in {start_year}–{year_val}. Try a longer period.",
                    icon=":material/info:")
        else:
            with card("compare", f"Yield: {crop} vs {compare_crop}", "MT per hectare"):
                fig = go.Figure([line(years, data["Yield"], crop, GREEN, "MT/ha"),
                                 line(data_b["Year"], data_b["Yield"], compare_crop, BLUE, "MT/ha",
                                      symbol="square")])
                span_years = np.array([min(years.min(), data_b["Year"].min()), year_val])
                render_chart(fig, years=span_years, height=280, legend=True)

            slope_b = np.polyfit(data_b["Year"], data_b["Yield"], 1)[0]
            lat_b = data_b.iloc[-1]
            rows = [
                ("Latest record", f"{year_val}", f"{int(lat_b['Year'])}"),
                ("Yield, latest (MT/ha)", f"{latest['Yield']:.2f}", f"{lat_b['Yield']:.2f}"),
                ("Yield, period average", f"{avg_yield:.2f}", f"{data_b['Yield'].mean():.2f}"),
                ("Yield trend (MT/ha per year)", f"{slope:+.3f}", f"{slope_b:+.3f}"),
                ("Food security index, latest", f"{latest['FSI']:.0f}", f"{lat_b['FSI']:.0f}"),
                ("Profit margin, average", f"{profit_margin:.1f}%", f"{data_b['ProfitMargin'].mean():.1f}%"),
                ("Post-harvest loss, average", f"{loss_rate:.1f}%", f"{data_b['LossRate'].mean():.1f}%"),
            ]
            with card("compare-table", "Side by side", f"{start_year}–{year_val}"):
                html(f"""
                <table class="af-table">
                    <thead><tr><th>Metric</th>
                    <th><span class="af-swatch" style="background:{GREEN}"></span>{crop}</th>
                    <th><span class="af-swatch" style="background:{BLUE}"></span>{compare_crop}</th></tr></thead>
                    <tbody>{"".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)}</tbody>
                </table>
                """)

with tab_data:
    table = data[["Year", "Yield", "Production", "FSI", "FSStatus", "DroughtRisk", "Rainfall", "Temperature",
                  "Price", "TradeBalance", "ProfitMargin", "LossRate", "Irrigation", "Mechanization",
                  "SoilHealth"]]
    st.dataframe(
        table, hide_index=True, width="stretch",
        column_config={
            "Year": st.column_config.NumberColumn(format="%d"),
            "Yield": st.column_config.NumberColumn("Yield (MT/ha)", format="%.2f"),
            "Production": st.column_config.NumberColumn("Production (MT)", format="localized"),
            "FSI": st.column_config.NumberColumn("FSI", format="%.0f"),
            "FSStatus": "Food security status",
            "DroughtRisk": "Drought risk",
            "Rainfall": st.column_config.NumberColumn("Rainfall (mm)", format="localized"),
            "Temperature": st.column_config.NumberColumn("Temp (°C)", format="%.1f"),
            "Price": st.column_config.NumberColumn("Price (USD/MT)", format="localized"),
            "TradeBalance": st.column_config.NumberColumn("Trade balance (USD M)", format="localized"),
            "ProfitMargin": st.column_config.NumberColumn("Profit margin (%)", format="%.1f"),
            "LossRate": st.column_config.NumberColumn("Loss rate (%)", format="%.1f"),
            "Irrigation": st.column_config.NumberColumn("Irrigation (%)", format="%.0f"),
            "Mechanization": st.column_config.NumberColumn("Mechanization (%)", format="%.0f"),
            "SoilHealth": st.column_config.NumberColumn("Soil health", format="%.0f"),
        },
    )
    st.download_button("Download CSV", table.to_csv(index=False),
                       file_name=f"agriflow_{country}_{crop}_{start_year}-{year_val}.csv".replace(" ", "_"),
                       mime="text/csv", icon=":material/download:")

# ─────────────────────────── FOOTER ───────────────────────────
html(f"""
<div class="af-footer">
    <span>AgriFlow AI · Crop &amp; food security analytics</span>
    <span>Forecast: RandomForest per country and crop · Yield: linear trend · {now.strftime("%d %b %Y")}</span>
</div>
""")
