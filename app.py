import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import charts
from model import predict, prepare_features, train_model
from ui import THEMES, callout, card, html, icon, ring, sparkline, theme_css
from utils import ISO3, load_data

st.set_page_config(page_title="AgriFlow AI", layout="wide", page_icon=":material/eco:")

MIN_RECORDS = 3  # the forecast needs at least two year-to-year outcomes to learn from
TABS = ["Overview", "Climate", "Economics", "Benchmark", "Portfolio", "Compare", "Data"]
PERIODS = [("Last 5 yrs", 5), ("Last 3 yrs", 3)]
METRICS = {  # benchmark metric -> (column, label, number format)
    "Yield": ("Yield", "Yield (MT/ha)", ",.2f"),
    "Food security": ("FSI", "FSI score", ",.0f"),
    "Profit margin": ("ProfitMargin", "Margin (%)", ",.1f"),
    "Production": ("Production", "Production (MT)", ",.0f"),
}
THEME_LABELS = {"light": ":material/light_mode: Light", "dark": ":material/dark_mode: Dark"}


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
def crop_rows(crop):
    df = get_data()
    return df[df["Crop"] == crop]


@st.cache_data(show_spinner=False)
def country_rows(country):
    df = get_data()
    return df[df["Country"] == country]


@st.cache_data(show_spinner=False)
def crop_stats(crop):
    """Per-country averages for one crop, over countries it can be analysed in."""
    eligible = [c for c, crops in get_catalog().items() if crop in crops]
    d = crop_rows(crop)
    stats = (d[d["Country"].isin(eligible)].groupby("Country")
             .agg(Yield=("Yield", "mean"), FSI=("FSI", "mean"), ProfitMargin=("ProfitMargin", "mean"),
                  Production=("Production", "mean"), LossRate=("LossRate", "mean"),
                  Irrigation=("Irrigation", "mean"), Mechanization=("Mechanization", "mean"),
                  SoilHealth=("SoilHealth", "mean"), Region=("region", "first"))
             .reset_index())
    stats["iso3"] = stats["Country"].map(ISO3)
    return stats


def percentiles(stats, country):
    row = stats.loc[stats["Country"] == country].iloc[0]
    out = {}
    for col, label, higher_is_better in charts.PROFILE:
        s = stats[col]
        pct = ((s < row[col]).mean() + (s == row[col]).mean() / 2) * 100
        out[label] = pct if higher_is_better else 100 - pct
    return out


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


FS_TONES = [("food insecure", "bad"), ("moderately insecure", "warn"), ("moderately secure", "neutral"),
            ("secure", "good")]
DROUGHT_TONES = [("critical", "bad"), ("extreme", "bad"), ("high", "bad"), ("moderate", "warn"),
                 ("medium", "warn"), ("low", "good")]


def tone_for(value, table):
    return next((tone for key, tone in table if key in str(value).lower()), "neutral")


def correlation_label(r):
    if not np.isfinite(r):
        return "n/a"
    strength = "Strong" if abs(r) >= 0.7 else "Moderate" if abs(r) >= 0.4 else "Weak" if abs(r) >= 0.2 else "No clear"
    return strength if strength == "No clear" else f"{strength} {'positive' if r > 0 else 'negative'}"


def rows_html(pairs):
    return "".join(f'<div class="af-row"><span class="k">{k}</span><span class="v">{v}</span></div>'
                   for k, v in pairs)


# ─────────────────────────── STATE ───────────────────────────
catalog = get_catalog()
countries = sorted(catalog)

# First load: restore the view from the URL (?country=India&crop=Wheat&theme=light&tab=Benchmark).
if "booted" not in st.session_state:
    qp = st.query_params
    st.session_state.booted = True
    st.session_state.theme = qp.get("theme") if qp.get("theme") in THEMES else "dark"
    st.session_state.country = qp.get("country") if qp.get("country") in catalog else (
        "India" if "India" in catalog else countries[0])
    crops0 = catalog[st.session_state.country]
    st.session_state.crop = qp.get("crop") if qp.get("crop") in crops0 else ("Wheat" if "Wheat" in crops0 else crops0[0])
    st.session_state.period = qp.get("period", "All years")
    st.session_state.start_tab = qp.get("tab") if qp.get("tab") in TABS else TABS[0]

mode = st.session_state.theme
t = THEMES[mode]
st.html(theme_css(mode))

df = get_data()

# ─────────────────────────── SIDEBAR ───────────────────────────
with st.sidebar:
    html(f"""
    <div class="af-brand">
        <div class="af-mark">{icon("leaf", 19)}</div>
        <div><div class="af-brand-name">AgriFlow <span>AI</span></div>
        <div class="af-brand-sub">crop intelligence platform</div></div>
    </div>
    <div class="af-side-title">Filters</div>
    """)
    country = st.selectbox("Country", countries, key="country")

    crops = catalog[country]
    if st.session_state.get("crop") not in crops:
        st.session_state.crop = "Wheat" if "Wheat" in crops else crops[0]
    crop = st.selectbox("Crop", crops, key="crop")

    pair_all = pair_records(country, crop)
    first_year, last_year = int(pair_all["Year"].min()), int(pair_all["Year"].max())
    periods = {"All years": first_year}
    for label, span in PERIODS:
        start = last_year - span + 1
        if start > first_year and (pair_all["Year"] >= start).sum() >= MIN_RECORDS:
            periods[label] = start
    if st.session_state.get("period") not in periods:
        st.session_state.period = "All years"
    period = st.segmented_control("Period", list(periods), key="period", required=True)
    start_year = periods.get(period, first_year)
    st.caption(f"{start_year}–{last_year} · {int((pair_all['Year'] >= start_year).sum())} records")

    html(f"""
    <div class="af-side-title">Dataset</div>
    <div class="af-cover">
        <div><b>{df["Country"].nunique()}</b><span>countries</span></div>
        <div><b>{df["Crop"].nunique()}</b><span>crops</span></div>
        <div><b>{int(df["Year"].min())}–{str(int(df["Year"].max()))[2:]}</b><span>years</span></div>
        <div><b>{compact(len(df))}</b><span>records</span></div>
    </div>
    <div class="af-hint">Views update the moment you change a filter. The address bar always holds a
    <b>shareable link</b> to this exact view.</div>
    """)

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
fs_status, drought = str(latest["FSStatus"]), str(latest["DroughtRisk"])
region = latest.get("region")

# ─────────────────────────── HEADER ───────────────────────────
head_l, head_r = st.columns([5, 1.4], vertical_alignment="top")
with head_r:
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.segmented_control("Theme", list(THEME_LABELS), key="theme", required=True,
                             format_func=THEME_LABELS.get, label_visibility="collapsed")
with head_l:
    html(f"""
    <div class="af-head">
        <div class="af-eyebrow"><span class="af-live"></span>{region if isinstance(region, str) else "Global"} · crop intelligence</div>
        <div class="af-title"><span class="crop">{crop}</span> <span class="in">in</span> {country}</div>
        <div class="af-meta">
            <span class="af-meta-text">{start_year}–{year_val} · {len(data)} annual records</span>
            <span class="af-chip {tone_for(fs_status, FS_TONES)}"><i></i>{fs_status}</span>
            <span class="af-chip {tone_for(drought, DROUGHT_TONES)}"><i></i>Drought risk <b>{drought}</b></span>
        </div>
    </div>
    """)


# ─────────────────────────── KPI TILES ───────────────────────────
def delta(change, text, good_when="up"):
    if not np.isfinite(change) or change == 0:
        return "neutral", "● 0"
    tone = "neutral" if good_when is None else ("good" if (change > 0) == (good_when == "up") else "bad")
    return tone, f"{'▲' if change > 0 else '▼'} {text}"


def pct(cur, old, good_when="up"):
    change = (cur - old) / abs(old) * 100 if old else float("nan")
    return delta(change, f"{abs(change):.1f}%", good_when)


fsi_change = latest["FSI"] - prev["FSI"]
trade_change = latest["TradeBalance"] - prev["TradeBalance"]
kpis = [
    ("Yield", f"{latest['Yield']:.2f}", "MT/ha", pct(latest["Yield"], prev["Yield"]), data["Yield"]),
    ("Food security", f"{latest['FSI']:.0f}", "/ 100", delta(fsi_change, f"{abs(fsi_change):.0f} pts"), data["FSI"]),
    ("Production", compact(latest["Production"]), "MT", pct(latest["Production"], prev["Production"]), data["Production"]),
    ("Price", f"&#36;{latest['Price']:,.0f}", "/ MT", pct(latest["Price"], prev["Price"], None), data["Price"]),
    ("Trade balance", money_m(latest["TradeBalance"]), "", delta(trade_change, money_m(abs(trade_change))),
     data["TradeBalance"]),
]
html('<div class="af-kpis">' + "".join(
    f'<div class="af-kpi"><div class="af-kpi-label">{label}</div>'
    f'<div class="af-kpi-value">{value}<small>{unit}</small></div>'
    f'<div class="af-delta-row"><span class="af-delta {tone}">{text}</span><span class="af-vs">vs {prev_year}</span></div>'
    f'{sparkline(series, i)}</div>'
    for i, (label, value, unit, (tone, text), series) in enumerate(kpis)) + "</div>")

# ─────────────────────────── OUTLOOK ───────────────────────────
if p_up >= 0.6:
    fsi_text = "Likely to improve"
elif p_up <= 0.4:
    fsi_text = "Likely to decline"
else:
    fsi_text = "Too close to call"
method = (f"RandomForest trained on <b>{n_outcomes}</b> year-to-year changes" if used_model
          else f"History shows one outcome only; smoothed share of past rises ({n_outcomes} changes)")
trend_word = "rising" if slope > 0 else "falling" if slope < 0 else "flat"


def projection_panel(year, mid, lo, hi):
    now = latest["Yield"]
    tone = "good" if mid > now else "bad" if mid < now else "neutral"
    arrow = {"good": "up", "bad": "down"}.get(tone, "flat")
    change = (mid - now) / now * 100 if now else 0
    d_lo, d_hi = min(lo, now), max(hi, now)
    pad = (d_hi - d_lo) * 0.08 or 1
    d_lo, d_hi = d_lo - pad, d_hi + pad
    pos = lambda v: (v - d_lo) / (d_hi - d_lo) * 100
    return f"""
    <div class="af-panel">
        <div class="af-panel-label">Yield projection<span class="af-tag">{year}</span></div>
        <div class="af-panel-main"><span class="ic {tone}">{icon(arrow)}</span>
            <div class="af-outlook-value">{mid:.2f}<small>MT/ha · {change:+.1f}%</small></div></div>
        <div class="af-range">
            <div class="af-range-track">
                <span class="af-range-band" style="left:{pos(lo):.1f}%;width:{pos(hi) - pos(lo):.1f}%"></span>
                <i class="af-range-now" style="left:{pos(now):.1f}%"></i>
                <i class="af-range-mid" style="left:{pos(mid):.1f}%"></i>
            </div>
            <div class="af-range-legend"><span>range <b>{lo:.2f}–{hi:.2f}</b></span><span>now <b>{now:.2f}</b></span></div>
        </div>
        <div class="af-panel-foot">Linear trend, {trend_word} <b>{slope:+.3f}</b> MT/ha per year.</div>
    </div>"""


(y1, p1, lo1, hi1), (y3, p3, lo3, hi3) = projections
html(f"""
<div class="af-section"><b>Outlook</b><span>forecasts from the {start_year}–{year_val} records</span></div>
<div class="af-outlook">
    <div class="af-panel">
        <div class="af-panel-label">Food security index<span class="af-tag">Next record</span></div>
        <div class="af-panel-main">{ring(p_up)}
            <div><div class="af-outlook-value">{fsi_text}</div>
            <div class="af-panel-foot" style="margin-top:4px">chance the index rises from <b>{latest['FSI']:.0f}</b></div></div>
        </div>
        <div class="af-panel-foot">{method}.</div>
    </div>
    {projection_panel(y1, p1, lo1, hi1)}
    {projection_panel(y3, p3, lo3, hi3)}
</div>
""")

# ─────────────────────────── ALERTS & INSIGHTS ───────────────────────────
alerts = []
if drought.lower() in ("critical", "extreme", "high"):
    alerts.append(("bad", f"{drought} drought risk", "High",
                   "Prioritise drought-tolerant varieties and water-saving irrigation."))
elif drought.lower() in ("moderate", "medium"):
    alerts.append(("warn", "Moderate drought risk", "Medium", "Monitor soil moisture and plan contingency irrigation."))

fsi_drop = data["FSI"].iloc[-1] - data["FSI"].iloc[-3 if len(data) >= 3 else -2]
if fsi_drop < -5:
    alerts.append(("bad", "Food security index falling", "High",
                   f"Down <b>{abs(fsi_drop):.0f} pts</b> over the last records. Policy support may be needed."))
elif fsi_drop < 0:
    alerts.append(("warn", "Food security index under pressure", "Medium",
                   f"Down <b>{abs(fsi_drop):.0f} pts</b> over the last records. Watch supply closely."))
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
    alerts.append(("good", "All indicators normal", "Info", "No critical risk factors detected."))

yield_vs_avg = (latest["Yield"] - avg_yield) / avg_yield * 100 if avg_yield else 0
best, worst = data.loc[data["Yield"].idxmax()], data.loc[data["Yield"].idxmin()]
rain_r = data["Rainfall"].corr(data["Yield"]) if data["Rainfall"].nunique() > 1 else float("nan")

# ─────────────────────────── TABS ───────────────────────────
st.write("")
tabs = st.tabs(TABS, key="tab", on_change="rerun", default=st.session_state.start_tab)
open_tab = next((name for name, tab in zip(TABS, tabs) if tab.open), TABS[0])
tab = dict(zip(TABS, tabs))

if tab["Overview"].open:
    with tab["Overview"]:
        left, right = st.columns([2, 1], gap="medium")
        with left:
            with card("yield", "Yield trajectory", "MT per hectare · linear trend with a ±1 s.e. projection fan"):
                charts.show(charts.yield_fan(data, slope, intercept, projections, t), "c-yield")
        with right:
            with card("alerts", "Risk alerts", f"{len(alerts)} active · latest record {year_val}", delay=0.08):
                html("".join(
                    f'<div class="af-alert"><span class="ic {tone}">{icon("check" if tone == "good" else "alert")}</span>'
                    f'<div><div class="af-alert-title">{title}<span class="af-sev {tone}">{sev}</span></div>'
                    f'<div class="af-alert-body">{body}</div></div></div>'
                    for tone, title, sev, body in alerts))
        left, right = st.columns([2, 1], gap="medium")
        with left:
            with card("fsi", "Food security index", "Score out of 100 · shaded by status band", delay=0.12):
                charts.show(charts.fsi_bands(data, t), "c-fsi")
        with right:
            with card("insights", "Insights", f"{crop} in {country}, {start_year}–{year_val}", delay=0.18):
                html(rows_html([
                    ("Latest yield vs average", f"{abs(yield_vs_avg):.1f}% {'above' if yield_vs_avg >= 0 else 'below'}"),
                    ("Best year", f"{int(best['Year'])} <small>{best['Yield']:.2f} MT/ha</small>"),
                    ("Weakest year", f"{int(worst['Year'])} <small>{worst['Yield']:.2f} MT/ha</small>"),
                    ("Rainfall–yield link", correlation_label(rain_r)
                     + (f" <small>r = {rain_r:.2f}</small>" if np.isfinite(rain_r) else "")),
                    ("Average profit margin", f"{profit_margin:.1f}%"),
                    ("Records analysed", f"{len(data)} <small>{start_year}–{year_val}</small>"),
                ]))
        meters = [
            ("Mechanization", latest["Mechanization"], "%", "Share of farm work mechanized", latest["Mechanization"]),
            ("Irrigation", latest["Irrigation"], "%", "Share of area irrigated", latest["Irrigation"]),
            ("Soil health", latest["SoilHealth"], "/100", "Composite soil score", latest["SoilHealth"]),
            ("Post-harvest loss", latest["LossRate"], "%", "Lower is better", min(latest["LossRate"] * 2.5, 100)),
        ]
        with card("ops", "Farm operations", f"Latest record, {year_val}", delay=0.22):
            html('<div class="af-meters">' + "".join(
                f'<div class="af-meter"><div class="af-meter-top"><span class="af-meter-label">{label}</span>'
                f'<span class="af-meter-value">{value:.0f}<small>{unit}</small></span></div>'
                f'<div class="af-bar"><span style="width:{width:.0f}%"></span></div>'
                f'<div class="af-meter-note">{note}</div></div>'
                for label, value, unit, note, width in meters) + "</div>")

if tab["Climate"].open:
    with tab["Climate"]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("d-rain", "Rainfall anomaly", f"mm above or below the {start_year}–{year_val} average"):
                charts.show(charts.rainfall_anomaly(data, t), "c-rain")
        with c2:
            with card("temp", "Average temperature", "°C, with the period average", delay=0.08):
                charts.show(charts.temperature(data, t), "c-temp")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("scatter", "Rainfall vs yield",
                      "Each point is one year" + (f" · r = {rain_r:.2f}" if np.isfinite(rain_r) else ""), delay=0.12):
                charts.show(charts.rain_vs_yield(data, rain_r, t), "c-rainyield")
        with c2:
            with card("corr", f"What moves {crop.lower()} numbers", "Correlation across every country and year",
                      delay=0.18):
                charts.show(charts.correlation_matrix(crop_rows(crop), t), f"c-corr-{mode}")
        profile = [("Drought risk", drought)]
        for col, label in [("climate_zone", "Climate zone"), ("climate_vulnerability", "Vulnerability"),
                           ("flood_risk_score", "Flood risk"), ("drought_freq_per10yr", "Droughts / decade")]:
            if col in latest.index:
                profile.append((label, latest[col]))
        profile.append(("Avg rainfall", f"{data['Rainfall'].mean():,.0f} mm"))
        with card("profile", "Climate profile", f"Latest record, {year_val}", delay=0.22):
            html('<div class="af-stats">' + "".join(
                f'<div class="af-stat"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in profile)
                 + "</div>")

if tab["Economics"].open:
    with tab["Economics"]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("prod", "Production", "Metric tonnes per year"):
                charts.show(charts.style(go.Figure(charts.bars(years, data["Production"], "Production", t["c_a"], "MT")),
                                         t, years=years), "c-prod")
        with c2:
            with card("price", "Price", "USD per metric tonne", delay=0.08):
                charts.show(charts.style(go.Figure(charts.area(years, data["Price"], "Price", t["c_a"], t, "USD/MT")),
                                         t, years=years), "c-price")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("d-margin", "Profit margin", "Percent of revenue", delay=0.12):
                fig = charts.signed_bars(years, data["ProfitMargin"], t, "Profit", "Loss", "%", ".1f")
                charts.show(charts.style(fig, t, legend=True, years=years, zero_line=True), "c-margin")
        with c2:
            with card("d-trade", "Trade balance", "Exports minus imports, USD millions", delay=0.18):
                fig = charts.signed_bars(years, data["TradeBalance"], t, "Surplus", "Deficit", "USD M")
                charts.show(charts.style(fig, t, legend=True, years=years, zero_line=True), "c-trade")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("d-waterfall", "Trade flows", f"Exports, imports and the net balance in {year_val}", delay=0.22):
                charts.show(charts.trade_waterfall(latest, t), "c-waterfall")
        with c2:
            with card("sankey", "Where the harvest goes", f"Production losses along the chain, {year_val}", delay=0.26):
                charts.show(charts.harvest_sankey(latest, t), "c-sankey")

if tab["Benchmark"].open:
    with tab["Benchmark"]:
        stats = crop_stats(crop)
        if st.session_state.get("_metric") not in METRICS:
            st.session_state._metric = "Yield"
        st.session_state.bench_metric = st.session_state._metric
        metric = st.segmented_control(
            "Compare countries by", list(METRICS), key="bench_metric", required=True,
            on_change=lambda: st.session_state.update(_metric=st.session_state.bench_metric))
        col, label, fmt = METRICS[metric]
        order = stats.sort_values(col, ascending=False)["Country"].tolist()
        rank = order.index(country) + 1 if country in order else None

        left, right = st.columns([2, 1], gap="medium")
        with left:
            with card("map", f"{crop} around the world",
                      f"{label} · country averages, all years · {len(stats)} countries"):
                charts.show(charts.world_map(stats, col, label, fmt, country, t), "c-map")
        with right:
            with card("h-rank", f"Top countries by {metric.lower()}",
                      f"{country} ranks #{rank} of {len(stats)}" if rank else "", delay=0.08):
                charts.show(charts.ranking(stats, col, fmt, country, t), "c-rank")

        peer_key = f"c-peer-{crop}-{country}"

        def jump_to_country():
            """Clicking a bubble opens that country for the same crop."""
            points = (st.session_state.get(peer_key) or {}).get("selection", {}).get("points", [])
            if not points:
                return
            custom = points[0].get("customdata")
            target = custom[0] if custom else None
            if target is None:
                groups = [g for _, g in stats.groupby(stats["Country"] == country)]
                idx = points[0].get("point_index", points[0].get("point_number"))
                curve = points[0].get("curve_number", 0)
                if idx is not None and curve < len(groups):
                    target = groups[curve]["Country"].iloc[idx]
            if target in catalog and crop in catalog[target]:
                st.session_state.country = target

        left, right = st.columns([3, 2], gap="medium")
        with left:
            with card("peer", "Peer landscape", "Bubble size is production · click a bubble to open that country",
                      delay=0.12):
                charts.show(charts.peer_scatter(stats, country, t), peer_key, on_select=jump_to_country,
                            selection_mode="points")
        with right:
            with card("radar", f"{country} vs the world", f"Percentile rank among countries growing {crop.lower()}",
                      delay=0.18):
                charts.show(charts.radar([(country, percentiles(stats, country), t["c_a"])], t), "c-radar")
        with card("gap", f"{crop}, year by year", "Rainfall vs yield for every country · press play", delay=0.22):
            charts.show(charts.gapminder(crop_rows(crop), country, t), f"c-gap-{crop}-{country}-{mode}")

if tab["Portfolio"].open:
    with tab["Portfolio"]:
        crow = country_rows(country)
        left, right = st.columns([2, 1], gap="medium")
        with left:
            with card("heat", f"{country}'s crop portfolio", "Yield each year as % of that crop's own average"):
                charts.show(charts.portfolio_heatmap(crow, crop, t), f"c-heat-{mode}")
        with right:
            with card("d-trends", "Yield momentum", "Linear trend in yield, % per year", delay=0.08):
                charts.show(charts.crop_trends(crow, crops, crop, t), "c-trends")
        with card("tree", "Production mix", "Tile size is average production · colour is average profit margin",
                  delay=0.14):
            charts.show(charts.portfolio_treemap(crow, t), f"c-tree-{mode}")

if tab["Compare"].open:
    with tab["Compare"]:
        others = [c for c in crops if c != crop]
        if not others:
            callout(f"{country} has no other crop with at least {MIN_RECORDS} records to compare with.")
        else:
            if st.session_state.get("_compare") not in others:
                st.session_state._compare = others[0]
            st.session_state.compare_crop = st.session_state._compare
            pick = st.selectbox(f"Compare {crop.lower()} with", others, key="compare_crop",
                                on_change=lambda: st.session_state.update(_compare=st.session_state.compare_crop))
            data_b = pair_records(country, pick, start_year)
            data_b = data_b[data_b["Year"] <= year_val]
            if len(data_b) < 2:
                callout(f"{pick} has fewer than 2 records in {start_year}–{year_val}. Try a longer period.")
            else:
                left, right = st.columns([3, 2], gap="medium")
                with left:
                    with card("cmp", f"Yield: {crop} vs {pick}", "MT per hectare"):
                        fig = go.Figure([charts.line(years, data["Yield"], crop, t["c_a"], t, "MT/ha"),
                                         charts.line(data_b["Year"], data_b["Yield"], pick, t["c_b"], t, "MT/ha",
                                                     symbol="square")])
                        span = np.array([min(years.min(), data_b["Year"].min()), year_val])
                        charts.show(charts.style(fig, t, height=330, legend=True, years=span), "c-cmp")
                with right:
                    with card("cmp-radar", "Percentile profile", "Each crop against countries growing that crop",
                              delay=0.08):
                        charts.show(charts.radar([(crop, percentiles(crop_stats(crop), country), t["c_a"]),
                                                  (pick, percentiles(crop_stats(pick), country), t["c_b"])], t),
                                    "c-cmp-radar")
                lat_b = data_b.iloc[-1]
                slope_b = np.polyfit(data_b["Year"], data_b["Yield"], 1)[0]
                rows = [
                    ("Latest record", f"{year_val}", f"{int(lat_b['Year'])}"),
                    ("Yield, latest (MT/ha)", f"{latest['Yield']:.2f}", f"{lat_b['Yield']:.2f}"),
                    ("Yield, period average", f"{avg_yield:.2f}", f"{data_b['Yield'].mean():.2f}"),
                    ("Yield trend (MT/ha per year)", f"{slope:+.3f}", f"{slope_b:+.3f}"),
                    ("Food security index, latest", f"{latest['FSI']:.0f}", f"{lat_b['FSI']:.0f}"),
                    ("Profit margin, average", f"{profit_margin:.1f}%", f"{data_b['ProfitMargin'].mean():.1f}%"),
                    ("Post-harvest loss, average", f"{loss_rate:.1f}%", f"{data_b['LossRate'].mean():.1f}%"),
                ]
                with card("cmp-table", "Side by side", f"{start_year}–{year_val}", delay=0.14):
                    html(f"""
                    <div class="af-table-wrap"><table class="af-table">
                        <thead><tr><th>Metric</th>
                        <th><span class="af-swatch" style="background:{t['c_a']}"></span>{crop}</th>
                        <th><span class="af-swatch" style="background:{t['c_b']}"></span>{pick}</th></tr></thead>
                        <tbody>{"".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)}</tbody>
                    </table></div>
                    """)

if tab["Data"].open:
    with tab["Data"]:
        columns = [("Year", "Year", "{:.0f}"), ("Yield", "Yield MT/ha", "{:.2f}"), ("Production", "Production MT", "{:,.0f}"),
                   ("FSI", "FSI", "{:.0f}"), ("FSStatus", "Status", "{}"), ("DroughtRisk", "Drought", "{}"),
                   ("Rainfall", "Rain mm", "{:,.0f}"), ("Temperature", "Temp °C", "{:.1f}"),
                   ("Price", "Price USD/MT", "{:,.0f}"), ("TradeBalance", "Trade USD M", "{:,.0f}"),
                   ("ProfitMargin", "Margin %", "{:.1f}"), ("LossRate", "Loss %", "{:.1f}"),
                   ("Irrigation", "Irrig. %", "{:.0f}"), ("Mechanization", "Mech. %", "{:.0f}"),
                   ("SoilHealth", "Soil", "{:.0f}")]
        body = "".join("<tr>" + "".join(f"<td>{fmt.format(row[c])}</td>" for c, _, fmt in columns) + "</tr>"
                       for _, row in data.iloc[::-1].iterrows())
        with card("data", f"{len(data)} records", f"{crop} in {country}, newest first"):
            html(f'<div class="af-table-wrap"><table class="af-table"><thead><tr>'
                 + "".join(f"<th>{name}</th>" for _, name, _ in columns)
                 + f"</tr></thead><tbody>{body}</tbody></table></div>")
            table = data[[c for c, _, _ in columns]]
            st.download_button("Download CSV", table.to_csv(index=False), mime="text/csv", icon=":material/download:",
                               file_name=f"agriflow_{country}_{crop}_{start_year}-{year_val}.csv".replace(" ", "_"))

# ─────────────────────────── FOOTER ───────────────────────────
html(f"""
<div class="af-footer">
    <span>AgriFlow AI · crop &amp; food security analytics</span>
    <span>RandomForest outlook · linear yield trend · {datetime.date.today().strftime("%d %b %Y")}</span>
</div>
""")

params = {"country": country, "crop": crop, "theme": mode, "tab": open_tab}
if period != "All years":
    params["period"] = period
st.query_params.from_dict(params)
