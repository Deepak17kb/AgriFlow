import datetime

import numpy as np
import plotly.graph_objects as go
import streamlit as st

import charts
from model import predict, prepare_features, train_model
from ui import THEMES, callout, card, html, icon, sparkline, theme_css
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
THEME_ICONS = {"light": ":material/light_mode:", "dark": ":material/dark_mode:"}


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
    """All records for a crop, limited to countries where it can be analysed."""
    eligible = [c for c, crops in get_catalog().items() if crop in crops]
    df = get_data()
    return df[(df["Crop"] == crop) & df["Country"].isin(eligible)]


@st.cache_data(show_spinner=False)
def country_rows(country):
    df = get_data()
    return df[df["Country"] == country]


@st.cache_data(show_spinner=False)
def crop_stats(crop):
    """Per-country averages for one crop."""
    stats = (crop_rows(crop).groupby("Country")
             .agg(Yield=("Yield", "mean"), YieldSD=("Yield", "std"), FSI=("FSI", "mean"),
                  ProfitMargin=("ProfitMargin", "mean"), Production=("Production", "mean"),
                  LossRate=("LossRate", "mean"), Irrigation=("Irrigation", "mean"),
                  Mechanization=("Mechanization", "mean"), SoilHealth=("SoilHealth", "mean"),
                  Region=("region", "first"))
             .reset_index())
    stats["YieldCV"] = stats["YieldSD"] / stats["Yield"] * 100
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


def sentence(text):
    text = str(text)
    return text[:1].upper() + text[1:].lower()


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

# First load: restore the view from the URL (?country=India&crop=Wheat&theme=dark&tab=Benchmark).
if "booted" not in st.session_state:
    qp = st.query_params
    st.session_state.booted = True
    st.session_state.theme = qp.get("theme") if qp.get("theme") in THEMES else "light"
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
        <div class="af-mark">{icon("leaf", 17)}</div>
        <div><div class="af-brand-name">AgriFlow AI</div>
        <div class="af-brand-sub">Crop and food security analytics</div></div>
    </div>
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
    <div class="af-side-title" style="margin-top:14px">Dataset</div>
    {rows_html([("Countries", df["Country"].nunique()), ("Crops", df["Crop"].nunique()),
                ("Years", f'{int(df["Year"].min())}–{int(df["Year"].max())}'), ("Records", f"{len(df):,}")])}
    <div class="af-hint">Views update as soon as a filter changes. The address bar holds a shareable link
    to the current view.</div>
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
head_l, head_r = st.columns([6, 1], vertical_alignment="top")
with head_r:
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.segmented_control("Theme", list(THEME_ICONS), key="theme", required=True,
                             format_func=THEME_ICONS.get, label_visibility="collapsed")
with head_l:
    html(f"""
    <div class="af-eyebrow">{region if isinstance(region, str) else "Crop analytics"}</div>
    <div class="af-title">{crop} <span>in</span> {country}</div>
    <div class="af-meta">
        <span>{start_year}–{year_val} · {len(data)} annual records</span>
        <span><i class="af-dot {tone_for(fs_status, FS_TONES)}"></i>{sentence(fs_status)}</span>
        <span><i class="af-dot {tone_for(drought, DROUGHT_TONES)}"></i>{sentence(drought)} drought risk</span>
    </div>
    """)


# ─────────────────────────── KPI TILES ───────────────────────────
def delta(change, text, good_when="up"):
    if not np.isfinite(change) or change == 0:
        return "neutral", "No change"
    tone = "neutral" if good_when is None else ("good" if (change > 0) == (good_when == "up") else "bad")
    return tone, f"{'▲' if change > 0 else '▼'} {text}"


def pct(cur, old, good_when="up"):
    change = (cur - old) / abs(old) * 100 if old else float("nan")
    return delta(change, f"{abs(change):.1f}%", good_when)


fsi_change = latest["FSI"] - prev["FSI"]
trade_change = latest["TradeBalance"] - prev["TradeBalance"]
kpis = [
    ("Yield", f"{latest['Yield']:.2f}", "MT/ha", pct(latest["Yield"], prev["Yield"]), data["Yield"]),
    ("Food security index", f"{latest['FSI']:.0f}", "/ 100", delta(fsi_change, f"{abs(fsi_change):.0f} pts"), data["FSI"]),
    ("Production", compact(latest["Production"]), "MT", pct(latest["Production"], prev["Production"]), data["Production"]),
    ("Price", f"&#36;{latest['Price']:,.0f}", "/ MT", pct(latest["Price"], prev["Price"], None), data["Price"]),
    ("Trade balance", money_m(latest["TradeBalance"]), "", delta(trade_change, money_m(abs(trade_change))),
     data["TradeBalance"]),
]
html('<div class="af-kpis">' + "".join(
    f'<div class="af-kpi"><div class="af-kpi-label">{label}</div>'
    f'<div class="af-kpi-value">{value}<small>{unit}</small></div>'
    f'<div class="af-delta {tone}-text">{text}<span>vs {prev_year}</span></div>'
    f'{sparkline(series)}</div>'
    for label, value, unit, (tone, text), series in kpis) + "</div>")

# ─────────────────────────── OUTLOOK ───────────────────────────
if p_up >= 0.6:
    fsi_text = "Likely to improve"
elif p_up <= 0.4:
    fsi_text = "Likely to decline"
else:
    fsi_text = "Too close to call"
method = (f"RandomForest trained on <b>{n_outcomes}</b> year-to-year changes" if used_model
          else f"History shows one outcome only, so this is the smoothed share of past rises ({n_outcomes} changes)")
trend_word = "rising" if slope > 0 else "falling" if slope < 0 else "flat"


def projection_panel(year, mid, lo, hi):
    now = latest["Yield"]
    change = (mid - now) / now * 100 if now else 0
    tone = "good" if change > 0 else "bad" if change < 0 else "neutral"
    d_lo, d_hi = min(lo, now), max(hi, now)
    pad = (d_hi - d_lo) * 0.08 or 1
    d_lo, d_hi = d_lo - pad, d_hi + pad
    pos = lambda v: (v - d_lo) / (d_hi - d_lo) * 100
    return f"""
    <div class="af-panel">
        <div class="af-panel-label">Yield projection<span class="af-tag">{year}</span></div>
        <div class="af-outlook-value">{mid:.2f} MT/ha<small class="{tone}-text">{change:+.1f}% vs now</small></div>
        <div class="af-track">
            <span class="band" style="left:{pos(lo):.1f}%;width:{pos(hi) - pos(lo):.1f}%"></span>
            <i class="dot now" style="left:{pos(now):.1f}%"></i>
            <i class="dot proj" style="left:{pos(mid):.1f}%"></i>
        </div>
        <div class="af-scale"><span>Range <b>{lo:.2f}–{hi:.2f}</b></span><span>Now <b>{now:.2f}</b></span></div>
        <div class="af-panel-foot">Linear trend, {trend_word} {slope:+.3f} MT/ha per year.</div>
    </div>"""


(y1, p1, lo1, hi1), (y3, p3, lo3, hi3) = projections
html(f"""
<div class="af-section"><b>Outlook</b><span>Forecasts from the {start_year}–{year_val} records</span></div>
<div class="af-outlook">
    <div class="af-panel">
        <div class="af-panel-label">Food security index<span class="af-tag">Next record</span></div>
        <div class="af-outlook-value">{fsi_text}</div>
        <div class="af-track"><span class="fill" style="width:{p_up * 100:.0f}%"></span><i class="mid" style="left:50%"></i></div>
        <div class="af-scale"><span>Decline</span><span><b>{p_up * 100:.0f}%</b> chance of a rise from {latest['FSI']:.0f}</span><span>Rise</span></div>
        <div class="af-panel-foot">{method}.</div>
    </div>
    {projection_panel(y1, p1, lo1, hi1)}
    {projection_panel(y3, p3, lo3, hi3)}
</div>
""")

# ─────────────────────────── ALERTS & INSIGHTS ───────────────────────────
alerts = []
if drought.lower() in ("critical", "extreme", "high"):
    alerts.append(("bad", f"{sentence(drought)} drought risk", "High",
                   "Prioritise drought-tolerant varieties and water-saving irrigation."))
elif drought.lower() in ("moderate", "medium"):
    alerts.append(("warn", "Moderate drought risk", "Medium", "Monitor soil moisture and plan contingency irrigation."))

fsi_drop = data["FSI"].iloc[-1] - data["FSI"].iloc[-3 if len(data) >= 3 else -2]
if fsi_drop < -5:
    alerts.append(("bad", "Food security index falling", "High",
                   f"Down {abs(fsi_drop):.0f} pts over the last records. Policy support may be needed."))
elif fsi_drop < 0:
    alerts.append(("warn", "Food security index under pressure", "Medium",
                   f"Down {abs(fsi_drop):.0f} pts over the last records. Watch supply closely."))
if loss_rate > 15:
    alerts.append(("bad", "High post-harvest loss", "High",
                   f"Average loss of {loss_rate:.1f}%. Invest in cold chain and storage."))
elif loss_rate > 8:
    alerts.append(("warn", "Elevated post-harvest loss", "Medium",
                   f"Average loss of {loss_rate:.1f}%. Review storage and transport."))
if p_up >= 0.65:
    alerts.append(("good", "Positive food security outlook", "Info",
                   f"{p_up * 100:.0f}% chance the index rises by the next record."))
elif p_up <= 0.35:
    alerts.append(("warn", "Negative food security outlook", "Medium",
                   f"Only {p_up * 100:.0f}% chance the index rises by the next record."))
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
            with card("yield", "Yield trajectory", "MT per hectare, with linear trend and a ±1 s.e. projection range"):
                charts.show(charts.yield_fan(data, slope, intercept, projections, t), "c-yield")
        with right:
            with card("alerts", "Risk alerts", f"{len(alerts)} active for the latest record, {year_val}"):
                html("".join(
                    f'<div class="af-alert {tone}">{icon("check" if tone == "good" else "alert")}'
                    f'<div><div class="af-alert-title">{title}<span>{sev}</span></div>'
                    f'<div class="af-alert-body">{body}</div></div></div>'
                    for tone, title, sev, body in alerts))
        left, right = st.columns([2, 1], gap="medium")
        with left:
            with card("fsi", "Food security index", "Score out of 100, shaded by status band"):
                charts.show(charts.fsi_bands(data, t), "c-fsi")
        with right:
            with card("z", f"How {year_val} compares", f"Latest record against the {start_year}–{year_val} average"):
                charts.show(charts.latest_vs_history(data, t), "c-z")
        left, right = st.columns([2, 1], gap="medium")
        with left:
            with card("bullets", "Farm operations benchmark",
                      f"Bar: {country}'s average · tick: median country · shading: middle 50% and full range "
                      f"of countries growing {crop.lower()}"):
                charts.show(charts.bullets(crop_stats(crop), country, t), "c-bullets")
        with right:
            with card("insights", "Insights", f"{crop} in {country}, {start_year}–{year_val}"):
                html(rows_html([
                    ("Latest yield vs average", f"{abs(yield_vs_avg):.1f}% {'above' if yield_vs_avg >= 0 else 'below'}"),
                    ("Best year", f"{int(best['Year'])}<small>{best['Yield']:.2f} MT/ha</small>"),
                    ("Weakest year", f"{int(worst['Year'])}<small>{worst['Yield']:.2f} MT/ha</small>"),
                    ("Rainfall–yield link", correlation_label(rain_r)
                     + (f"<small>r = {rain_r:.2f}</small>" if np.isfinite(rain_r) else "")),
                    ("Average profit margin", f"{profit_margin:.1f}%"),
                    ("Average post-harvest loss", f"{loss_rate:.1f}%"),
                    ("Records analysed", f"{len(data)}<small>{start_year}–{year_val}</small>"),
                ]))

if tab["Climate"].open:
    with tab["Climate"]:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("rain", "Rainfall anomaly", f"mm above or below the {start_year}–{year_val} average"):
                charts.show(charts.rainfall_anomaly(data, t), "c-rain")
        with c2:
            with card("temp", "Average temperature", "°C, with the period average"):
                charts.show(charts.temperature(data, t), "c-temp")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("scatter", "Rainfall vs yield",
                      "Each point is one year" + (f" · r = {rain_r:.2f}" if np.isfinite(rain_r) else "")):
                charts.show(charts.rain_vs_yield(data, rain_r, t), "c-rainyield")
        with c2:
            with card("corr", "Driver correlations", f"Pearson r across every country and year for {crop.lower()}"):
                charts.show(charts.correlation_matrix(crop_rows(crop), t), f"c-corr-{mode}")
        profile = [("Drought risk", sentence(drought))]
        for col, label in [("climate_zone", "Climate zone"), ("climate_vulnerability", "Vulnerability score"),
                           ("flood_risk_score", "Flood risk score"), ("drought_freq_per10yr", "Droughts per decade")]:
            if col in latest.index:
                profile.append((label, latest[col]))
        profile.append(("Average rainfall", f"{data['Rainfall'].mean():,.0f} mm"))
        with card("profile", "Climate profile", f"Latest record, {year_val}"):
            html('<div class="af-stats">' + "".join(
                f'<div class="af-stat"><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in profile)
                 + "</div>")

if tab["Economics"].open:
    with tab["Economics"]:
        with card("indexed", "Indexed trends", f"Each series relative to its {start_year} value"):
            charts.show(charts.indexed_trends(data, t), "c-indexed")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("prod", "Production", "Metric tonnes per year"):
                charts.show(charts.style(go.Figure(charts.bars(years, data["Production"], "Production", t["c_a"], "MT")),
                                         t, years=years), "c-prod")
        with c2:
            with card("price", "Price", "USD per metric tonne"):
                charts.show(charts.style(go.Figure(charts.area(years, data["Price"], "Price", t["c_a"], t, "USD/MT")),
                                         t, years=years), "c-price")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("margin", "Profit margin", "Percent of revenue"):
                fig = charts.signed_bars(years, data["ProfitMargin"], t, "Profit", "Loss", "%", ".1f")
                charts.show(charts.style(fig, t, legend=True, years=years, zero_line=True), "c-margin")
        with c2:
            with card("trade", "Trade balance", "Exports minus imports, USD millions"):
                fig = charts.signed_bars(years, data["TradeBalance"], t, "Surplus", "Deficit", "USD M")
                charts.show(charts.style(fig, t, legend=True, years=years, zero_line=True), "c-trade")
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with card("waterfall", "Trade flows", f"Exports, imports and the net balance in {year_val}"):
                charts.show(charts.trade_waterfall(latest, t), "c-waterfall")
        with c2:
            with card("sankey", "Where the harvest goes", f"Losses along the supply chain, {year_val}"):
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
            with card("map", f"{crop} by country", f"{label}, country averages over all years · {len(stats)} countries"):
                charts.show(charts.world_map(stats, col, label, fmt, country, t), "c-map")
        with right:
            with card("rank", f"Top countries by {metric.lower()}",
                      f"{country} ranks {rank} of {len(stats)}" if rank else ""):
                charts.show(charts.ranking(stats, col, fmt, country, t), "c-rank")

        with card("dist", f"{country} within the global distribution",
                  f"{crop} yield each year against the spread of all {len(stats)} countries"):
            charts.show(charts.distribution_bands(crop_rows(crop), country, t), "c-dist")

        peer_key = f"c-peer-{crop}-{country}"

        def jump_to_country():
            """Clicking a bubble opens that country for the same crop."""
            points = (st.session_state.get(peer_key) or {}).get("selection", {}).get("points", [])
            if not points:
                return
            custom = points[0].get("customdata")
            target = custom[0] if custom else None
            if target is None:
                groups = charts._two_groups(stats, country)
                idx = points[0].get("point_index", points[0].get("point_number"))
                curve = points[0].get("curve_number", 0)
                if idx is not None and curve < len(groups):
                    target = groups[curve]["Country"].iloc[idx]
            if target in catalog and crop in catalog[target]:
                st.session_state.country = target

        left, right = st.columns(2, gap="medium")
        with left:
            with card("peer", "Yield and food security", "Bubble size is production · select a bubble to open that country"):
                charts.show(charts.peer_scatter(stats, country, t), peer_key, on_select=jump_to_country,
                            selection_mode="points")
        with right:
            with card("stability", "Yield and stability", "Average yield against year-to-year volatility"):
                charts.show(charts.stability(stats, country, t), "c-stability")
        with card("pct", "Percentile profile", f"Where {country} ranks among countries growing {crop.lower()}"):
            charts.show(charts.percentile_dots([(country, percentiles(stats, country), t["c_a"], "circle")], t),
                        "c-pct")

if tab["Portfolio"].open:
    with tab["Portfolio"]:
        crow = country_rows(country)
        left, right = st.columns([2, 1], gap="medium")
        with left:
            with card("heat", f"{country} crop portfolio", "Yield each year as a percentage of that crop's average"):
                charts.show(charts.portfolio_heatmap(crow, crop, t), f"c-heat-{mode}")
        with right:
            with card("trends", "Yield momentum", "Linear trend in yield, percent per year"):
                charts.show(charts.crop_trends(crow, crops, crop, t), "c-trends")
        left, right = st.columns(2, gap="medium")
        with left:
            with card("tree", "Production mix", "Tile size is average production · colour is average profit margin"):
                charts.show(charts.portfolio_treemap(crow, t), f"c-tree-{mode}")
        with right:
            with card("pareto", "Production concentration", "Share of average production by crop, with the cumulative total"):
                charts.show(charts.production_pareto(crow, crop, t), "c-pareto")

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
                    with card("cmp", f"Yield: {crop} and {pick}", "MT per hectare"):
                        fig = go.Figure([charts.line(years, data["Yield"], crop, t["c_a"], t, "MT/ha"),
                                         charts.line(data_b["Year"], data_b["Yield"], pick, t["c_b"], t, "MT/ha",
                                                     symbol="square")])
                        span = np.array([min(years.min(), data_b["Year"].min()), year_val])
                        charts.show(charts.style(fig, t, height=350, legend=True, years=span), "c-cmp")
                with right:
                    with card("cmp-pct", "Percentile profile", "Each crop against countries growing that crop"):
                        charts.show(charts.percentile_dots(
                            [(crop, percentiles(crop_stats(crop), country), t["c_a"], "circle"),
                             (pick, percentiles(crop_stats(pick), country), t["c_b"], "square")], t), "c-cmp-pct")
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
                with card("cmp-table", "Side by side", f"{start_year}–{year_val}"):
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
                   ("Irrigation", "Irrigation %", "{:.0f}"), ("Mechanization", "Mechanization %", "{:.0f}"),
                   ("SoilHealth", "Soil health", "{:.0f}")]
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
    <span>AgriFlow AI · Crop and food security analytics</span>
    <span>RandomForest outlook · linear yield trend · {datetime.date.today().strftime("%d %b %Y")}</span>
</div>
""")

params = {"country": country, "crop": crop, "theme": mode, "tab": open_tab}
if period != "All years":
    params["period"] = period
st.query_params.from_dict(params)
