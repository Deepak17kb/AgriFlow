"""Theme tokens, global CSS and small HTML helpers for the AgriFlow UI.

Streamlit's own theme stays pinned to light; the app draws both modes itself.
Every color below becomes a CSS variable, so switching theme only swaps the
variable block and the page fades between modes instead of reloading.
"""
import streamlit as st

SANS = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"

# Sky blue for the interface, field green for crop data. The categorical order
# (c_a, c_b, c_orange, c_violet) and the blue/red polarity pair pass the
# color-vision-deficiency check against each mode's surface.
THEMES = {
    "light": {
        "bg": "#f6f8fa", "surface": "#ffffff", "surface_2": "#f3f6f9", "field": "#ffffff",
        "border": "#e3e8ee", "border_strong": "#cfd7e0",
        "ink": "#0f1c2b", "ink_2": "#3e4c5d", "muted": "#66758a", "subtle": "#9aa6b5",
        "accent": "#0284c7", "accent_ink": "#0369a1", "accent_soft": "rgba(2,132,199,.08)",
        "shadow": "0 1px 2px rgba(15,28,43,.04)",
        "good": "#15803d", "warn": "#b45309", "bad": "#b91c1c",
        "c_a": "#2f8f5b", "c_b": "#2b7fd0", "c_orange": "#d9722b", "c_violet": "#6b5bd2", "c_red": "#d64545",
        "c_trend": "#9aa6b5", "c_peer": "#c9d2dc", "c_grid": "#eef2f6", "c_axis": "#cfd7e0", "c_tick": "#66758a",
        "c_mid": "#eef0f2", "c_land": "#edf1f5", "c_band1": "#e6ecf2", "c_band2": "#cdd7e2",
        "seq": ["#e8f4ec", "#b9dfc6", "#7fc29a", "#3f9b68", "#1f6b43"],
    },
    "dark": {
        "bg": "#0e1419", "surface": "#141b22", "surface_2": "#1a222b", "field": "#111820",
        "border": "#242e39", "border_strong": "#33404e",
        "ink": "#e4eaf0", "ink_2": "#b3bfcc", "muted": "#8795a5", "subtle": "#5f6d7c",
        "accent": "#4ea8de", "accent_ink": "#6cb6e6", "accent_soft": "rgba(78,168,222,.12)",
        "shadow": "none",
        "good": "#4caf7d", "warn": "#d9a441", "bad": "#e06c6c",
        "c_a": "#3f9e6c", "c_b": "#4f9bdc", "c_orange": "#d07a40", "c_violet": "#8f86dc", "c_red": "#e06666",
        "c_trend": "#5f6d7c", "c_peer": "#3a4654", "c_grid": "rgba(255,255,255,.06)", "c_axis": "#2c3743",
        "c_tick": "#8795a5", "c_mid": "#2a323c", "c_land": "#1a222b", "c_band1": "#212b35", "c_band2": "#34414f",
        "seq": ["#17271f", "#1f4a35", "#2f7350", "#4a9d6f", "#86c9a2"],
    },
}


def rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha})"


def html(markup):
    """Render trusted HTML. Lines are stripped so Markdown never reads indentation as a code block."""
    st.markdown(" ".join(line.strip() for line in markup.splitlines() if line.strip()),
                unsafe_allow_html=True)


_SVG = ('<svg viewBox="0 0 16 16" width="{s}" height="{s}" fill="none" stroke="currentColor" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">{p}</svg>')
_PATHS = {
    "leaf":  '<path d="M3.5 12.5c0-5.5 3.5-9 9-9 0 5.5-3.5 9-9 9z"/><path d="M3.5 12.5l5-5"/>',
    "alert": '<path d="M8 2.5l6 10.5H2L8 2.5z"/><path d="M8 6.5v3"/><path d="M8 11.5h.01"/>',
    "check": '<circle cx="8" cy="8" r="6"/><path d="M5.5 8.2l1.8 1.8 3.2-3.5"/>',
    "info":  '<circle cx="8" cy="8" r="6"/><path d="M8 7.5v3.5"/><path d="M8 5h.01"/>',
}


def icon(name, size=16):
    return _SVG.format(s=size, p=_PATHS[name])


def card(key, title, subtitle=None, info=None):
    """A bordered panel for one chart or list; `info` is a hover note on how the chart is built."""
    box = st.container(key=f"card-{key}")
    with box:
        sub = f'<div class="af-card-sub">{subtitle}</div>' if subtitle else ""
        tip = f'<span class="af-info" title="{info}">{icon("info", 15)}</span>' if info else ""
        html(f'<div class="af-card-head"><div><div class="af-card-title">{title}</div>{sub}</div>{tip}</div>')
    return box


def group(title, subtitle=""):
    """A light heading that groups the cards below it inside a tab."""
    html(f'<div class="af-group"><b>{title}</b><span>{subtitle}</span></div>')


def callout(text, kind="info"):
    html(f'<div class="af-callout">{icon(kind)}<div>{text}</div></div>')


def sparkline(values, w=240, h=32):
    """Inline SVG trend line with a faint area under it."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pts = [(i / (len(vals) - 1) * (w - 2) + 1, h - 3 - (v - lo) / span * (h - 6)) for i, v in enumerate(vals)]
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{line} L{pts[-1][0]:.1f},{h} L{pts[0][0]:.1f},{h} Z"
    return (f'<svg class="af-spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<path class="ar" d="{area}"/><path class="ln" d="{line}"/></svg>')


def theme_css(mode):
    t = THEMES[mode]
    tokens = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in t.items() if isinstance(v, str))
    return f"<style>:root{{{tokens};color-scheme:{mode}}}{_CSS}</style>"


_CSS = """
:root { --radius: 10px; }

/* ---------- App shell ---------- */
.stApp { background-color: var(--bg) !important; color: var(--ink); transition: background-color .3s ease, color .3s ease; }
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"],
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 2rem; padding-bottom: 3rem; max-width: 1440px; }
[data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-stale="true"] { opacity: .85 !important; transition: opacity .2s ease; }
[data-testid="stStatusWidget"], [data-testid="stStatusWidget"] * { color: var(--muted) !important; }
* { scrollbar-color: var(--border-strong) transparent; scrollbar-width: thin; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border);
  transition: background-color .3s ease, border-color .3s ease; }
[data-testid="stSidebar"] > div { background: transparent !important; }
[data-testid="stSidebarCollapseButton"] *, [data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] * { color: var(--muted) !important; }

/* ---------- Native widgets, themed through the same tokens ---------- */
[data-testid="stWidgetLabel"] p { font-size: 12.5px !important; font-weight: 500 !important; color: var(--ink-2) !important; }
[data-baseweb="select"] > div { background-color: var(--field) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; transition: border-color .15s ease, background-color .3s ease, box-shadow .15s ease; }
[data-baseweb="select"] > div:hover { border-color: var(--border-strong) !important; }
[data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft) !important; }
[data-baseweb="select"] div, [data-baseweb="select"] input, [data-baseweb="select"] span { color: var(--ink) !important; }
[data-baseweb="select"] svg { color: var(--muted) !important; }
[data-baseweb="popover"] > div { background-color: var(--surface) !important; border: 1px solid var(--border) !important;
  border-radius: 8px !important; box-shadow: 0 8px 24px rgba(15,28,43,.12) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"] { background-color: var(--surface) !important; }
[data-baseweb="popover"] div, [data-baseweb="popover"] ul { border-color: var(--border) !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"] { color: var(--ink) !important; background-color: transparent !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] { background-color: var(--accent-soft) !important; }
button[data-testid^="stBaseButton-segmented_control"] { background-color: var(--field) !important;
  border-color: var(--border) !important; color: var(--ink-2) !important; transition: background-color .15s ease, color .15s ease; }
button[data-testid^="stBaseButton-segmented_control"]:hover { color: var(--ink) !important; }
button[data-testid="stBaseButton-segmented_controlActive"] { background-color: var(--accent-soft) !important;
  border-color: var(--accent) !important; color: var(--accent-ink) !important; z-index: 1; }
button[data-testid^="stBaseButton-segmented_control"] p, button[data-testid^="stBaseButton-segmented_control"] span { color: inherit !important; }
[data-testid="stDownloadButton"] button { background-color: var(--surface) !important; color: var(--ink) !important;
  border: 1px solid var(--border) !important; border-radius: 8px !important; }
[data-testid="stDownloadButton"] button:hover { border-color: var(--accent) !important; color: var(--accent-ink) !important; }
[data-testid="stDownloadButton"] button * { color: inherit !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important; font-size: 12px; }

/* Tabs: standard underline */
.stTabs [data-baseweb="tab-list"] { gap: 26px; border-bottom: 1px solid var(--border); max-width: 100%; overflow-x: auto; }
.stTabs [data-baseweb="tab"] { height: 42px; padding: 0 1px; background: transparent; }
.stTabs [data-baseweb="tab"] p { font-size: 14px; font-weight: 500; color: var(--muted); transition: color .15s ease; }
.stTabs [data-baseweb="tab"]:hover p { color: var(--ink); }
.stTabs [aria-selected="true"] p { color: var(--ink) !important; font-weight: 600; }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--accent) !important; height: 2px; }
.stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 18px; }

/* ---------- Cards ---------- */
[class*="st-key-card"] { background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 16px 18px 10px; box-shadow: var(--shadow); animation: af-fade .35s ease both;
  transition: background-color .3s ease, border-color .3s ease; }
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> [class*="st-key-card"]) { flex: 1 1 auto; }
.af-card-head { display: flex; justify-content: space-between; align-items: flex-start; gap: 12px; }
.af-card-title { font-size: 14px; font-weight: 600; color: var(--ink); }
.af-card-sub { font-size: 12.5px; color: var(--muted); margin-top: 2px; line-height: 1.45; }
.af-info { flex: none; color: var(--subtle); cursor: help; line-height: 0; margin-top: 2px; transition: color .15s ease; }
.af-info:hover { color: var(--ink-2); }
.af-group { display: flex; align-items: baseline; gap: 10px; margin: 8px 0 -2px; }
.af-group b { font-size: 13px; font-weight: 600; color: var(--ink-2); white-space: nowrap; }
.af-group span { font-size: 12.5px; color: var(--muted); white-space: nowrap; }
.af-group::after { content: ""; flex: 1; height: 1px; background: var(--border); align-self: center; }

/* ---------- Header ---------- */
.af-eyebrow { font-size: 13px; font-weight: 500; color: var(--muted); }
.af-title { font-size: 28px; font-weight: 650; letter-spacing: -.02em; color: var(--ink); line-height: 1.2; margin-top: 2px; }
.af-title span { color: var(--subtle); font-weight: 500; }
.af-meta { display: flex; flex-wrap: wrap; align-items: center; row-gap: 6px; margin-top: 8px; font-size: 13px; color: var(--muted); }
.af-meta > span { display: inline-flex; align-items: center; gap: 7px; padding: 0 12px; border-left: 1px solid var(--border); }
.af-meta > span:first-child { padding-left: 0; border-left: none; }
.af-meta b { color: var(--ink-2); font-weight: 500; }
.af-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--subtle); }
.af-dot.good { background: var(--good); } .af-dot.warn { background: var(--warn); } .af-dot.bad { background: var(--bad); }

/* ---------- KPI tiles ---------- */
/* One joined strip: the 1px grid gap over a border-coloured background draws the dividers */
.af-kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 1px; margin-top: 6px; background: var(--border);
  border: 1px solid var(--border); border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow); animation: af-fade .35s ease both; }
.af-kpi { background-color: var(--surface); padding: 14px 18px 10px; transition: background-color .3s ease; }
.af-kpi-label { font-size: 12.5px; font-weight: 500; color: var(--muted); }
.af-kpi-value { font-size: 24px; font-weight: 600; letter-spacing: -.02em; color: var(--ink); margin-top: 6px; line-height: 1.2; white-space: nowrap; }
.af-kpi-value small { font-size: 12.5px; font-weight: 400; color: var(--muted); margin-left: 4px; letter-spacing: 0; }
.af-delta { font-size: 12.5px; font-weight: 500; margin-top: 4px; }
.af-delta span { color: var(--muted); font-weight: 400; margin-left: 4px; }
.good-text { color: var(--good); } .bad-text { color: var(--bad); } .neutral-text { color: var(--ink-2); }
.af-spark { display: block; width: 100%; height: 32px; margin-top: 8px; }
.af-spark .ln { fill: none; stroke: var(--accent); stroke-width: 1.5; stroke-linejoin: round; stroke-linecap: round; vector-effect: non-scaling-stroke; }
.af-spark .ar { fill: var(--accent); opacity: .07; }
@media (max-width: 1100px) { .af-kpis { grid-template-columns: repeat(6, minmax(0, 1fr)); }
  .af-kpi { grid-column: span 2; } .af-kpi:nth-child(n+4) { grid-column: span 3; } }
@media (max-width: 640px) { .af-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .af-kpi, .af-kpi:nth-child(n+4) { grid-column: span 1; } .af-kpi:last-child { grid-column: span 2; }
  .af-title { font-size: 24px; } [data-testid="stMainBlockContainer"] { padding-top: 3.2rem; } }

/* ---------- Outlook ---------- */
.af-section { display: flex; align-items: baseline; gap: 10px; margin: 18px 0 0; }
.af-section b { font-size: 15px; font-weight: 600; color: var(--ink); }
.af-section span { font-size: 12.5px; color: var(--muted); }
.af-outlook { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.af-panel { background-color: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 16px 18px;
  box-shadow: var(--shadow); animation: af-fade .35s ease both; transition: background-color .3s ease, border-color .3s ease; }
.af-panel-label { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-size: 12.5px; font-weight: 500; color: var(--muted); }
.af-tag { font-size: 11.5px; font-weight: 500; color: var(--ink-2); background: var(--surface-2); border: 1px solid var(--border);
  border-radius: 5px; padding: 1px 6px; }
.af-outlook-value { font-size: 20px; font-weight: 600; color: var(--ink); letter-spacing: -.01em; margin-top: 10px; }
.af-outlook-value small { font-size: 12.5px; font-weight: 500; margin-left: 6px; letter-spacing: 0; }
.af-track { position: relative; height: 6px; border-radius: 3px; background: var(--surface-2); margin-top: 14px; }
.af-track .fill { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 3px; background: var(--accent); }
.af-track .band { position: absolute; top: 0; bottom: 0; border-radius: 3px; background: var(--accent); opacity: .28; }
.af-track .mid { position: absolute; top: -3px; bottom: -3px; width: 1px; background: var(--border-strong); }
.af-track .dot { position: absolute; top: 50%; width: 10px; height: 10px; margin: -5px 0 0 -5px; border-radius: 50%; }
.af-track .dot.proj { background: var(--accent); }
.af-track .dot.now { background: var(--surface); border: 2px solid var(--ink-2); }
.af-scale { display: flex; justify-content: space-between; gap: 8px; margin-top: 7px; font-size: 11.5px; color: var(--muted); }
.af-scale b { color: var(--ink-2); font-weight: 600; }
.af-panel-foot { font-size: 12.5px; color: var(--muted); margin-top: 10px; line-height: 1.5; }
.af-panel-foot b { color: var(--ink-2); font-weight: 600; }
@media (max-width: 900px) { .af-outlook { grid-template-columns: 1fr; } }

/* ---------- Alerts, rows, stats ---------- */
.af-alert { display: flex; gap: 10px; padding: 10px 0; border-top: 1px solid var(--border); }
.af-alert:first-child { border-top: none; padding-top: 4px; }
.af-alert > svg { flex: none; margin-top: 2px; }
.af-alert.good > svg { color: var(--good); } .af-alert.warn > svg { color: var(--warn); } .af-alert.bad > svg { color: var(--bad); }
.af-alert-title { font-size: 13.5px; font-weight: 600; color: var(--ink); }
.af-alert-title span { font-size: 12px; font-weight: 500; margin-left: 6px; }
.af-alert.good .af-alert-title span { color: var(--good); } .af-alert.warn .af-alert-title span { color: var(--warn); }
.af-alert.bad .af-alert-title span { color: var(--bad); }
.af-alert-body { font-size: 12.5px; color: var(--ink-2); line-height: 1.5; margin-top: 2px; }
.af-row { display: flex; justify-content: space-between; gap: 12px; padding: 9px 0; border-top: 1px solid var(--border); font-size: 13px; }
.af-row:first-child { border-top: none; padding-top: 4px; }
.af-row .k { color: var(--muted); } .af-row .v { color: var(--ink); font-weight: 500; text-align: right; }
.af-row .v small { color: var(--muted); font-weight: 400; margin-left: 4px; }
.af-stats { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 0; padding: 2px 0 8px; }
.af-stat { padding: 4px 16px; border-left: 1px solid var(--border); }
.af-stat:first-child { border-left: none; padding-left: 0; }
.af-stat .k { font-size: 12px; color: var(--muted); }
.af-stat .v { font-size: 16px; font-weight: 600; color: var(--ink); margin-top: 4px; }
@media (max-width: 1100px) { .af-stats { grid-template-columns: repeat(3, minmax(0, 1fr)); row-gap: 12px; } }

/* ---------- Tables ---------- */
.af-table-wrap { overflow: auto; max-height: 560px; border: 1px solid var(--border); border-radius: 8px; }
.af-table { width: 100%; border-collapse: collapse; font-size: 13px; font-variant-numeric: tabular-nums; }
.af-table th { position: sticky; top: 0; z-index: 1; background: var(--surface-2); color: var(--muted); font-weight: 500;
  font-size: 12px; text-align: right; padding: 9px 12px; white-space: nowrap; border-bottom: 1px solid var(--border); }
.af-table td { padding: 9px 12px; text-align: right; color: var(--ink); border-bottom: 1px solid var(--border); white-space: nowrap; }
.af-table th:first-child, .af-table td:first-child { text-align: left; }
.af-table td:first-child { color: var(--ink-2); }
.af-table tr:last-child td { border-bottom: none; }
.af-table tbody tr:hover td { background: var(--surface-2); }
.af-swatch { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; }
.af-callout { display: flex; gap: 10px; align-items: center; padding: 12px 14px; border-radius: 8px; font-size: 13px;
  color: var(--ink-2); background: var(--surface); border: 1px solid var(--border); }
.af-callout svg { color: var(--muted); flex: none; }

/* ---------- Sidebar blocks ---------- */
.af-brand { display: flex; align-items: center; gap: 10px; padding-bottom: 16px; margin-bottom: 4px; border-bottom: 1px solid var(--border); }
.af-mark { width: 30px; height: 30px; border-radius: 7px; background: var(--accent); color: #fff; display: flex; align-items: center; justify-content: center; }
.af-brand-name { font-size: 15px; font-weight: 600; color: var(--ink); }
.af-brand-sub { font-size: 12px; color: var(--muted); }
.af-side-title { font-size: 12px; font-weight: 600; color: var(--muted); margin: 8px 0 -2px; }
.af-hint { font-size: 12px; color: var(--muted); line-height: 1.55; margin-top: 12px; }
.af-footer { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-size: 12px; color: var(--muted);
  border-top: 1px solid var(--border); padding-top: 14px; margin-top: 24px; }

[data-testid="stPlotlyChart"] { animation: af-fade .4s ease both; }
@keyframes af-fade { from { opacity: 0; } to { opacity: 1; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; } }
"""
