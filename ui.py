"""Theme tokens, global CSS and small HTML helpers for the AgriFlow UI.

The app pins Streamlit's own theme to light and draws both modes itself: every
color below becomes a CSS variable, so the theme toggle only swaps the variable
block and the page fades between modes instead of reloading.
"""
import streamlit as st

SANS = "Inter, system-ui, -apple-system, 'Segoe UI', sans-serif"
MONO = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"

# Chart pairs that share a plot (crop A/B, blue/red polarity) pass the
# color-vision-deficiency check against each mode's surface.
THEMES = {
    "dark": {
        "bg": "#0a0c10", "surface": "#111318", "surface_2": "#161a21", "field": "#0f1116",
        "border": "#232834", "border_strong": "#323a4a",
        "ink": "#e8eaf0", "ink_2": "#aab1bf", "muted": "#7d8595", "subtle": "#586070",
        "accent": "#2dd4a0", "accent_ink": "#2dd4a0", "accent_soft": "rgba(45,212,160,.12)",
        "accent_glow": "rgba(45,212,160,.45)",
        "glow": "rgba(45,212,160,.10)", "glow_2": "rgba(74,143,232,.07)", "grid": "rgba(120,140,170,.055)",
        "sheen": "linear-gradient(180deg, rgba(255,255,255,.028), rgba(255,255,255,0) 42%)",
        "shadow": "0 1px 0 rgba(255,255,255,.03) inset, 0 10px 30px rgba(0,0,0,.35)",
        "shadow_lg": "0 18px 48px rgba(0,0,0,.55)",
        "good": "#4ade80", "good_bg": "rgba(74,222,128,.12)", "good_line": "rgba(74,222,128,.30)",
        "warn": "#fbbf24", "warn_bg": "rgba(251,191,36,.12)", "warn_line": "rgba(251,191,36,.30)",
        "bad": "#f87171", "bad_bg": "rgba(248,113,113,.12)", "bad_line": "rgba(248,113,113,.30)",
        # charts
        "c_a": "#1aaa80", "c_b": "#4a8fe8", "c_violet": "#9085e9", "c_orange": "#e8743b", "c_red": "#e24b4a",
        "c_trend": "#5b6475", "c_peer": "#4b5364", "c_grid": "rgba(255,255,255,.06)", "c_axis": "#2a303c",
        "c_tick": "#7d8595", "c_mid": "#2c313b", "c_land": "#171b22",
        "seq": ["#14241f", "#154a3b", "#177a60", "#1aaa80", "#7de3c1"],
    },
    "light": {
        "bg": "#f3f6f4", "surface": "#ffffff", "surface_2": "#f5f8f6", "field": "#ffffff",
        "border": "#e2e8e5", "border_strong": "#cdd6d2",
        "ink": "#0f1a15", "ink_2": "#44524b", "muted": "#66756e", "subtle": "#9aa6a0",
        "accent": "#0e9f6e", "accent_ink": "#0b7d57", "accent_soft": "rgba(14,159,110,.10)",
        "accent_glow": "rgba(14,159,110,.40)",
        "glow": "rgba(14,159,110,.10)", "glow_2": "rgba(42,120,214,.06)", "grid": "rgba(16,24,40,.045)",
        "sheen": "none",
        "shadow": "0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.05)",
        "shadow_lg": "0 18px 48px rgba(16,24,40,.14)",
        "good": "#067647", "good_bg": "#ecfdf3", "good_line": "#abefc6",
        "warn": "#b54708", "warn_bg": "#fffaeb", "warn_line": "#fedf89",
        "bad": "#b42318", "bad_bg": "#fef3f2", "bad_line": "#fecdca",
        "c_a": "#0e9f6e", "c_b": "#2a78d6", "c_violet": "#4a3aa7", "c_orange": "#eb6834", "c_red": "#e34948",
        "c_trend": "#9aa6a0", "c_peer": "#c3ccc8", "c_grid": "#edf1ef", "c_axis": "#cdd6d2",
        "c_tick": "#66756e", "c_mid": "#f0efec", "c_land": "#e9eeeb",
        "seq": ["#e3f5ec", "#a6e0c6", "#5bc29b", "#16a275", "#0b6b4a"],
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
    "up":    '<path d="M2.5 11l4-4 2.5 2.5 4.5-5"/><path d="M10 4.5h3.5V8"/>',
    "down":  '<path d="M2.5 5l4 4 2.5-2.5 4.5 5"/><path d="M10 11.5h3.5V8"/>',
    "flat":  '<path d="M2.5 8h11"/><path d="M10.5 5l3 3-3 3"/>',
    "alert": '<path d="M8 2.5l6 10.5H2L8 2.5z"/><path d="M8 6.5v3"/><path d="M8 11.5h.01"/>',
    "check": '<circle cx="8" cy="8" r="6"/><path d="M5.5 8.2l1.8 1.8 3.2-3.5"/>',
    "info":  '<circle cx="8" cy="8" r="6"/><path d="M8 7.5v3.5"/><path d="M8 5h.01"/>',
}


def icon(name, size=16):
    return _SVG.format(s=size, p=_PATHS[name])


def card(key, title, subtitle=None, delay=0.0):
    """A styled container; keys starting with 'h-' or 'd-' pick horizontal / diverging bar animations."""
    box = st.container(key=f"card-{key}")
    with box:
        sub = f'<div class="af-card-sub">{subtitle}</div>' if subtitle else ""
        style = f"<style>.st-key-card-{key}{{animation-delay:{delay:.2f}s}}</style>" if delay else ""
        html(f'{style}<div><div class="af-card-title">{title}</div>{sub}</div>')
    return box


def callout(text, kind="info"):
    html(f'<div class="af-callout"><span class="ic neutral">{icon(kind)}</span><div>{text}</div></div>')


def sparkline(values, uid, w=240, h=40):
    """Inline SVG trend line with a soft gradient area; draws itself in via CSS."""
    vals = [float(v) for v in values]
    if len(vals) < 2:
        return ""
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    pts = [(i / (len(vals) - 1) * (w - 6) + 3, h - 4 - (v - lo) / span * (h - 10)) for i, v in enumerate(vals)]
    line = "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    area = f"{line} L{pts[-1][0]:.1f},{h} L{pts[0][0]:.1f},{h} Z"
    x_end, y_end = pts[-1]
    return (
        f'<svg class="af-spark" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
        f'<defs><linearGradient id="sg-{uid}" x1="0" y1="0" x2="0" y2="1">'
        f'<stop offset="0" style="stop-color:var(--accent);stop-opacity:.28"/>'
        f'<stop offset="1" style="stop-color:var(--accent);stop-opacity:0"/></linearGradient></defs>'
        f'<path class="ar" d="{area}" fill="url(#sg-{uid})"/>'
        f'<path class="ln" d="{line}" pathLength="1"/>'
        f'<circle class="dot" cx="{x_end:.1f}" cy="{y_end:.1f}" r="2.6"/></svg>'
    )


def ring(p, size=64):
    """Probability ring (0-1) with the percentage in the middle."""
    r = size / 2 - 5
    c = 2 * 3.14159265 * r
    return (
        f'<div class="af-ring" style="width:{size}px;height:{size}px">'
        f'<svg viewBox="0 0 {size} {size}"><circle class="trk" cx="{size / 2}" cy="{size / 2}" r="{r:.1f}"/>'
        f'<circle class="val" cx="{size / 2}" cy="{size / 2}" r="{r:.1f}" '
        f'style="stroke-dasharray:{p * c:.1f} {c:.1f}" transform="rotate(-90 {size / 2} {size / 2})"/></svg>'
        f'<span>{p * 100:.0f}<small>%</small></span></div>'
    )


def theme_css(mode):
    t = THEMES[mode]
    tokens = ";".join(f"--{k.replace('_', '-')}:{v}" for k, v in t.items() if isinstance(v, str))
    return (f"<style>@import url('https://fonts.googleapis.com/css2?family=Syne:wght@600;700;800"
            f"&family=JetBrains+Mono:wght@400;500;600&display=swap');"
            f":root{{{tokens};color-scheme:{mode}}}{_CSS}</style>")


_CSS = """
:root { --sans: Inter, system-ui, -apple-system, "Segoe UI", sans-serif;
  --mono: "JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  --display: Syne, Inter, system-ui, sans-serif; --ease: cubic-bezier(.2,.7,.2,1); }

/* ---------- App shell ---------- */
.stApp {
  background-color: var(--bg) !important;
  background-image:
    radial-gradient(1100px 540px at 82% -12%, var(--glow), transparent 62%),
    radial-gradient(900px 480px at -12% 108%, var(--glow-2), transparent 60%),
    linear-gradient(var(--grid) 1px, transparent 1px),
    linear-gradient(90deg, var(--grid) 1px, transparent 1px);
  background-size: auto, auto, 34px 34px, 34px 34px;
  background-attachment: fixed;
  color: var(--ink);
  transition: background-color .45s ease, color .45s ease;
}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stHeader"],
[data-testid="stBottomBlockContainer"] { background: transparent !important; }
[data-testid="stDecoration"] { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 2rem; padding-bottom: 3rem; max-width: 1480px; }
[data-testid="stMarkdownContainer"] { color: var(--ink); }
[data-stale="true"] { opacity: .82 !important; transition: opacity .25s ease; }
[data-testid="stStatusWidget"], [data-testid="stStatusWidget"] * { color: var(--muted) !important; }
* { scrollbar-color: var(--border-strong) transparent; scrollbar-width: thin; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background-color: var(--surface) !important; border-right: 1px solid var(--border);
  transition: background-color .45s ease, border-color .45s ease; }
[data-testid="stSidebar"] > div { background: transparent !important; }
[data-testid="stSidebarCollapseButton"] *, [data-testid="stExpandSidebarButton"],
[data-testid="stExpandSidebarButton"] * { color: var(--muted) !important; }

/* ---------- Native widgets, themed through the same tokens ---------- */
[data-testid="stWidgetLabel"] p { font-family: var(--mono) !important; font-size: 10.5px !important;
  font-weight: 500 !important; letter-spacing: .12em; text-transform: uppercase; color: var(--muted) !important; }
[data-baseweb="select"] > div { background-color: var(--field) !important; border: 1px solid var(--border) !important;
  border-radius: 10px !important; transition: border-color .2s ease, background-color .45s ease, box-shadow .2s ease; }
[data-baseweb="select"] > div:hover { border-color: var(--border-strong) !important; }
[data-baseweb="select"] > div:focus-within { border-color: var(--accent) !important; box-shadow: 0 0 0 3px var(--accent-soft) !important; }
[data-baseweb="select"] div, [data-baseweb="select"] input, [data-baseweb="select"] span { color: var(--ink) !important; }
[data-baseweb="select"] svg { color: var(--muted) !important; }
[data-baseweb="popover"] > div { background-color: var(--surface-2) !important; border: 1px solid var(--border) !important;
  border-radius: 12px !important; box-shadow: var(--shadow-lg) !important; }
[data-baseweb="popover"] ul, [data-baseweb="popover"] [role="listbox"] { background-color: var(--surface-2) !important; }
[data-baseweb="popover"] div, [data-baseweb="popover"] ul { border-color: var(--border) !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] [role="option"] { color: var(--ink) !important; background-color: transparent !important; }
[data-baseweb="popover"] li:hover, [data-baseweb="popover"] [role="option"]:hover,
[data-baseweb="popover"] [aria-selected="true"] { background-color: var(--accent-soft) !important; }
button[data-testid^="stBaseButton-segmented_control"] { background-color: var(--field) !important;
  border-color: var(--border) !important; color: var(--ink-2) !important; transition: all .2s ease; }
button[data-testid^="stBaseButton-segmented_control"]:hover { color: var(--ink) !important; border-color: var(--border-strong) !important; }
button[data-testid="stBaseButton-segmented_controlActive"] { background-color: var(--accent-soft) !important;
  border-color: var(--accent) !important; color: var(--accent-ink) !important; z-index: 1; }
button[data-testid^="stBaseButton-segmented_control"] p, button[data-testid^="stBaseButton-segmented_control"] span { color: inherit !important; }
[data-testid="stDownloadButton"] button { background-color: var(--surface-2) !important; color: var(--ink) !important;
  border: 1px solid var(--border) !important; border-radius: 10px !important; transition: all .2s ease; }
[data-testid="stDownloadButton"] button:hover { border-color: var(--accent) !important; color: var(--accent-ink) !important; }
[data-testid="stDownloadButton"] button * { color: inherit !important; }
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p { color: var(--muted) !important;
  font-family: var(--mono); font-size: 11.5px; }

/* Tabs as a pill bar */
.stTabs [data-baseweb="tab-list"] { gap: 2px; padding: 4px; background-color: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; width: fit-content; max-width: 100%; overflow-x: auto; box-shadow: var(--shadow);
  transition: background-color .45s ease, border-color .45s ease; }
.stTabs [data-baseweb="tab"] { height: 34px; padding: 0 15px; border-radius: 8px; background: transparent;
  transition: background-color .25s ease; }
.stTabs [data-baseweb="tab"] p { font-size: 13px; font-weight: 600; color: var(--muted); transition: color .25s ease; }
.stTabs [data-baseweb="tab"]:hover p { color: var(--ink); }
.stTabs [aria-selected="true"] { background-color: var(--accent-soft) !important; }
.stTabs [aria-selected="true"] p { color: var(--accent-ink) !important; }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 16px; }

/* ---------- Cards ---------- */
[class*="st-key-card"] { position: relative; background-color: var(--surface); background-image: var(--sheen);
  border: 1px solid var(--border); border-radius: 16px; padding: 18px 20px 12px; box-shadow: var(--shadow);
  animation: af-rise .65s var(--ease) both;
  transition: background-color .45s ease, border-color .3s ease, box-shadow .3s ease; }
[class*="st-key-card"]::before { content: ""; position: absolute; left: 18px; right: 18px; top: -1px; height: 1px;
  background: linear-gradient(90deg, transparent, var(--accent), transparent); opacity: 0; transition: opacity .35s ease; }
[class*="st-key-card"]:hover { border-color: var(--border-strong); }
[class*="st-key-card"]:hover::before { opacity: .9; }
[data-testid="stColumn"] > [data-testid="stVerticalBlock"] > [data-testid="stLayoutWrapper"]:has(> [class*="st-key-card"]) { flex: 1 1 auto; }
.af-card-title { font-family: var(--display); font-size: 15.5px; font-weight: 700; color: var(--ink); letter-spacing: .005em; }
.af-card-sub { font-family: var(--mono); font-size: 11px; color: var(--muted); margin-top: 4px; }

/* ---------- Header ---------- */
.af-head { animation: af-fade .6s ease both; }
.af-eyebrow { display: inline-flex; align-items: center; gap: 10px; font-family: var(--mono); font-size: 11px;
  font-weight: 500; letter-spacing: .16em; text-transform: uppercase; color: var(--accent-ink); }
.af-live { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); animation: af-pulse 2.4s ease-out infinite; }
.af-title { font-family: var(--display); font-size: 42px; font-weight: 800; letter-spacing: -.025em; line-height: 1.05;
  color: var(--ink); margin-top: 10px; }
.af-title .crop { background: linear-gradient(95deg, var(--ink) 10%, var(--accent) 115%);
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.af-title .in { color: var(--subtle); font-weight: 600; }
.af-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; margin-top: 14px; }
.af-meta-text { font-family: var(--mono); font-size: 11.5px; color: var(--muted); margin-right: 6px; }
.af-chip { display: inline-flex; align-items: center; gap: 7px; padding: 4px 11px; border-radius: 999px; font-size: 12px;
  font-weight: 500; color: var(--ink-2); background: var(--surface); border: 1px solid var(--border); }
.af-chip i { width: 6px; height: 6px; border-radius: 50%; background: var(--subtle); }
.af-chip b { font-weight: 600; }
.af-chip.good { background: var(--good-bg); border-color: var(--good-line); color: var(--good); }
.af-chip.warn { background: var(--warn-bg); border-color: var(--warn-line); color: var(--warn); }
.af-chip.bad  { background: var(--bad-bg);  border-color: var(--bad-line);  color: var(--bad); }
.af-chip.good i { background: var(--good); } .af-chip.warn i { background: var(--warn); } .af-chip.bad i { background: var(--bad); }

/* ---------- KPI tiles ---------- */
.af-kpis { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; }
.af-kpi { position: relative; overflow: hidden; background-color: var(--surface); background-image: var(--sheen);
  border: 1px solid var(--border); border-radius: 16px; padding: 15px 18px 0; box-shadow: var(--shadow);
  animation: af-rise .6s var(--ease) both; transition: background-color .45s ease, border-color .3s ease, transform .3s var(--ease); }
.af-kpi:hover { border-color: var(--border-strong); transform: translateY(-2px); }
.af-kpi:nth-child(2) { animation-delay: .05s; } .af-kpi:nth-child(3) { animation-delay: .1s; }
.af-kpi:nth-child(4) { animation-delay: .15s; } .af-kpi:nth-child(5) { animation-delay: .2s; }
.af-kpi-label { font-family: var(--mono); font-size: 10.5px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.af-kpi-value { font-size: 28px; font-weight: 700; letter-spacing: -.03em; color: var(--ink); margin-top: 9px; line-height: 1.1; white-space: nowrap; }
.af-kpi-value small { font-size: 12.5px; font-weight: 500; color: var(--muted); margin-left: 5px; letter-spacing: 0; }
.af-delta-row { display: flex; align-items: center; gap: 8px; margin-top: 9px; }
.af-delta { display: inline-flex; align-items: center; gap: 4px; padding: 2px 8px; border-radius: 999px;
  font-family: var(--mono); font-size: 11px; font-weight: 600; }
.af-delta.good { color: var(--good); background: var(--good-bg); } .af-delta.bad { color: var(--bad); background: var(--bad-bg); }
.af-delta.neutral { color: var(--ink-2); background: var(--surface-2); }
.af-vs { font-family: var(--mono); font-size: 11px; color: var(--muted); }
.af-spark { display: block; width: calc(100% + 36px); height: 40px; margin: 10px -18px 0; }
.af-spark .ln { fill: none; stroke: var(--accent); stroke-width: 1.8; stroke-linecap: round; stroke-linejoin: round;
  stroke-dasharray: 1; stroke-dashoffset: 1; animation: af-draw 1.5s .25s var(--ease) forwards; }
.af-spark .ar { animation: af-fade 1s .7s ease both; }
.af-spark .dot { fill: var(--accent); animation: af-fade .4s 1.5s ease both; }
@media (max-width: 1100px) { .af-kpis { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 640px) { .af-kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); } .af-kpi:last-child { grid-column: span 2; }
  .af-title { font-size: 32px; } [data-testid="stMainBlockContainer"] { padding-top: 3.2rem; } }

/* ---------- Outlook ---------- */
.af-section { display: flex; align-items: baseline; gap: 10px; margin: 20px 0 2px; }
.af-section b { font-family: var(--display); font-size: 17px; font-weight: 700; color: var(--ink); }
.af-section span { font-family: var(--mono); font-size: 11.5px; color: var(--muted); }
.af-outlook { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 12px; }
.af-panel { position: relative; background-color: var(--surface); background-image: var(--sheen); border: 1px solid var(--border);
  border-radius: 16px; padding: 18px 20px; box-shadow: var(--shadow); animation: af-rise .6s var(--ease) both;
  transition: background-color .45s ease, border-color .3s ease; }
.af-panel:nth-child(2) { animation-delay: .07s; } .af-panel:nth-child(3) { animation-delay: .14s; }
.af-panel:hover { border-color: var(--border-strong); }
.af-panel-label { display: flex; justify-content: space-between; align-items: center; gap: 8px; font-family: var(--mono);
  font-size: 10.5px; letter-spacing: .12em; text-transform: uppercase; color: var(--muted); }
.af-tag { font-family: var(--mono); font-size: 10.5px; font-weight: 600; letter-spacing: .04em; color: var(--ink-2);
  background: var(--surface-2); border: 1px solid var(--border); border-radius: 6px; padding: 2px 7px; }
.af-panel-main { display: flex; align-items: center; gap: 14px; margin-top: 14px; }
.af-outlook-value { font-size: 22px; font-weight: 700; color: var(--ink); letter-spacing: -.02em; }
.af-outlook-value small { font-size: 12.5px; font-weight: 500; color: var(--muted); letter-spacing: 0; margin-left: 4px; }
.af-panel-foot { font-size: 12.5px; color: var(--muted); margin-top: 12px; line-height: 1.55; }
.af-panel-foot b { color: var(--ink-2); font-weight: 600; }
.af-ring { position: relative; flex: none; }
.af-ring svg { width: 100%; height: 100%; }
.af-ring .trk { fill: none; stroke: var(--surface-2); stroke-width: 6; }
.af-ring .val { fill: none; stroke: var(--accent); stroke-width: 6; stroke-linecap: round; animation: af-ring 1.3s .2s var(--ease) both;
  filter: drop-shadow(0 0 6px var(--accent-soft)); }
.af-ring span { position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
  font-size: 16px; font-weight: 700; color: var(--ink); }
.af-ring span small { font-size: 10px; color: var(--muted); margin-left: 1px; }
.af-range { margin-top: 14px; }
.af-range-track { position: relative; height: 8px; border-radius: 999px; background: var(--surface-2); border: 1px solid var(--border); }
.af-range-band { position: absolute; top: -1px; bottom: -1px; border-radius: 999px; background: var(--accent-soft);
  border: 1px solid var(--accent); animation: af-grow-x .9s .2s var(--ease) both; transform-origin: 0 50%; }
.af-range-mid, .af-range-now { position: absolute; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%; }
.af-range-mid { background: var(--accent); box-shadow: 0 0 0 3px var(--surface); animation: af-fade .4s .9s ease both; }
.af-range-now { background: var(--surface); border: 2px solid var(--ink-2); }
.af-range-legend { display: flex; justify-content: space-between; margin-top: 8px; font-family: var(--mono); font-size: 11px; color: var(--muted); }
.af-range-legend b { color: var(--ink-2); font-weight: 600; }
.ic { flex: none; display: inline-flex; align-items: center; justify-content: center; width: 30px; height: 30px; border-radius: 9px; }
.ic.good { background: var(--good-bg); color: var(--good); } .ic.warn { background: var(--warn-bg); color: var(--warn); }
.ic.bad { background: var(--bad-bg); color: var(--bad); } .ic.neutral { background: var(--surface-2); color: var(--ink-2); }
@media (max-width: 900px) { .af-outlook { grid-template-columns: 1fr; } }

/* ---------- Alerts, rows, meters ---------- */
.af-alert { display: flex; gap: 12px; padding: 12px 0; border-top: 1px solid var(--border); animation: af-rise .5s var(--ease) both; }
.af-alert:first-child { border-top: none; padding-top: 6px; }
.af-alert:nth-child(2) { animation-delay: .08s; } .af-alert:nth-child(3) { animation-delay: .16s; } .af-alert:nth-child(4) { animation-delay: .24s; }
.af-alert-title { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; font-size: 13.5px; font-weight: 600; color: var(--ink); }
.af-sev { font-family: var(--mono); font-size: 10px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; padding: 1px 7px; border-radius: 999px; }
.af-sev.good { background: var(--good-bg); color: var(--good); } .af-sev.warn { background: var(--warn-bg); color: var(--warn); }
.af-sev.bad { background: var(--bad-bg); color: var(--bad); }
.af-alert-body { font-size: 12.5px; color: var(--ink-2); line-height: 1.55; margin-top: 3px; }
.af-row { display: flex; justify-content: space-between; gap: 12px; padding: 10px 0; border-top: 1px solid var(--border); font-size: 13px; }
.af-row:first-child { border-top: none; padding-top: 6px; }
.af-row .k { color: var(--muted); } .af-row .v { color: var(--ink); font-weight: 600; text-align: right; }
.af-row .v small { color: var(--muted); font-weight: 400; font-family: var(--mono); font-size: 11px; }
.af-meters { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 28px; padding: 6px 0 8px; }
.af-meter-top { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; }
.af-meter-label { font-family: var(--mono); font-size: 10.5px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.af-meter-value { font-size: 20px; font-weight: 700; color: var(--ink); letter-spacing: -.02em; }
.af-meter-value small { font-size: 11.5px; font-weight: 500; color: var(--muted); margin-left: 2px; }
.af-bar { height: 6px; border-radius: 999px; background: var(--surface-2); margin-top: 10px; overflow: hidden; }
.af-bar span { display: block; height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--accent-soft), var(--accent));
  transform-origin: 0 50%; animation: af-grow-x 1.1s .2s var(--ease) both; }
.af-meter-note { font-size: 12px; color: var(--muted); margin-top: 8px; }
.af-stats { display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; padding: 4px 0 8px; }
.af-stat { background: var(--surface-2); border: 1px solid var(--border); border-radius: 12px; padding: 12px 14px; }
.af-stat .k { font-family: var(--mono); font-size: 10px; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
.af-stat .v { font-size: 17px; font-weight: 700; color: var(--ink); margin-top: 6px; letter-spacing: -.01em; }
@media (max-width: 1100px) { .af-stats { grid-template-columns: repeat(3, minmax(0, 1fr)); } }
@media (max-width: 900px) { .af-meters { grid-template-columns: repeat(2, minmax(0, 1fr)); } }

/* ---------- Tables ---------- */
.af-table-wrap { overflow: auto; max-height: 560px; border: 1px solid var(--border); border-radius: 12px; }
.af-table { width: 100%; border-collapse: collapse; font-family: var(--mono); font-size: 12px; }
.af-table th { position: sticky; top: 0; z-index: 1; background: var(--surface-2); color: var(--muted); font-weight: 600;
  font-size: 10px; letter-spacing: .08em; text-transform: uppercase; text-align: right; padding: 10px 12px;
  white-space: nowrap; border-bottom: 1px solid var(--border); }
.af-table td { padding: 9px 12px; text-align: right; color: var(--ink); border-bottom: 1px solid var(--border); white-space: nowrap;
  transition: background-color .2s ease; }
.af-table th:first-child, .af-table td:first-child { text-align: left; color: var(--ink-2); }
.af-table tr:last-child td { border-bottom: none; }
.af-table tbody tr:hover td { background: var(--accent-soft); }
.af-swatch { display: inline-block; width: 8px; height: 8px; border-radius: 2px; margin-right: 6px; }
.af-callout { display: flex; gap: 12px; align-items: center; padding: 14px 16px; border-radius: 12px; font-size: 13px;
  color: var(--ink-2); background: var(--surface); border: 1px dashed var(--border-strong); }

/* ---------- Sidebar blocks ---------- */
.af-brand { display: flex; align-items: center; gap: 11px; padding-bottom: 18px; margin-bottom: 4px; border-bottom: 1px solid var(--border); }
.af-mark { width: 36px; height: 36px; border-radius: 10px; color: #fff; display: flex; align-items: center; justify-content: center;
  background: linear-gradient(140deg, var(--accent), #0b7d57); box-shadow: 0 6px 18px var(--accent-soft), 0 0 0 1px rgba(255,255,255,.08) inset; }
.af-brand-name { font-family: var(--display); font-size: 17px; font-weight: 800; color: var(--ink); letter-spacing: -.01em; }
.af-brand-name span { color: var(--accent-ink); }
.af-brand-sub { font-family: var(--mono); font-size: 10.5px; color: var(--muted); letter-spacing: .04em; }
.af-side-title { display: flex; align-items: center; gap: 10px; font-family: var(--mono); font-size: 10px; letter-spacing: .16em;
  text-transform: uppercase; color: var(--subtle); margin: 10px 0 2px; }
.af-side-title::after { content: ""; flex: 1; height: 1px; background: var(--border); }
.af-cover { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
.af-cover div { background: var(--surface-2); border: 1px solid var(--border); border-radius: 10px; padding: 10px 12px; }
.af-cover b { display: block; font-size: 16px; font-weight: 700; color: var(--ink); }
.af-cover span { font-family: var(--mono); font-size: 10px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.af-hint { font-size: 12px; color: var(--muted); line-height: 1.6; margin-top: 14px; }
.af-hint b { color: var(--ink-2); }
.af-footer { display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px; font-family: var(--mono); font-size: 11px;
  color: var(--muted); border-top: 1px solid var(--border); padding-top: 16px; margin-top: 26px; }

/* ---------- Chart entrance animations ---------- */
[data-testid="stPlotlyChart"] { animation: af-fade .8s ease both; }
[data-testid="stPlotlyChart"] .scatterlayer .js-line { stroke-dasharray: 4000; stroke-dashoffset: 4000;
  animation: af-draw 1.6s .15s var(--ease) forwards; }
[data-testid="stPlotlyChart"] .scatterlayer .js-fill { animation: af-fade 1s .5s ease both; }
[data-testid="stPlotlyChart"] .scatterlayer .points path { animation: af-fade .5s .75s ease both; }
[data-testid="stPlotlyChart"] .barlayer .point path { transform-box: fill-box; transform-origin: 50% 100%;
  animation: af-grow .85s var(--ease) both; }
[class*="st-key-card-h-"] .barlayer .point path { transform-origin: 0 50%; animation-name: af-grow-x; }
[class*="st-key-card-d-"] .barlayer .point path { animation-name: af-fade; }

@keyframes af-rise { from { opacity: 0; transform: translateY(12px); } to { opacity: 1; transform: none; } }
@keyframes af-fade { from { opacity: 0; } to { opacity: 1; } }
@keyframes af-draw { to { stroke-dashoffset: 0; } }
@keyframes af-grow { from { transform: scaleY(0); } to { transform: scaleY(1); } }
@keyframes af-grow-x { from { transform: scaleX(0); } to { transform: scaleX(1); } }
@keyframes af-ring { from { stroke-dasharray: 0 400; } }
@keyframes af-pulse { 0% { box-shadow: 0 0 0 0 var(--accent-glow); } 70% { box-shadow: 0 0 0 9px transparent; }
  100% { box-shadow: 0 0 0 0 transparent; } }
@media (prefers-reduced-motion: reduce) { *, *::before, *::after { animation: none !important; transition: none !important; }
  .af-spark .ln, [data-testid="stPlotlyChart"] .scatterlayer .js-line { stroke-dashoffset: 0 !important; } }
"""
