"""Plotly figure builders. Each takes the active theme dict `t` from ui.THEMES."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from ui import SANS, rgba

# FSI score bands behind the fs_status labels in the data.
FSI_BANDS = [(0, 35, "bad", "Insecure"), (35, 50, "warn", "At risk"), (50, 70, None, "Moderate"), (70, 100, "good", "Secure")]
PROFILE = [("Yield", "Yield", True), ("FSI", "Food security", True), ("ProfitMargin", "Profit margin", True),
           ("Irrigation", "Irrigation", True), ("Mechanization", "Mechanization", True),
           ("SoilHealth", "Soil health", True), ("LossRate", "Low post-harvest loss", False)]


def ordinal(n):
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def style(fig, t, *, height=260, legend=False, years=None, y_range=None, zero_line=False,
          hovermode="x unified", margin=None):
    fig.update_layout(
        height=height,
        margin=margin or dict(l=4, r=10, t=34 if legend else 8, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, size=12, color=t["ink_2"]),
        showlegend=legend,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                    font=dict(family=SANS, size=12, color=t["ink_2"]), itemclick=False, itemdoubleclick=False),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border_strong"],
                        font=dict(family=SANS, size=12, color=t["ink"])),
        transition=dict(duration=400, easing="cubic-in-out"),
        bargap=0.38, barcornerradius=3,
    )
    tick = dict(family=SANS, size=11, color=t["c_tick"])
    title_font = dict(family=SANS, size=11.5, color=t["muted"])
    fig.update_xaxes(showgrid=False, showline=True, linecolor=t["c_axis"], ticks="", zeroline=False, tickfont=tick,
                     title_font=title_font, fixedrange=True, automargin=True)
    if hovermode == "x unified":
        fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor", spikethickness=1,
                         spikedash="solid", spikecolor=t["border_strong"])
    if years is not None:
        span = int(np.max(years) - np.min(years))
        fig.update_xaxes(tickformat="d", dtick=1 if span <= 8 else 2)
    fig.update_yaxes(gridcolor=t["c_grid"], showline=False, ticks="", tickfont=tick, title_font=title_font,
                     zeroline=zero_line, zerolinecolor=t["c_axis"], zerolinewidth=1, range=y_range,
                     separatethousands=True, fixedrange=True, automargin=True)
    return fig


def show(fig, key, **kwargs):
    return st.plotly_chart(fig, key=key, theme=None, config={"displayModeBar": False}, **kwargs)


def line(x, y, name, color, t, unit="", fmt=".2f", symbol="circle"):
    return go.Scatter(
        x=x, y=y, name=name, mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=7, color=color, symbol=symbol, line=dict(color=t["surface"], width=1.5)),
        hovertemplate=f"%{{y:{fmt}}} {unit}<extra>{name}</extra>",
    )


def area(x, y, name, color, t, unit="", fmt=",.0f"):
    trace = line(x, y, name, color, t, unit, fmt)
    trace.update(fill="tozeroy",
                 fillgradient=dict(type="vertical", colorscale=[[0, rgba(color, 0)], [1, rgba(color, 0.14)]]))
    return trace


def bars(x, y, name, color, unit="", fmt=",.0f"):
    return go.Bar(x=x, y=y, name=name, marker=dict(color=color),
                  hovertemplate=f"%{{y:{fmt}}} {unit}<extra>{name}</extra>")


def signed_bars(years, values, t, pos_name, neg_name, unit, fmt=",.0f"):
    """Blue above zero, red below: the validated polarity pair, with a legend so sign is never color-only."""
    fig = go.Figure([bars(years, values.where(values >= 0), pos_name, t["c_b"], unit, fmt),
                     bars(years, values.where(values < 0), neg_name, t["c_red"], unit, fmt)])
    fig.update_layout(barmode="relative")
    return fig


def end_label(fig, x, y, text, t):
    """Label the latest value just above its point, as in editorial charts."""
    fig.add_annotation(x=x, y=y, text=f"<b>{text}</b>", showarrow=False, yshift=13,
                       font=dict(family=SANS, size=11.5, color=t["ink"]))


def _two_groups(stats, country):
    """(others, selected) rows, in that order."""
    mask = stats["Country"] == country
    return stats[~mask], stats[mask]


# ─────────────────────────── Overview ───────────────────────────
def yield_fan(data, slope, intercept, fan, t):
    """Observed yield, the fitted trend, and a projection fan with 50% and 80% prediction ranges."""
    years = data["Year"]
    first, last = int(years.iloc[0]), int(years.iloc[-1])
    xs = [int(x) for x in fan["years"]]
    fig = go.Figure()
    for lo, hi, name, alpha in [("lo80", "hi80", "80% range", 0.10), ("lo50", "hi50", "50% range", 0.20)]:
        fig.add_scatter(x=xs + xs[::-1], y=list(fan[hi]) + list(fan[lo])[::-1], fill="toself", mode="lines",
                        line=dict(width=0), fillcolor=rgba(t["c_a"], alpha), name=name, hoverinfo="skip")
    fig.add_scatter(x=[first, last], y=[slope * first + intercept, slope * last + intercept], mode="lines",
                    name="Trend", line=dict(color=t["c_trend"], width=1.5), hoverinfo="skip")
    fig.add_scatter(x=xs, y=fan["mid"], mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color=t["c_a"], width=1.5, dash="dot"))
    fig.add_scatter(x=xs[1:], y=fan["mid"][1:], mode="markers", name="Projection",
                    customdata=np.stack([fan["lo80"][1:], fan["hi80"][1:]], axis=-1),
                    marker=dict(size=8, color=t["surface"], line=dict(color=t["c_a"], width=2)),
                    hovertemplate="%{y:.2f} MT/ha · 80% range %{customdata[0]:.2f}–%{customdata[1]:.2f}<extra>Projection</extra>")
    fig.add_trace(line(years, data["Yield"], "Yield", t["c_a"], t, "MT/ha"))
    end_label(fig, last, data["Yield"].iloc[-1], f"{data['Yield'].iloc[-1]:.2f}", t)
    return style(fig, t, height=320, legend=True, years=np.array([first, xs[-1]]))


def fsi_bands(data, t):
    fig = go.Figure()
    for y0, y1, tone, label in FSI_BANDS:
        color = t[tone] if tone else t["c_trend"]
        fig.add_hrect(y0=y0, y1=y1, fillcolor=rgba(color, 0.05), line_width=0, layer="below")
        fig.add_annotation(xref="paper", x=1.01, xanchor="left", y=(y0 + y1) / 2, text=label, showarrow=False,
                           font=dict(family=SANS, size=11, color=t["muted"]))
    fig.add_trace(line(data["Year"], data["FSI"], "FSI", t["c_violet"], t, "", ".0f"))
    end_label(fig, data["Year"].iloc[-1], data["FSI"].iloc[-1], f"{data['FSI'].iloc[-1]:.0f}", t)
    style(fig, t, height=290, years=data["Year"], y_range=[0, 100], margin=dict(l=4, r=70, t=8, b=4))
    fig.update_yaxes(dtick=25)
    return fig


def latest_vs_history(data, t):
    """How far the latest record sits from its own period average, in standard deviations."""
    metrics = [("Yield", "Yield"), ("FSI", "Food security"), ("Production", "Production"), ("Price", "Price"),
               ("ProfitMargin", "Profit margin"), ("LossRate", "Post-harvest loss"), ("Rainfall", "Rainfall"),
               ("Temperature", "Temperature")]
    rows = []
    for col, label in metrics:
        s = data[col].astype(float)
        sd = s.std(ddof=1)
        z = (s.iloc[-1] - s.mean()) / sd if sd and np.isfinite(sd) else 0.0
        rows.append((label, z, s.iloc[-1], s.mean()))
    d = pd.DataFrame(rows, columns=["Metric", "Z", "Latest", "Mean"])
    fig = go.Figure()
    for name, part, color in [("Above average", d[d["Z"] >= 0], t["c_b"]), ("Below average", d[d["Z"] < 0], t["c_red"])]:
        fig.add_bar(x=part["Z"], y=part["Metric"], orientation="h", name=name, marker_color=color,
                    text=[f"{z:+.1f}" for z in part["Z"]], textposition="outside", cliponaxis=False,
                    textfont=dict(family=SANS, size=11, color=t["ink_2"]),
                    customdata=np.stack([part["Latest"], part["Mean"]], axis=-1) if len(part) else None,
                    hovertemplate="%{y}: %{x:+.2f} SD<br>Latest %{customdata[0]:,.2f} · average %{customdata[1]:,.2f}<extra></extra>")
    lim = max(2.0, float(d["Z"].abs().max()) * 1.3)
    style(fig, t, height=310, legend=True, hovermode="closest", zero_line=True)
    fig.update_xaxes(range=[-lim, lim], showline=False, title_text="SD from period average")
    fig.update_yaxes(showgrid=False, categoryorder="array", categoryarray=[m for _, m in metrics][::-1],
                     tickfont=dict(family=SANS, size=12, color=t["ink_2"]))
    fig.update_layout(bargap=0.35)
    return fig


def bullets(stats, country, t):
    """Bullet charts: the country's average against the spread of all countries growing the crop."""
    metrics = [("Mechanization", "Mechanization", "%", 100), ("Irrigation", "Irrigation", "%", 100),
               ("SoilHealth", "Soil health", "", 100), ("LossRate", "Post-harvest loss", "%", None)]
    row = stats.loc[stats["Country"] == country].iloc[0]
    fig = make_subplots(rows=len(metrics), cols=1, vertical_spacing=0.14)
    for i, (col, label, unit, cap) in enumerate(metrics, start=1):
        s = stats[col]
        v = float(row[col])
        lo, q1, med, q3, hi = s.min(), s.quantile(0.25), s.median(), s.quantile(0.75), s.max()
        fig.add_bar(x=[hi - lo], base=[lo], y=[label], orientation="h", width=0.62, marker_color=t["c_band1"],
                    hovertemplate=f"All countries: {lo:.0f}–{hi:.0f}{unit}<extra>{label}</extra>", row=i, col=1)
        fig.add_bar(x=[q3 - q1], base=[q1], y=[label], orientation="h", width=0.62, marker_color=t["c_band2"],
                    hovertemplate=f"Middle 50%: {q1:.0f}–{q3:.0f}{unit}<extra>{label}</extra>", row=i, col=1)
        fig.add_bar(x=[v], base=[0], y=[label], orientation="h", width=0.24, marker_color=t["c_a"],
                    hovertemplate=f"{country}: %{{x:.1f}}{unit}<extra>{label}</extra>", row=i, col=1)
        fig.add_scatter(x=[med], y=[label], mode="markers",
                        marker=dict(symbol="line-ns", size=24, line=dict(color=t["ink"], width=2)),
                        hovertemplate=f"Median: %{{x:.1f}}{unit}<extra>{label}</extra>", row=i, col=1)
        fig.update_xaxes(range=[0, cap or float(np.ceil(max(hi, v) * 1.15))], row=i, col=1)
        fig.update_yaxes(range=[-0.5, 0.5], row=i, col=1)
        fig.add_annotation(xref="paper", yref=f"y{'' if i == 1 else i} domain", x=1.01, y=0.5, xanchor="left",
                           showarrow=False, text=f"<b>{v:.0f}{unit}</b>  ·  median {med:.0f}{unit}",
                           font=dict(family=SANS, size=12, color=t["ink_2"]))
    style(fig, t, height=56 * len(metrics) + 20, hovermode="closest", margin=dict(l=4, r=150, t=10, b=4))
    fig.update_layout(barmode="overlay", barcornerradius=2)
    fig.update_yaxes(showgrid=False, tickfont=dict(family=SANS, size=12.5, color=t["ink_2"]))
    fig.update_xaxes(showline=False, showgrid=False, showticklabels=False)
    return fig


# ─────────────────────────── Climate ───────────────────────────
def rainfall_anomaly(data, t):
    mean = data["Rainfall"].mean()
    fig = signed_bars(data["Year"], data["Rainfall"] - mean, t, "Wetter than average", "Drier than average", "mm")
    fig.update_traces(customdata=data["Rainfall"],
                      hovertemplate="%{customdata:,.0f} mm · %{y:+,.0f} vs average<extra>%{fullData.name}</extra>")
    fig.data[1].marker.color = t["c_orange"]
    style(fig, t, legend=True, years=data["Year"], zero_line=True)
    fig.add_annotation(xref="paper", x=1, xanchor="right", y=0, yshift=10, showarrow=False,
                       text=f"average {mean:,.0f} mm", font=dict(family=SANS, size=11, color=t["muted"]))
    return fig


def temperature(data, t):
    mean = data["Temperature"].mean()
    fig = go.Figure(line(data["Year"], data["Temperature"], "Temperature", t["c_orange"], t, "°C", ".1f"))
    fig.add_hline(y=mean, line=dict(color=t["c_trend"], width=1))
    fig.add_annotation(xref="paper", x=1, xanchor="right", y=mean, yshift=10, showarrow=False,
                       text=f"average {mean:.1f} °C", font=dict(family=SANS, size=11, color=t["muted"]))
    return style(fig, t, years=data["Year"])


def rain_vs_yield(data, rain_r, t):
    fig = go.Figure()
    if np.isfinite(rain_r):
        m, b = np.polyfit(data["Rainfall"], data["Yield"], 1)
        rx = np.array([data["Rainfall"].min(), data["Rainfall"].max()])
        fig.add_scatter(x=rx, y=m * rx + b, mode="lines", line=dict(color=t["c_trend"], width=1.5), hoverinfo="skip")
    fig.add_scatter(x=data["Rainfall"], y=data["Yield"], mode="markers", customdata=data["Year"],
                    marker=dict(size=10, color=t["c_a"], line=dict(color=t["surface"], width=1.5)),
                    hovertemplate="<b>%{customdata}</b><br>Rainfall %{x:,.0f} mm<br>Yield %{y:.2f} MT/ha<extra></extra>")
    for idx in {data["Yield"].idxmax(), data["Yield"].idxmin()}:
        row = data.loc[idx]
        fig.add_annotation(x=row["Rainfall"], y=row["Yield"], text=str(int(row["Year"])), showarrow=False, yshift=13,
                           font=dict(family=SANS, size=11, color=t["ink_2"]))
    style(fig, t, height=290, hovermode="closest")
    fig.update_xaxes(title_text="Rainfall (mm)")
    return fig


def correlation_matrix(crop_df, t):
    variables = [("FSI", "FSI"), ("Yield", "Yield"), ("LossRate", "Loss rate"), ("Rainfall", "Rainfall"),
                 ("Temperature", "Temperature"), ("Irrigation", "Irrigation"), ("fertilizer_kg_ha", "Fertilizer"),
                 ("SoilHealth", "Soil health"), ("ProfitMargin", "Margin")]
    variables = [(c, label) for c, label in variables if c in crop_df.columns]
    labels = [label for _, label in variables]
    corr = crop_df[[c for c, _ in variables]].corr().to_numpy()
    z = [[corr[i][j] if j <= i else None for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1, xgap=2, ygap=2, hoverongaps=False,
        colorscale=[[0, t["c_red"]], [0.5, t["c_mid"]], [1, t["c_b"]]],
        texttemplate="%{z:.2f}", textfont=dict(family=SANS, size=10.5, color=t["ink"]),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=8, len=0.8, outlinewidth=0, tickfont=dict(family=SANS, size=10.5, color=t["c_tick"])),
    ))
    style(fig, t, height=360, hovermode="closest")
    fig.update_xaxes(showline=False, tickangle=-35)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


# ─────────────────────────── Economics ───────────────────────────
def indexed_trends(data, t):
    """Yield, production, price and FSI on one axis, each indexed to its first record."""
    series = [("Yield", "Yield", t["c_a"], ".2f", "MT/ha"), ("Production", "Production", t["c_b"], ",.0f", "kt"),
              ("Price", "Price", t["c_orange"], ",.0f", "USD/MT"), ("FSI", "Food security", t["c_violet"], ".0f", "")]
    fig = go.Figure()
    values = []
    for col, name, color, fmt, unit in series:
        base = float(data[col].iloc[0])
        if not base:
            continue
        idx = data[col] / base * 100
        values.extend(idx.tolist())
        trace = line(data["Year"], idx, name, color, t)
        trace.update(customdata=data[col],
                     hovertemplate=f"%{{y:.0f}} · %{{customdata:{fmt}}} {unit}<extra>{name}</extra>")
        fig.add_trace(trace)
    fig.add_hline(y=100, line=dict(color=t["c_axis"], width=1))
    style(fig, t, height=320, legend=True, years=data["Year"])
    positive = [v for v in values if v > 0]
    log = bool(positive) and max(positive) / min(positive) > 12
    first = int(data["Year"].iloc[0])
    fig.update_yaxes(type="log" if log else "linear",
                     title_text=f"Index, {first} = 100" + (" (log scale)" if log else ""))
    if log:
        lo, hi = min(positive), max(positive)
        ticks = [m * 10 ** e for e in range(-1, 5) for m in (1, 2, 5) if lo / 1.5 <= m * 10 ** e <= hi * 1.5]
        fig.update_yaxes(tickvals=ticks, ticktext=[f"{v:g}" for v in ticks])
    return fig


def kt(v):
    """Thousand tonnes as a short label."""
    return f"{v / 1000:,.2f} Mt" if abs(v) >= 1000 else f"{v:,.0f} kt"


def production_decomposition(data, t):
    """Split the change in production between the first and latest record into yield and area effects.

    Production (kt) = yield (t/ha) x area (ha) / 1000, so the change is exactly
    yield effect + area effect + their interaction.
    """
    f, l = data.iloc[0], data.iloc[-1]
    y0, y1, a0, a1 = f["Yield"], l["Yield"], f["Area"], l["Area"]
    p0, p1 = y0 * a0 / 1000, y1 * a1 / 1000
    effects = [(y1 - y0) * a0 / 1000, (a1 - a0) * y0 / 1000, (y1 - y0) * (a1 - a0) / 1000]
    labels = [str(int(f["Year"])), "Yield effect", "Area effect", "Both together", str(int(l["Year"]))]

    def signed(v):
        return ("+" if v >= 0 else "−") + kt(abs(v))

    fig = go.Figure(go.Waterfall(
        x=labels, measure=["absolute", "relative", "relative", "relative", "total"], y=[p0, *effects, 0],
        text=[kt(p0), *[signed(e) for e in effects], kt(p1)], textposition="outside", cliponaxis=False,
        textfont=dict(family=SANS, size=11.5, color=t["ink_2"]),
        increasing=dict(marker=dict(color=t["c_b"])), decreasing=dict(marker=dict(color=t["c_red"])),
        totals=dict(marker=dict(color=t["c_a"])), connector=dict(line=dict(color=t["c_axis"], width=1)),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    style(fig, t, height=310, hovermode="closest")
    fig.update_xaxes(type="category")  # "2013"/"2022" labels must not turn the axis numeric
    fig.update_yaxes(title_text="Thousand tonnes", rangemode="tozero")
    return fig


def area_yield_path(data, t):
    """Year-by-year path through harvested area and yield, drawn over curves of equal production."""
    x = data["Area"] / 1000
    y = data["Yield"]
    fig = go.Figure()
    # Production (kt) = yield x area (thousand ha), so a production level P is the curve yield = P / area.
    x_lo, x_hi = x.min() * 0.85, x.max() * 1.1
    y_pad = (y.max() - y.min()) * 0.2 or y.max() * 0.1
    y_lo, y_hi = max(y.min() - y_pad, 0), y.max() + y_pad
    prod = x * y
    levels = sorted({float(f"{v:.2g}") for v in np.geomspace(max(prod.min(), 1e-6), prod.max(), 3)})
    xs = np.linspace(x_lo, x_hi, 80)
    for p in levels:
        fig.add_scatter(x=xs, y=p / xs, mode="lines", line=dict(color=t["c_band2"], width=1),
                        hoverinfo="skip", showlegend=False)
        # Label each curve where it leaves the plot: the top edge if it crosses it, else the right edge.
        x_top = p / y_hi
        if x_lo <= x_top <= x_hi:
            fig.add_annotation(x=x_top, y=y_hi, text=kt(p), showarrow=False, yanchor="bottom", yshift=1,
                               font=dict(family=SANS, size=10.5, color=t["subtle"]))
        elif y_lo <= p / x_hi <= y_hi:
            fig.add_annotation(x=x_hi, y=p / x_hi, text=kt(p), showarrow=False, xanchor="left", xshift=4,
                               font=dict(family=SANS, size=10.5, color=t["subtle"]))
    fig.add_scatter(
        x=x, y=y, mode="lines+markers", showlegend=False,
        line=dict(color=t["c_trend"], width=1.5),
        marker=dict(size=9, color=data["Year"], colorscale=t["seq"][1:], line=dict(color=t["surface"], width=1.5)),
        customdata=np.stack([data["Year"], data["Production"]], axis=-1),
        hovertemplate="<b>%{customdata[0]}</b><br>Area %{x:,.0f} thousand ha<br>Yield %{y:.2f} MT/ha"
                      "<br>Production %{customdata[1]:,.0f} kt<extra></extra>",
    )
    for i, bold in [(0, False), (len(data) - 1, True)]:
        year = int(data["Year"].iloc[i])
        fig.add_annotation(x=x.iloc[i], y=y.iloc[i], text=f"<b>{year}</b>" if bold else str(year), showarrow=False,
                           yshift=13, font=dict(family=SANS, size=11, color=t["ink_2"]))
    style(fig, t, height=310, hovermode="closest", margin=dict(l=4, r=60, t=22, b=4))
    fig.update_xaxes(title_text="Harvested area (thousand ha)", range=[x_lo, x_hi])
    fig.update_yaxes(title_text="Yield (MT/ha)", range=[y_lo, y_hi])
    return fig


def harvest_sankey(latest, t):
    prod = float(latest["Production"])
    phl = float(latest.get("post_harvest_loss_pct", 0)) / 100
    waste = float(latest.get("food_waste_pct", 0)) / 100
    lost, market = prod * phl, prod * (1 - phl)
    wasted = min(prod * waste, market)
    used = market - wasted
    # The middle node stays unlabelled: its label would collide with the consumed share beside it.
    labels = [f"Harvest · {kt(prod)}", f"Post-harvest loss · {phl * 100:.1f}%", "",
              f"Food waste · {waste * 100:.1f}%", f"Consumed / used · {used / prod * 100 if prod else 0:.1f}%"]
    colors = [t["c_a"], t["c_red"], t["c_b"], t["c_orange"], t["c_a"]]
    target = [1, 2, 3, 4]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colors, pad=26, thickness=12, line=dict(width=0),
                  customdata=["Harvest", "Post-harvest loss", "Reaches market", "Food waste", "Consumed / used"],
                  hovertemplate="%{customdata}<br>%{value:,.0f} kt<extra></extra>"),
        link=dict(source=[0, 0, 2, 2], target=target, value=[lost, market, wasted, used],
                  color=[rgba(colors[i], 0.22) for i in target],
                  hovertemplate="%{source.customdata} → %{target.customdata}<br>%{value:,.0f} kt<extra></extra>"),
        textfont=dict(family=SANS, size=12, color=t["ink"]),
    ))
    fig.update_layout(height=290, margin=dict(l=4, r=4, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=SANS, color=t["ink"]),
                      hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border_strong"],
                                      font=dict(family=SANS, size=12, color=t["ink"])))
    return fig


# ─────────────────────────── Benchmark ───────────────────────────
def world_map(stats, metric, label, fmt, selected, t):
    scale = [[i / (len(t["seq"]) - 1), c] for i, c in enumerate(t["seq"])]
    fig = go.Figure(go.Choropleth(
        locations=stats["iso3"], z=stats[metric], text=stats["Country"], colorscale=scale,
        marker=dict(line=dict(color=t["surface"], width=0.6)),
        colorbar=dict(title=dict(text=label, font=dict(family=SANS, size=11, color=t["muted"])), thickness=9,
                      len=0.62, x=0.98, outlinewidth=0, tickfont=dict(family=SANS, size=10.5, color=t["c_tick"])),
        hovertemplate="<b>%{text}</b><br>" + label + " %{z:" + fmt + "}<extra></extra>",
    ))
    sel = stats[stats["Country"] == selected]
    if len(sel):
        fig.add_trace(go.Choropleth(locations=sel["iso3"], z=[1], showscale=False, hoverinfo="skip",
                                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                    marker=dict(line=dict(color=t["accent"], width=2.2))))
    fig.update_geos(projection_type="natural earth", showframe=False, showcoastlines=False, showland=True,
                    landcolor=t["c_land"], showcountries=True, countrycolor=t["border"], showocean=False,
                    showlakes=False, bgcolor="rgba(0,0,0,0)", lataxis_range=[-56, 84])
    fig.update_layout(height=390, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border_strong"],
                                      font=dict(family=SANS, size=12, color=t["ink"])))
    return fig


def ranking(stats, metric, fmt, selected, t, top_n=10):
    ranked = stats.sort_values(metric, ascending=False).reset_index(drop=True)
    ranked["Rank"] = ranked.index + 1
    rows = ranked.head(top_n)
    if selected not in rows["Country"].values:
        rows = pd.concat([rows, ranked[ranked["Country"] == selected]])
    rows = rows.iloc[::-1]
    names = [f"<b>{r}. {c}</b>" if c == selected else f"{r}. {c}" for r, c in zip(rows["Rank"], rows["Country"])]
    fig = go.Figure(go.Bar(
        x=rows[metric], y=names, orientation="h", cliponaxis=False,
        marker=dict(color=[t["c_a"] if c == selected else t["c_peer"] for c in rows["Country"]]),
        text=[format(v, fmt) for v in rows[metric]], textposition="outside",
        textfont=dict(family=SANS, size=11, color=t["ink_2"]),
        hovertemplate="%{y}<br>%{x:" + fmt + "}<extra></extra>",
    ))
    style(fig, t, height=29 * len(rows) + 30, hovermode="closest", margin=dict(l=4, r=48, t=4, b=4))
    fig.update_xaxes(showticklabels=False, showline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(family=SANS, size=12, color=t["ink_2"]))
    fig.update_layout(bargap=0.32)
    return fig


def distribution_bands(crop_df, country, t):
    """The country's yield against the spread of every country growing the crop, year by year."""
    q = crop_df.groupby("Year")["Yield"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).unstack()
    own = crop_df[crop_df["Country"] == country].sort_values("Year")
    years = q.index
    fig = go.Figure()
    for low, high, name, alpha in [(0.1, 0.9, "Middle 80% of countries", 0.10), (0.25, 0.75, "Middle 50%", 0.20)]:
        fig.add_scatter(x=years, y=q[low], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False)
        fig.add_scatter(x=years, y=q[high], mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor=rgba(t["c_b"], alpha), name=name, hoverinfo="skip")
    fig.add_scatter(x=years, y=q[0.5], mode="lines", name="Median country", line=dict(color=t["c_b"], width=1.5),
                    customdata=np.stack([q[0.25], q[0.75]], axis=-1),
                    hovertemplate="%{y:.2f} MT/ha · middle 50% %{customdata[0]:.2f}–%{customdata[1]:.2f}<extra>Median</extra>")
    fig.add_trace(line(own["Year"], own["Yield"], country, t["c_a"], t, "MT/ha"))
    style(fig, t, height=320, legend=True, years=years)
    fig.update_yaxes(title_text="Yield (MT/ha)")
    return fig


def peer_scatter(stats, selected, t):
    ref = 2.0 * stats["Production"].max() / (36 ** 2)
    fig = go.Figure()
    for is_sel, group in zip((False, True), _two_groups(stats, selected)):
        fig.add_scatter(
            x=group["Yield"], y=group["FSI"], mode="markers+text" if is_sel else "markers",
            name=selected if is_sel else "Other countries",
            text=group["Country"] if is_sel else None, textposition="top center",
            textfont=dict(family=SANS, size=12, color=t["ink"]),
            customdata=np.stack([group["Country"], group["Region"], group["Production"]], axis=-1),
            marker=dict(size=group["Production"], sizemode="area", sizeref=ref, sizemin=5,
                        color=t["c_a"] if is_sel else rgba(t["c_peer"], 0.85),
                        line=dict(color=t["surface"], width=1)),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>Yield %{x:.2f} MT/ha · FSI %{y:.0f}"
                          "<br>Production %{customdata[2]:,.0f} kt<extra></extra>",
        )
    fig.add_vline(x=stats["Yield"].median(), line=dict(color=t["c_axis"], width=1))
    fig.add_hline(y=stats["FSI"].median(), line=dict(color=t["c_axis"], width=1))
    style(fig, t, height=370, legend=True, hovermode="closest")
    fig.update_xaxes(title_text="Average yield (MT/ha)")
    fig.update_yaxes(title_text="Average FSI")
    fig.update_layout(clickmode="event+select")
    return fig


def stability(stats, selected, t):
    """Average yield against year-to-year volatility: up and to the left is better."""
    fig = go.Figure()
    for is_sel, group in zip((False, True), _two_groups(stats, selected)):
        fig.add_scatter(
            x=group["YieldCV"], y=group["Yield"], mode="markers+text" if is_sel else "markers",
            name=selected if is_sel else "Other countries", text=group["Country"] if is_sel else None,
            textposition="top center", textfont=dict(family=SANS, size=12, color=t["ink"]),
            customdata=group["Country"],
            marker=dict(size=12 if is_sel else 9, color=t["c_a"] if is_sel else rgba(t["c_peer"], 0.85),
                        line=dict(color=t["surface"], width=1)),
            hovertemplate="<b>%{customdata}</b><br>Average yield %{y:.2f} MT/ha<br>Volatility %{x:.1f}%<extra></extra>",
        )
    fig.add_vline(x=stats["YieldCV"].median(), line=dict(color=t["c_axis"], width=1))
    fig.add_hline(y=stats["Yield"].median(), line=dict(color=t["c_axis"], width=1))
    for x, y, xa, ya, text in [(0.01, 0.99, "left", "top", "Higher yield, more stable"),
                               (0.99, 0.01, "right", "bottom", "Lower yield, more volatile")]:
        fig.add_annotation(xref="paper", yref="paper", x=x, y=y, xanchor=xa, yanchor=ya, text=text, showarrow=False,
                           font=dict(family=SANS, size=11, color=t["subtle"]))
    style(fig, t, height=370, legend=True, hovermode="closest")
    fig.update_xaxes(title_text="Yield volatility (coefficient of variation, %)")
    fig.update_yaxes(title_text="Average yield (MT/ha)")
    return fig


def percentile_dots(profiles, t):
    """profiles: list of (name, {label: percentile}, color, symbol). Two profiles draw as a dumbbell."""
    labels = [label for _, label, _ in PROFILE]
    fig = go.Figure()
    for label in labels:
        fig.add_shape(type="line", x0=0, x1=100, y0=label, y1=label, layer="below",
                      line=dict(color=t["c_band1"], width=6))
    fig.add_vline(x=50, line=dict(color=t["c_axis"], width=1))
    if len(profiles) == 2:
        a, b = profiles[0][1], profiles[1][1]
        for label in labels:
            fig.add_shape(type="line", x0=a[label], x1=b[label], y0=label, y1=label, layer="below",
                          line=dict(color=t["c_trend"], width=2))
    single = len(profiles) == 1
    for name, values, color, symbol in profiles:
        xs = [values[label] for label in labels]
        fig.add_scatter(
            x=xs, y=labels, name=name, mode="markers+text" if single else "markers",
            text=[ordinal(v) for v in xs] if single else None, textposition="middle right",
            textfont=dict(family=SANS, size=11.5, color=t["ink_2"]), customdata=[ordinal(v) for v in xs],
            marker=dict(size=12, color=color, symbol=symbol, line=dict(color=t["surface"], width=2)),
            hovertemplate="%{y}: %{customdata} percentile<extra>" + name + "</extra>",
        )
    style(fig, t, height=40 * len(labels) + 70, legend=True, hovermode="closest")
    fig.update_xaxes(range=[-3, 108], tickvals=[0, 25, 50, 75, 100], showline=False,
                     title_text="Percentile (50 = median)")
    fig.update_yaxes(autorange="reversed", showgrid=False, tickfont=dict(family=SANS, size=12, color=t["ink_2"]))
    return fig


# ─────────────────────────── Portfolio ───────────────────────────
def portfolio_heatmap(country_df, selected_crop, t):
    d = country_df[["Crop", "Year", "Yield"]].copy()
    d["Index"] = d["Yield"] / d.groupby("Crop")["Yield"].transform("mean") * 100
    index = d.pivot_table(index="Crop", columns="Year", values="Index")
    raw = d.pivot_table(index="Crop", columns="Year", values="Yield").reindex_like(index)
    crops = index.index.tolist()
    fig = go.Figure(go.Heatmap(
        z=index.values, x=index.columns, y=crops, customdata=raw.values, zmid=100, zmin=70, zmax=130,
        xgap=2, ygap=2, hoverongaps=False, colorscale=[[0, t["c_red"]], [0.5, t["c_mid"]], [1, t["c_b"]]],
        hovertemplate="<b>%{y}</b> · %{x}<br>Yield %{customdata:.2f} MT/ha<br>%{z:.0f}% of its average<extra></extra>",
        colorbar=dict(title=dict(text="% of avg", font=dict(family=SANS, size=11, color=t["muted"])), thickness=8,
                      len=0.8, outlinewidth=0, tickfont=dict(family=SANS, size=10.5, color=t["c_tick"])),
    ))
    style(fig, t, height=max(260, 27 * len(crops) + 60), hovermode="closest", years=index.columns)
    fig.update_xaxes(showline=False)
    fig.update_yaxes(autorange="reversed", showgrid=False, tickmode="array", tickvals=crops,
                     ticktext=[f"<b>{c}</b>" if c == selected_crop else c for c in crops],
                     tickfont=dict(family=SANS, size=12, color=t["ink_2"]))
    return fig


def crop_trends(country_df, crops, selected_crop, t):
    rows = []
    for crop in crops:
        d = country_df[country_df["Crop"] == crop]
        if len(d) >= 3 and d["Yield"].mean():
            slope = np.polyfit(d["Year"], d["Yield"], 1)[0]
            rows.append((crop, slope / d["Yield"].mean() * 100))
    trends = pd.DataFrame(rows, columns=["Crop", "Pct"]).sort_values("Pct")
    names = [f"<b>{c}</b>" if c == selected_crop else c for c in trends["Crop"]]
    fig = go.Figure(go.Bar(
        x=trends["Pct"], y=names, orientation="h", cliponaxis=False,
        marker=dict(color=[t["c_b"] if v >= 0 else t["c_red"] for v in trends["Pct"]]),
        text=[f"{v:+.1f}%" for v in trends["Pct"]], textposition="outside",
        textfont=dict(family=SANS, size=11, color=t["ink_2"]),
        hovertemplate="%{y}: %{x:+.1f}% per year<extra></extra>",
    ))
    style(fig, t, height=max(260, 27 * len(trends) + 40), hovermode="closest", zero_line=True,
          margin=dict(l=4, r=44, t=4, b=4))
    lo, hi = min(trends["Pct"].min(), 0), max(trends["Pct"].max(), 0)
    pad = (hi - lo) or 1
    # Room on both sides so labels on negative bars clear the crop names.
    fig.update_xaxes(showticklabels=False, showline=False, range=[lo - pad * 0.45, hi + pad * 0.25])
    fig.update_yaxes(showgrid=False, tickfont=dict(family=SANS, size=12, color=t["ink_2"]))
    fig.update_layout(bargap=0.3)
    return fig


def portfolio_treemap(country_df, t):
    g = (country_df.groupby(["crop_category", "Crop"])
         .agg(Production=("Production", "mean"), Margin=("ProfitMargin", "mean")).reset_index())
    fig = px.treemap(g, path=[px.Constant("All crops"), "crop_category", "Crop"], values="Production",
                     color="Margin", color_continuous_scale=t["seq"], template="none")
    fig.update_traces(
        marker=dict(line=dict(color=t["surface"], width=2), cornerradius=4), root_color="rgba(0,0,0,0)",
        texttemplate="<b>%{label}</b><br>%{value:,.0f} kt", textfont=dict(family=SANS, size=12.5),
        hovertemplate="<b>%{label}</b><br>Avg production %{value:,.0f} kt<br>Avg margin %{color:.1f}%<extra></extra>",
        pathbar_visible=False,
    )
    fig.update_layout(
        height=370, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family=SANS),
        coloraxis_colorbar=dict(title=dict(text="Margin %", font=dict(family=SANS, size=11, color=t["muted"])),
                                thickness=9, len=0.7, outlinewidth=0,
                                tickfont=dict(family=SANS, size=10.5, color=t["c_tick"])),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["border_strong"],
                        font=dict(family=SANS, size=12, color=t["ink"])),
    )
    return fig


