"""Plotly figure builders. Each takes the active theme dict `t` from ui.THEMES."""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ui import MONO, SANS, rgba

DISPLAY = "Syne, Inter, sans-serif"
# FSI score bands behind the fs_status labels in the data.
FSI_BANDS = [(0, 35, "bad", "INSECURE"), (35, 50, "warn", "AT RISK"), (50, 70, None, "MODERATE"), (70, 100, "good", "SECURE")]
PROFILE = [("Yield", "Yield", True), ("FSI", "Food security", True), ("ProfitMargin", "Margin", True),
           ("Irrigation", "Irrigation", True), ("Mechanization", "Mechanization", True),
           ("SoilHealth", "Soil health", True), ("LossRate", "Low loss", False)]


def style(fig, t, *, height=260, legend=False, years=None, y_range=None, zero_line=False,
          hovermode="x unified", margin=None):
    fig.update_layout(
        height=height,
        margin=margin or dict(l=4, r=10, t=34 if legend else 8, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, size=12, color=t["ink_2"]),
        showlegend=legend,
        legend=dict(orientation="h", x=0, xanchor="left", y=1.02, yanchor="bottom", bgcolor="rgba(0,0,0,0)",
                    font=dict(family=SANS, size=11.5, color=t["ink_2"]), itemclick=False, itemdoubleclick=False),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor=t["surface_2"], bordercolor=t["border_strong"],
                        font=dict(family=SANS, size=12, color=t["ink"])),
        transition=dict(duration=550, easing="cubic-in-out"),
        bargap=0.38, barcornerradius=5,
    )
    tick = dict(family=MONO, size=10.5, color=t["c_tick"])
    title_font = dict(family=MONO, size=10.5, color=t["muted"])
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


def line(x, y, name, color, t, unit="", fmt=".2f", symbol="circle", smooth=True):
    shape = dict(shape="spline", smoothing=0.55) if smooth else {}
    return go.Scatter(
        x=x, y=y, name=name, mode="lines+markers",
        line=dict(color=color, width=2.4, **shape),
        marker=dict(size=8, color=color, symbol=symbol, line=dict(color=t["surface"], width=2)),
        hovertemplate=f"%{{y:{fmt}}} {unit}<extra>{name}</extra>",
    )


def area(x, y, name, color, t, unit="", fmt=",.0f"):
    trace = line(x, y, name, color, t, unit, fmt)
    trace.update(fill="tozeroy",
                 fillgradient=dict(type="vertical", colorscale=[[0, rgba(color, 0)], [1, rgba(color, 0.32)]]))
    return trace


def bars(x, y, name, color, unit="", fmt=",.0f", customdata=None, hovertemplate=None):
    return go.Bar(x=x, y=y, name=name, marker=dict(color=color), customdata=customdata,
                  hovertemplate=hovertemplate or f"%{{y:{fmt}}} {unit}<extra>{name}</extra>")


def signed_bars(years, values, t, pos_name, neg_name, unit, fmt=",.0f"):
    """Blue above zero, red below: the validated diverging pair, with a legend so sign is never color-only."""
    fig = go.Figure([bars(years, values.where(values >= 0), pos_name, t["c_b"], unit, fmt),
                     bars(years, values.where(values < 0), neg_name, t["c_red"], unit, fmt)])
    fig.update_layout(barmode="relative")
    return fig


# ─────────────────────────── Overview ───────────────────────────
def yield_fan(data, slope, intercept, projections, t):
    years = data["Year"]
    first, last = int(years.iloc[0]), int(years.iloc[-1])
    (y1, p1, lo1, hi1), (y3, p3, lo3, hi3) = projections
    fit_last = slope * last + intercept
    fig = go.Figure()
    fig.add_scatter(x=[last, y1, y3, y3, y1, last], y=[fit_last, hi1, hi3, lo3, lo1, fit_last], fill="toself", mode="lines",
                    fillcolor=rgba(t["c_a"], 0.13), line=dict(width=0), name="Likely range", hoverinfo="skip")
    fig.add_scatter(x=[first, last], y=[slope * first + intercept, fit_last], mode="lines", name="Trend",
                    line=dict(color=t["c_trend"], width=1.5), hoverinfo="skip")
    fig.add_scatter(x=[last, y1, y3], y=[fit_last, p1, p3], mode="lines", showlegend=False, hoverinfo="skip",
                    line=dict(color=t["c_a"], width=1.8, dash="dot"))
    fig.add_scatter(x=[y1, y3], y=[p1, p3], mode="markers", name="Projection", customdata=[[lo1, hi1], [lo3, hi3]],
                    marker=dict(size=10, color=t["surface"], line=dict(color=t["c_a"], width=2.2)),
                    hovertemplate="%{y:.2f} MT/ha · range %{customdata[0]:.2f}–%{customdata[1]:.2f}<extra>Projection</extra>")
    fig.add_trace(line(years, data["Yield"], "Yield", t["c_a"], t, "MT/ha"))
    return style(fig, t, height=310, legend=True, years=np.array([first, y3]))


def fsi_bands(data, t):
    fig = go.Figure()
    for y0, y1, tone, label in FSI_BANDS:
        color = t[tone] if tone else t["c_trend"]
        fig.add_hrect(y0=y0, y1=y1, fillcolor=rgba(color, 0.07), line_width=0, layer="below")
        fig.add_annotation(xref="paper", x=1.01, xanchor="left", y=(y0 + y1) / 2, text=label, showarrow=False,
                           font=dict(family=MONO, size=9.5, color=color))
    fig.add_trace(line(data["Year"], data["FSI"], "FSI", t["c_violet"], t, "", ".0f"))
    style(fig, t, height=270, years=data["Year"], y_range=[0, 100], margin=dict(l=4, r=78, t=8, b=4))
    fig.update_yaxes(dtick=25)
    return fig


# ─────────────────────────── Climate ───────────────────────────
def rainfall_anomaly(data, t):
    mean = data["Rainfall"].mean()
    fig = signed_bars(data["Year"], data["Rainfall"] - mean, t, "Wetter than average", "Drier than average", "mm")
    fig.update_traces(customdata=data["Rainfall"],
                      hovertemplate="%{customdata:,.0f} mm · %{y:+,.0f} vs avg<extra>%{fullData.name}</extra>")
    fig.data[1].marker.color = t["c_orange"]
    style(fig, t, legend=True, years=data["Year"], zero_line=True)
    fig.add_annotation(xref="paper", x=1, xanchor="right", y=0, yshift=10, showarrow=False,
                       text=f"avg {mean:,.0f} mm", font=dict(family=MONO, size=10, color=t["muted"]))
    return fig


def temperature(data, t):
    mean = data["Temperature"].mean()
    fig = go.Figure(area(data["Year"], data["Temperature"], "Temperature", t["c_orange"], t, "°C", ".1f"))
    fig.add_hline(y=mean, line=dict(color=t["c_trend"], width=1))
    fig.add_annotation(xref="paper", x=1, xanchor="right", y=mean, yshift=10, showarrow=False,
                       text=f"avg {mean:.1f} °C", font=dict(family=MONO, size=10, color=t["muted"]))
    return style(fig, t, years=data["Year"])


def rain_vs_yield(data, rain_r, t):
    fig = go.Figure()
    if np.isfinite(rain_r):
        m, b = np.polyfit(data["Rainfall"], data["Yield"], 1)
        rx = np.array([data["Rainfall"].min(), data["Rainfall"].max()])
        fig.add_scatter(x=rx, y=m * rx + b, mode="lines", line=dict(color=t["c_trend"], width=1.5), hoverinfo="skip")
    fig.add_scatter(x=data["Rainfall"], y=data["Yield"], mode="markers", customdata=data["Year"],
                    marker=dict(size=12, color=t["c_a"], line=dict(color=t["surface"], width=2)),
                    hovertemplate="<b>%{customdata}</b><br>Rainfall %{x:,.0f} mm<br>Yield %{y:.2f} MT/ha<extra></extra>")
    for idx in {data["Yield"].idxmax(), data["Yield"].idxmin()}:
        row = data.loc[idx]
        fig.add_annotation(x=row["Rainfall"], y=row["Yield"], text=str(int(row["Year"])), showarrow=False, yshift=15,
                           font=dict(family=MONO, size=10, color=t["ink_2"]))
    style(fig, t, height=290, hovermode="closest")
    fig.update_xaxes(title_text="Rainfall (mm)")
    return fig


def correlation_matrix(crop_df, t):
    variables = [("FSI", "FSI"), ("Yield", "Yield"), ("LossRate", "Loss rate"), ("Rainfall", "Rainfall"),
                 ("Temperature", "Temp"), ("Irrigation", "Irrigation"), ("fertilizer_kg_ha", "Fertilizer"),
                 ("SoilHealth", "Soil health"), ("ProfitMargin", "Margin")]
    variables = [(c, label) for c, label in variables if c in crop_df.columns]
    labels = [label for _, label in variables]
    corr = crop_df[[c for c, _ in variables]].corr().to_numpy()
    z = [[corr[i][j] if j <= i else None for j in range(len(labels))] for i in range(len(labels))]
    fig = go.Figure(go.Heatmap(
        z=z, x=labels, y=labels, zmin=-1, zmax=1, xgap=3, ygap=3, hoverongaps=False,
        colorscale=[[0, t["c_red"]], [0.5, t["c_mid"]], [1, t["c_b"]]],
        texttemplate="%{z:.2f}", textfont=dict(family=MONO, size=10, color=t["ink"]),
        hovertemplate="%{y} × %{x}<br>r = %{z:.2f}<extra></extra>",
        colorbar=dict(thickness=8, len=0.8, outlinewidth=0, tickfont=dict(family=MONO, size=10, color=t["c_tick"])),
    ))
    style(fig, t, height=360, hovermode="closest")
    fig.update_xaxes(showline=False, tickangle=-35)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


# ─────────────────────────── Economics ───────────────────────────
def trade_waterfall(latest, t):
    exports, imports = float(latest.get("export_usd_m", 0)), float(latest.get("import_usd_m", 0))

    def money(v):
        sign = "−" if v < 0 else ""
        return f"{sign}${abs(v) / 1000:.1f}B" if abs(v) >= 1000 else f"{sign}${abs(v):,.0f}M"

    net = exports - imports
    fig = go.Figure(go.Waterfall(
        x=["Exports", "Imports", "Net balance"], y=[exports, -imports, 0], measure=["relative", "relative", "total"],
        increasing=dict(marker=dict(color=t["c_b"])), decreasing=dict(marker=dict(color=t["c_red"])),
        totals=dict(marker=dict(color=t["c_trend"])), connector=dict(line=dict(color=t["c_axis"], width=1)),
        text=[money(exports), money(-imports), money(net)], textposition="outside", cliponaxis=False,
        textfont=dict(family=MONO, size=11, color=t["ink_2"]),
        hovertemplate="%{x}: %{text}<extra></extra>",
    ))
    style(fig, t, height=280, hovermode="closest", zero_line=True)
    fig.update_yaxes(title_text="USD M")
    return fig


def harvest_sankey(latest, t):
    prod = float(latest["Production"])
    phl = float(latest.get("post_harvest_loss_pct", 0)) / 100
    waste = float(latest.get("food_waste_pct", 0)) / 100
    lost, market = prod * phl, prod * (1 - phl)
    wasted = min(prod * waste, market)
    used = market - wasted
    # The middle node stays unlabelled: its label would collide with the consumed share beside it.
    labels = [f"Harvest · {prod:,.0f} MT", f"Post-harvest loss · {phl * 100:.1f}%", "",
              f"Food waste · {waste * 100:.1f}%", f"Consumed / used · {used / prod * 100 if prod else 0:.1f}%"]
    colors = [t["c_a"], t["c_red"], t["c_b"], t["c_orange"], t["c_a"]]
    target = [1, 2, 3, 4]
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(label=labels, color=colors, pad=26, thickness=14, line=dict(width=0),
                  customdata=["Harvest", "Post-harvest loss", "Reaches market", "Food waste", "Consumed / used"],
                  hovertemplate="%{customdata}<br>%{value:,.0f} MT<extra></extra>"),
        link=dict(source=[0, 0, 2, 2], target=target, value=[lost, market, wasted, used],
                  color=[rgba(colors[i], 0.28) for i in target],
                  hovertemplate="%{source.customdata} → %{target.customdata}<br>%{value:,.0f} MT<extra></extra>"),
        textfont=dict(family=SANS, size=12, color=t["ink"]),
    ))
    fig.update_layout(height=290, margin=dict(l=4, r=4, t=10, b=10), paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(family=SANS, color=t["ink"]),
                      hoverlabel=dict(bgcolor=t["surface_2"], bordercolor=t["border_strong"],
                                      font=dict(family=SANS, size=12, color=t["ink"])))
    return fig


# ─────────────────────────── Benchmark ───────────────────────────
def world_map(stats, metric, label, fmt, selected, t):
    scale = [[i / (len(t["seq"]) - 1), c] for i, c in enumerate(t["seq"])]
    fig = go.Figure(go.Choropleth(
        locations=stats["iso3"], z=stats[metric], text=stats["Country"], colorscale=scale,
        marker=dict(line=dict(color=t["bg"], width=0.6)),
        colorbar=dict(title=dict(text=label, font=dict(family=MONO, size=10, color=t["muted"])), thickness=9,
                      len=0.62, x=0.98, outlinewidth=0, tickfont=dict(family=MONO, size=10, color=t["c_tick"])),
        hovertemplate="<b>%{text}</b><br>" + label + " %{z:" + fmt + "}<extra></extra>",
    ))
    sel = stats[stats["Country"] == selected]
    if len(sel):
        fig.add_trace(go.Choropleth(locations=sel["iso3"], z=[1], showscale=False, hoverinfo="skip",
                                    colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
                                    marker=dict(line=dict(color=t["accent"], width=2.4))))
    fig.update_geos(projection_type="natural earth", showframe=False, showcoastlines=False, showland=True,
                    landcolor=t["c_land"], showcountries=True, countrycolor=t["border"], showocean=False,
                    showlakes=False, bgcolor="rgba(0,0,0,0)", lataxis_range=[-56, 84])
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)",
                      hoverlabel=dict(bgcolor=t["surface_2"], bordercolor=t["border_strong"],
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
        textfont=dict(family=MONO, size=10.5, color=t["ink_2"]),
        hovertemplate="%{y}<br>%{x:" + fmt + "}<extra></extra>",
    ))
    style(fig, t, height=30 * len(rows) + 30, hovermode="closest", margin=dict(l=4, r=48, t=4, b=4))
    fig.update_xaxes(showticklabels=False, showline=False)
    fig.update_yaxes(showgrid=False, tickfont=dict(family=SANS, size=12, color=t["ink_2"]))
    fig.update_layout(bargap=0.3)
    return fig


def peer_scatter(stats, selected, t):
    ref = 2.0 * stats["Production"].max() / (40 ** 2)
    fig = go.Figure()
    for is_sel, group in stats.groupby(stats["Country"] == selected):
        fig.add_scatter(
            x=group["Yield"], y=group["FSI"], mode="markers+text" if is_sel else "markers",
            name=selected if is_sel else "Other countries",
            text=group["Country"] if is_sel else None, textposition="top center",
            textfont=dict(family=SANS, size=12, color=t["ink"]),
            customdata=np.stack([group["Country"], group["Region"], group["Production"]], axis=-1),
            marker=dict(size=group["Production"], sizemode="area", sizeref=ref, sizemin=6,
                        color=t["c_a"] if is_sel else rgba(t["c_peer"], 0.75),
                        line=dict(color=t["accent"] if is_sel else t["surface"], width=2 if is_sel else 1)),
            hovertemplate="<b>%{customdata[0]}</b> · %{customdata[1]}<br>Yield %{x:.2f} MT/ha · FSI %{y:.0f}"
                          "<br>Production %{customdata[2]:,.0f} MT<extra></extra>",
        )
    fig.add_vline(x=stats["Yield"].median(), line=dict(color=t["c_axis"], width=1))
    fig.add_hline(y=stats["FSI"].median(), line=dict(color=t["c_axis"], width=1))
    for x, y, xa, ya, text in [(0.99, 0.99, "right", "top", "HIGH YIELD · SECURE"), (0.01, 0.01, "left", "bottom", "LOW YIELD · AT RISK")]:
        fig.add_annotation(xref="paper", yref="paper", x=x, y=y, xanchor=xa, yanchor=ya, text=text, showarrow=False,
                           font=dict(family=MONO, size=9.5, color=t["subtle"]))
    style(fig, t, height=380, legend=True, hovermode="closest")
    fig.update_xaxes(title_text="Average yield (MT/ha)")
    fig.update_yaxes(title_text="Average FSI")
    fig.update_layout(clickmode="event+select")
    return fig


def gapminder(crop_df, selected, t):
    d = crop_df[["Country", "Year", "Rainfall", "Yield", "Production", "region"]].sort_values("Year").copy()
    d["Group"] = np.where(d["Country"] == selected, selected, "Other countries")
    pad = lambda s, f=0.06: [s.min() - (s.max() - s.min()) * f, s.max() + (s.max() - s.min()) * f]
    fig = px.scatter(
        d, x="Rainfall", y="Yield", size="Production", color="Group", animation_frame="Year",
        animation_group="Country", hover_name="Country", size_max=42, template="none",
        range_x=pad(d["Rainfall"]), range_y=pad(d["Yield"], 0.12),
        color_discrete_map={selected: t["c_a"], "Other countries": rgba(t["c_peer"], 0.8)},
        category_orders={"Group": ["Other countries", selected]},
        hover_data={"Group": False, "Year": False, "region": True, "Production": ":,.0f",
                    "Rainfall": ":,.0f", "Yield": ":.2f"},
    )
    fig.update_traces(marker=dict(line=dict(color=t["surface"], width=1)))
    style(fig, t, height=500, legend=True, hovermode="closest", margin=dict(l=4, r=10, t=34, b=70))
    fig.update_layout(legend_title_text="")
    fig.update_xaxes(title_text="Annual rainfall (mm)")
    fig.update_yaxes(title_text="Yield (MT/ha)")

    def watermark(year):
        return [dict(text=str(year), xref="paper", yref="paper", x=0.98, y=0.04, xanchor="right", yanchor="bottom",
                     showarrow=False, font=dict(family=DISPLAY, size=72, color=rgba(t["ink"], 0.07)))]

    for frame in fig.frames:
        frame.layout = go.Layout(annotations=watermark(frame.name))
    if fig.frames:
        fig.update_layout(annotations=watermark(fig.frames[0].name))
    button_style = dict(bgcolor=t["surface_2"], bordercolor=t["border"], borderwidth=1,
                        font=dict(family=MONO, size=11, color=t["ink"]))
    fig.update_layout(
        updatemenus=[dict(
            type="buttons", direction="left", showactive=False, x=0, xanchor="left", y=-0.2, yanchor="top",
            pad=dict(r=8, t=0), **button_style,
            buttons=[dict(label="▶  Play", method="animate",
                          args=[None, dict(frame=dict(duration=900, redraw=False), fromcurrent=True,
                                           transition=dict(duration=650, easing="cubic-in-out"))]),
                     dict(label="❚❚  Pause", method="animate",
                          args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate",
                                             transition=dict(duration=0))])],
        )],
    )
    if fig.layout.sliders:
        slider = fig.layout.sliders[0]
        slider.update(x=0.2, len=0.8, y=-0.16, pad=dict(t=0, b=0), bgcolor=t["surface_2"],
                      activebgcolor=t["accent"], bordercolor=t["border"], borderwidth=1, tickcolor=t["c_axis"],
                      font=dict(family=MONO, size=10, color=t["c_tick"]),
                      currentvalue=dict(visible=False),
                      steps=[dict(label=s.label, method=s.method,
                                  args=[s.args[0], {**s.args[1], "transition": dict(duration=500, easing="cubic-in-out")}])
                             for s in slider.steps])
    return fig


def radar(profiles, t):
    """profiles: list of (name, {label: percentile}, color)."""
    labels = list(profiles[0][1].keys())
    theta = labels + labels[:1]
    fig = go.Figure()
    fig.add_scatterpolar(r=[50] * len(theta), theta=theta, name="Global median", mode="lines",
                         line=dict(color=t["c_trend"], width=1.2, dash="dot"), hoverinfo="skip")
    for name, values, color in profiles:
        r = [values[k] for k in labels]
        fig.add_scatterpolar(r=r + r[:1], theta=theta, name=name, fill="toself", fillcolor=rgba(color, 0.16),
                             line=dict(color=color, width=2), marker=dict(size=6, color=color),
                             hovertemplate="%{theta}: %{r:.0f}th percentile<extra>" + name + "</extra>")
    fig.update_layout(
        height=370, margin=dict(l=80, r=80, t=46, b=30), paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=SANS, color=t["ink_2"]), showlegend=True,
        legend=dict(orientation="h", x=0, y=1.08, font=dict(family=SANS, size=11.5, color=t["ink_2"]),
                    bgcolor="rgba(0,0,0,0)", itemclick=False, itemdoubleclick=False),
        hoverlabel=dict(bgcolor=t["surface_2"], bordercolor=t["border_strong"],
                        font=dict(family=SANS, size=12, color=t["ink"])),
        polar=dict(
            bgcolor="rgba(0,0,0,0)",
            radialaxis=dict(range=[0, 100], tickvals=[25, 50, 75, 100], showline=False, gridcolor=t["c_grid"],
                            tickfont=dict(family=MONO, size=9, color=t["c_tick"]), angle=90, tickangle=90),
            angularaxis=dict(gridcolor=t["c_grid"], linecolor=t["c_axis"],
                             tickfont=dict(family=SANS, size=11.5, color=t["ink_2"])),
        ),
        transition=dict(duration=550, easing="cubic-in-out"),
    )
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
        xgap=3, ygap=3, hoverongaps=False, colorscale=[[0, t["c_red"]], [0.5, t["c_mid"]], [1, t["c_b"]]],
        hovertemplate="<b>%{y}</b> · %{x}<br>Yield %{customdata:.2f} MT/ha<br>%{z:.0f}% of its average<extra></extra>",
        colorbar=dict(title=dict(text="% of avg", font=dict(family=MONO, size=10, color=t["muted"])), thickness=8,
                      len=0.8, outlinewidth=0, tickfont=dict(family=MONO, size=10, color=t["c_tick"])),
    ))
    style(fig, t, height=max(260, 28 * len(crops) + 60), hovermode="closest", years=index.columns)
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
        textfont=dict(family=MONO, size=10.5, color=t["ink_2"]),
        hovertemplate="%{y}: %{x:+.1f}% per year<extra></extra>",
    ))
    style(fig, t, height=max(260, 28 * len(trends) + 40), hovermode="closest", zero_line=True,
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
        marker=dict(line=dict(color=t["surface"], width=2), cornerradius=6), root_color="rgba(0,0,0,0)",
        texttemplate="<b>%{label}</b><br>%{value:,.0f} MT", textfont=dict(family=SANS, size=13),
        hovertemplate="<b>%{label}</b><br>Avg production %{value:,.0f} MT<br>Avg margin %{color:.1f}%<extra></extra>",
        pathbar_visible=False,
    )
    fig.update_layout(
        height=380, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", font=dict(family=SANS),
        coloraxis_colorbar=dict(title=dict(text="Margin %", font=dict(family=MONO, size=10, color=t["muted"])),
                                thickness=9, len=0.7, outlinewidth=0,
                                tickfont=dict(family=MONO, size=10, color=t["c_tick"])),
        hoverlabel=dict(bgcolor=t["surface_2"], bordercolor=t["border_strong"],
                        font=dict(family=SANS, size=12, color=t["ink"])),
    )
    return fig
