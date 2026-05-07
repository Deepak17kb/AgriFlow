<!-- ╔══════════════════════════════════════════════════════════════════════╗ -->
<!-- ║                     AGRIFLOW AI  —  README                          ║ -->
<!-- ╚══════════════════════════════════════════════════════════════════════╝ -->

<div align="center">

<!-- ANIMATED HEADER BANNER — vortex style, clean text only -->
<img src="https://capsule-render.vercel.app/api?type=venom&color=0:0d2b1a,30:1a5c38,70:2e8b57,100:4caf7d&height=260&section=header&text=AgriFlow%20AI&fontSize=72&fontColor=ffffff&fontAlignY=40&desc=Agricultural%20Intelligence%20%7C%20Food%20Security%20Forecasting&descAlignY=60&descSize=19&animation=scaleIn&stroke=2e8b57&strokeWidth=2" width="100%"/>

<br/>

<!-- ANIMATED TYPING HEADLINE -->
<img src="https://readme-typing-svg.demolab.com?font=JetBrains+Mono&weight=700&size=20&duration=2800&pause=700&color=4CAF7D&center=true&vCenter=true&width=750&lines=Predict+Food+Security+Index+with+Machine+Learning;Visualize+Yield+%26+Climate+Trends+Interactively;RandomForest-Powered+FSI+Forecasting;10%2C500%2B+Records+·+Global+Coverage+2013-2024;Built+with+Streamlit+·+Plotly+·+scikit-learn" alt="Typing Animation"/>

<br/><br/>

<!-- TECH STACK BADGES -->
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Wrangling-150458?style=for-the-badge&logo=pandas&logoColor=white)

<br/>

<!-- STATUS BADGES -->
![Status](https://img.shields.io/badge/Status-Live-1b5e20?style=for-the-badge&labelColor=0d2b1a)
![License](https://img.shields.io/badge/License-MIT-2e8b57?style=for-the-badge&labelColor=0d2b1a)
![Dataset](https://img.shields.io/badge/Dataset-10%2C500%2B%20Records-4caf50?style=for-the-badge&labelColor=0d2b1a)
![Coverage](https://img.shields.io/badge/Coverage-Global-00796b?style=for-the-badge&labelColor=0d2b1a)
![Years](https://img.shields.io/badge/Years-2013--2024-388e3c?style=for-the-badge&labelColor=0d2b1a)

<br/><br/>

<!-- LIVE DEMO BUTTON -->
<a href="https://agriflowai.streamlit.app/">
  <img src="https://img.shields.io/badge/%20%20LIVE%20DEMO%20%E2%86%92%20agriflowai.streamlit.app%20%20-2e8b57?style=for-the-badge&logoColor=white&labelColor=1a5c38" height="46"/>
</a>

</div>

---

## What is AgriFlow AI?

> **AgriFlow AI** is an end-to-end agricultural intelligence platform that ingests historical crop, climate, and economic data — then uses a **Random Forest classifier** to forecast whether a country's **Food Security Index (FSI)** will improve or decline in the next period.

Built for agronomists, policy analysts, and food-security researchers who need **actionable insights at a glance** — the dashboard combines rich interactive charts, smart alert systems, and AI-driven recommendations in a polished Streamlit UI.

---

## Animated Module Coverage

| Module | Completeness | Role |
|---|---|---|
| `app.py` | ![](https://geps.dev/progress/95?dangerColor=1a5c38&warningColor=2e8b57&successColor=4caf7d) | UI · Charts · Alert Engine |
| `model.py` | ![](https://geps.dev/progress/80?dangerColor=1a5c38&warningColor=2e8b57&successColor=4caf7d) | Feature Engineering · RandomForest |
| `utils.py` | ![](https://geps.dev/progress/100?dangerColor=1a5c38&warningColor=2e8b57&successColor=4caf7d) | Data Loading · Column Normalization |
| `data.csv` | ![](https://geps.dev/progress/100?dangerColor=1a5c38&warningColor=2e8b57&successColor=4caf7d) | 10,500+ Records · 2013-2024 |

---

## Feature Highlights

<div align="center">

| Feature | Description |
|---|---|
| **ML Forecasting** | RandomForest predicts FSI direction with calibrated confidence score |
| **Yield Analytics** | Multi-year trend lines with automatic peak and trough detection |
| **Climate Overlay** | Rainfall and temperature overlaid directly on production charts |
| **Smart Alerts** | Auto-generated risk alerts — drought, loss rate, FSI decline |
| **Trade & Economics** | Trade balance bars, price trend line, profit margin gauge |
| **AI Insights Panel** | Six key derived metrics per country / crop selection |
| **Global Dataset** | 10,500+ records spanning Asia, Africa, Americas, and Oceania |
| **Responsive Layout** | `wide` layout with custom CSS — desktop and tablet ready |

</div>

---

## Project Structure

```
agriflow-ai/
│
├── app.py            ← Streamlit entry point · UI, charts, alert engine
├── model.py          ← Feature engineering + RandomForest training & predict
├── utils.py          ← CSV loader with robust column-name normalization
├── data.csv          ← 10,500+ row global agricultural dataset (2013-2024)
└── requirements.txt  ← Python dependencies
```

---

## Quick Start

**1 — Clone the repo**

```bash
git clone https://github.com/your-username/agriflow-ai.git
cd agriflow-ai
```

**2 — Install dependencies**

```bash
pip install -r requirements.txt
```

**3 — Run the app**

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`

---

## Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
```

---

## How the ML Model Works

```
┌────────────────────────────────────────────────────────────────┐
│                       MODEL PIPELINE                           │
│                                                                │
│   data.csv  ──►  utils.load_data()                            │
│                      │  column name normalization              │
│                      ▼                                         │
│               model.prepare_features()                         │
│                      │  target = FSI(t+1) > FSI(t)  →  0 / 1 │
│                      ▼                                         │
│               model.train_model()                              │
│                      │  RandomForestClassifier(n_estimators=100)│
│                      ▼                                         │
│               model.predict()  ──►  prediction + probability   │
└────────────────────────────────────────────────────────────────┘
```

**Input features:**

| Feature | Description |
|---|---|
| `Yield` | Crop yield in MT/ha |
| `Production` | Total production volume in MT |
| `Rainfall` | Annual rainfall in mm |
| `Temperature` | Average temperature in degrees C |

**Target:** `1` = FSI improves · `0` = FSI declines or flat

---

## Dataset Overview

| Attribute | Detail |
|---|---|
| **Rows** | 10,500+ records |
| **Years** | 2013 – 2024 |
| **Countries** | India, Bangladesh, Australia, New Zealand, and more |
| **Crops** | Wheat, Rice, Maize, Sugarcane, Vegetables, Cotton, Tea, Coffee |
| **Key metrics** | FSI score, Yield, Rainfall, Temp, Trade Balance, Profit Margin, Soil Health, Irrigation, Loss Rate |

---

## Dashboard Layout

```
┌──────────────────────────────────────────────────────────────┐
│  SIDEBAR                 │  MAIN PANEL                       │
│  ────────────────────    │  ───────────────────────────────  │
│  Country selector        │  FSI Overview Card                │
│  Crop selector           │  3-step Forecast (up / flat / dn) │
│  Year range slider       │  Yield Trend Chart                │
│  Run Forecast button     │  Rainfall & Temp Overlay          │
│                          │  Supply Chain Metrics             │
│                          │  Smart Alerts Panel               │
│                          │  AI Insights Panel                │
│                          │  Trade & Economics Charts         │
└──────────────────────────────────────────────────────────────┘
```

---

## Smart Alert System

The alert engine automatically flags:

- **High Drought Risk** — DroughtRisk = Critical
- **FSI Declining** — FSI dropped more than 5 pts
- **High Loss Rate** — Post-harvest loss above 15%
- **Elevated Loss Rate** — Post-harvest loss above 8%
- **Positive Forecast** — Model confidence above 65% for FSI improvement

---

## Configuration & Customization

**Change model features** — edit `model.py :: train_model()`:

```python
features = ["Yield", "Production", "Rainfall", "Temperature"]
# Extend with: "SoilHealth", "Irrigation", "Mechanization"
```

**Add new data columns** — `utils.py` normalizes column names automatically.
Match the expected aliases inside each block of `load_data()`.

**Change theme colors** — CSS variables live at the top of the `<style>` block in `app.py`.
Primary green: `#2e8b57`

---

## Live Demo

<div align="center">

[![Open App](https://img.shields.io/badge/Open%20AgriFlow%20AI%20%E2%86%92%20agriflowai.streamlit.app-2e8b57?style=for-the-badge&labelColor=1a5c38)](https://agriflowai.streamlit.app/)

</div>

---

## License

Licensed under the **MIT License** — free to fork, extend, and deploy.

---

<!-- ANIMATED CONTRIBUTION SNAKE -->
<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)"  srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake-dark.svg"/>
  <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg"/>
  <img alt="github contribution grid snake animation" src="https://raw.githubusercontent.com/platane/snk/output/github-contribution-grid-snake.svg" width="100%"/>
</picture>

</div>

<!-- ANIMATED FOOTER WAVE -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4caf7d,50:2e8b57,100:0d2b1a&height=130&section=footer&animation=twinkling" width="100%"/>

<sub>Powered by Streamlit · scikit-learn · Plotly — AgriFlow AI</sub>

</div>
