<!-- ╔══════════════════════════════════════════════════════════════════════╗ -->
<!-- ║                     AGRIFLOW AI  ·  README                          ║ -->
<!-- ╚══════════════════════════════════════════════════════════════════════╝ -->

<div align="center">

<!-- ANIMATED BANNER -->
<img src="https://capsule-render.vercel.app/api?type=waving&color=0:1a5c38,50:2e8b57,100:4caf7d&height=220&section=header&text=🌿%20AgriFlow%20AI&fontSize=62&fontColor=ffffff&fontAlignY=38&desc=Agricultural%20Intelligence%20%7C%20Food%20Security%20Forecasting&descAlignY=58&descSize=18&animation=fadeIn" width="100%"/>

<!-- BADGES ROW 1 -->
<br/>

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Scikit--Learn](https://img.shields.io/badge/scikit--learn-RandomForest-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)

<!-- BADGES ROW 2 -->

![License](https://img.shields.io/badge/License-MIT-2e8b57?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Live%20🟢-1b5e20?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-10%2C500%2B%20Records-4caf50?style=for-the-badge)
![Countries](https://img.shields.io/badge/Countries-Global%20Coverage-00796b?style=for-the-badge)

<br/>

<!-- LIVE DEMO BUTTON -->
<a href="https://agriflowai.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀%20%20LIVE%20DEMO%20%20--%20%20agriflowai.streamlit.app-2e8b57?style=for-the-badge&logoColor=white" height="42"/>
</a>

<br/><br/>

<!-- TYPING ANIMATION  (uses readme-typing-svg) -->
<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=18&duration=3000&pause=800&color=2E8B57&center=true&vCenter=true&multiline=false&width=700&lines=🌾+Predict+Food+Security+Index+with+ML;📊+Visualize+Yield+%26+Climate+Trends;🤖+RandomForest-Powered+Forecasting;🌍+10%2C500%2B+Records+·+Global+Coverage;⚡+Built+with+Streamlit+%2B+Plotly" alt="Typing SVG" />

</div>

---

## 🌱 What is AgriFlow AI?

> **AgriFlow AI** is an end-to-end agricultural intelligence platform that ingests historical crop, climate, and economic data — then uses a **Random Forest classifier** to forecast whether a country's **Food Security Index (FSI)** will improve or decline in the next period.

Built for agronomists, policy analysts, and food-security researchers who need **actionable insights at a glance**, the dashboard combines rich interactive charts, smart alert systems, and AI-driven recommendations — all in a clean Streamlit UI.

---

## ✨ Feature Highlights

<div align="center">

| Feature | Description |
|---|---|
| 🤖 **ML Forecasting** | RandomForest predicts FSI direction with calibrated confidence |
| 📈 **Yield Analytics** | Multi-year yield trend lines with peak/trough detection |
| 🌡️ **Climate Overlay** | Rainfall & temperature overlaid on production charts |
| 🚨 **Smart Alerts** | Auto-generated risk alerts (drought, loss rate, FSI decline) |
| 💹 **Trade & Economics** | Trade balance bars, price trend, profit margin gauge |
| 🔍 **AI Insights Panel** | Six key derived metrics per country/crop selection |
| 🌍 **Global Dataset** | 10,500+ records spanning Asia, Africa, Americas, Oceania |
| 📱 **Responsive Layout** | `wide` layout with custom CSS — works on desktop & tablet |

</div>

---

## 🗂️ Project Structure

```
agriflow-ai/
│
├── 📄 app.py            ← Streamlit entry point · UI, charts, alerts
├── 🧠 model.py          ← Feature engineering + RandomForest training
├── 🛠️  utils.py          ← CSV loader with robust column normalization
├── 📊 data.csv          ← 10,500+ row global agricultural dataset
└── 📦 requirements.txt  ← Python dependencies
```

---

## ⚡ Quick Start

### 1 · Clone the repo

```bash
git clone https://github.com/your-username/agriflow-ai.git
cd agriflow-ai
```

### 2 · Install dependencies

```bash
pip install -r requirements.txt
```

### 3 · Run the app

```bash
streamlit run app.py
```

> 🟢 The app will open at **`http://localhost:8501`**

---

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn
plotly
```

> All installable via `pip install -r requirements.txt`

---

## 🧠 How the ML Model Works

```
┌─────────────────────────────────────────────────────────────┐
│                     MODEL PIPELINE                          │
│                                                             │
│  data.csv  ──►  utils.load_data()                          │
│                    │  column normalization                  │
│                    ▼                                        │
│             model.prepare_features()                        │
│                    │  target = FSI(t+1) > FSI(t)  → 0/1   │
│                    ▼                                        │
│             model.train_model()                             │
│                    │  RandomForestClassifier(n=100)        │
│                    ▼                                        │
│             model.predict()  ──►  pred + probability        │
└─────────────────────────────────────────────────────────────┘
```

**Features used for prediction:**

| Feature | Description |
|---|---|
| `Yield` | Crop yield in MT/ha |
| `Production` | Total production in MT |
| `Rainfall` | Annual rainfall in mm |
| `Temperature` | Average temperature in °C |

**Target variable:**  
`1` → FSI improves next period  
`0` → FSI declines or stays flat

---

## 📊 Dataset Overview

| Attribute | Detail |
|---|---|
| **Rows** | 10,500+ records |
| **Years** | 2013 – 2024 |
| **Countries** | India, Bangladesh, Australia, New Zealand, and more |
| **Crops** | Wheat, Rice, Maize, Sugarcane, Vegetables, Cotton… |
| **Key metrics** | FSI score, Yield, Rainfall, Temp, Trade Balance, Profit Margin, Soil Health, Irrigation, Loss Rate |

---

## 🖥️ Dashboard Sections

```
┌──────────────────────────────────────────────────────────┐
│  SIDEBAR                │  MAIN PANEL                    │
│  ─────────────────────  │  ──────────────────────────── │
│  • Country selector     │  • FSI Overview Card           │
│  • Crop selector        │  • 3-step Forecast (↑ ↔ ↓)   │
│  • Year range           │  • Yield Trend Chart           │
│  • Run Forecast button  │  • Rainfall / Temp Overlay     │
│                         │  • Supply Chain Metrics        │
│                         │  • Smart Alerts Panel          │
│                         │  • AI Insights Panel           │
│                         │  • Trade & Economics Charts    │
└──────────────────────────────────────────────────────────┘
```

---

## 🚨 Smart Alert System

The alerts engine automatically flags conditions like:

- 🔴 **High Drought Risk** — DroughtRisk = Critical
- 🟠 **FSI Declining** — FSI dropped > 5 pts
- 🟡 **High Loss Rate** — Post-harvest loss > 15%
- 🟢 **Positive Forecast** — Model confidence > 65% for improvement

---

## 🔧 Configuration & Customization

**Change model features** → Edit `model.py :: train_model()`:
```python
features = ["Yield", "Production", "Rainfall", "Temperature"]
# Add more: "SoilHealth", "Irrigation", "Mechanization"
```

**Add new columns to data** → The `utils.py` loader normalizes column names automatically. Just match the expected aliases listed in each section of `load_data()`.

**Theme colors** → All CSS variables live at the top of the `<style>` block in `app.py`. Primary green: `#2e8b57`.

---

## 🌐 Live Demo

<div align="center">

[![Open App](https://img.shields.io/badge/🌿%20Open%20AgriFlow%20AI-agriflowai.streamlit.app-2e8b57?style=for-the-badge)](https://agriflowai.streamlit.app/)

</div>

---

## 📄 License

This project is licensed under the **MIT License** — feel free to fork, extend, and deploy.

---

<!-- FOOTER WAVE -->
<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=0:4caf7d,50:2e8b57,100:1a5c38&height=120&section=footer&animation=fadeIn" width="100%"/>

<sub>Built with 🌱 for a food-secure world · Powered by Streamlit · scikit-learn · Plotly</sub>

</div>
