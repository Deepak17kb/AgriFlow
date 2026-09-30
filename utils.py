from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).with_name("data.csv")

# ISO-3 codes for the map; names alone are ambiguous ("USA", "North Korea").
ISO3 = {
    "Afghanistan": "AFG", "Algeria": "DZA", "Argentina": "ARG", "Australia": "AUS", "Bangladesh": "BGD",
    "Bolivia": "BOL", "Brazil": "BRA", "Bulgaria": "BGR", "Burkina Faso": "BFA", "Cambodia": "KHM",
    "Canada": "CAN", "Chile": "CHL", "China": "CHN", "Colombia": "COL", "Ecuador": "ECU", "Egypt": "EGY",
    "Ethiopia": "ETH", "France": "FRA", "Germany": "DEU", "Ghana": "GHA", "Hungary": "HUN", "India": "IND",
    "Indonesia": "IDN", "Iran": "IRN", "Iraq": "IRQ", "Italy": "ITA", "Japan": "JPN", "Kazakhstan": "KAZ",
    "Kenya": "KEN", "Kyrgyzstan": "KGZ", "Laos": "LAO", "Malawi": "MWI", "Malaysia": "MYS", "Mali": "MLI",
    "Mexico": "MEX", "Mongolia": "MNG", "Morocco": "MAR", "Mozambique": "MOZ", "Myanmar": "MMR",
    "Nepal": "NPL", "Netherlands": "NLD", "New Zealand": "NZL", "Nigeria": "NGA", "North Korea": "PRK",
    "Pakistan": "PAK", "Papua New Guinea": "PNG", "Paraguay": "PRY", "Peru": "PER", "Philippines": "PHL",
    "Poland": "POL", "Romania": "ROU", "Saudi Arabia": "SAU", "Senegal": "SEN", "South Korea": "KOR",
    "Spain": "ESP", "Sri Lanka": "LKA", "Tajikistan": "TJK", "Tanzania": "TZA", "Thailand": "THA",
    "Tunisia": "TUN", "Turkey": "TUR", "USA": "USA", "Uganda": "UGA", "Ukraine": "UKR",
    "Uzbekistan": "UZB", "Venezuela": "VEN", "Vietnam": "VNM", "Zambia": "ZMB",
}

def load_data():
    df = pd.read_csv(DATA_PATH)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]

    # ---------- COUNTRY ----------
    for col in ["country", "area", "nation"]:
        if col in df.columns:
            df["Country"] = df[col]
            break
    else:
        df["Country"] = "Unknown"

    # ---------- CROP ----------
    for col in ["crop", "item", "commodity"]:
        if col in df.columns:
            df["Crop"] = df[col]
            break
    else:
        df["Crop"] = "General"

    # ---------- YEAR ----------
    for col in ["year", "yr", "y"]:
        if col in df.columns:
            df["Year"] = df[col]
            break
    else:
        df["Year"] = range(len(df))

    # ---------- YIELD ----------
    for col in ["yield_mt_ha", "yield", "yld", "value"]:
        if col in df.columns:
            df["Yield"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["Yield"] = 0

    # ---------- PRODUCTION ----------
    for col in ["production_mt", "production", "prod"]:
        if col in df.columns:
            df["Production"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["Production"] = df["Yield"]

    # ---------- RAINFALL ----------
    for col in ["annual_rainfall_mm", "rainfall_mm", "rainfall", "rain"]:
        if col in df.columns:
            df["Rainfall"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["Rainfall"] = 0

    # ---------- TEMPERATURE ----------
    for col in ["avg_temp_c", "temperature_c", "temperature", "temp"]:
        if col in df.columns:
            df["Temperature"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["Temperature"] = 0

    # ---------- FSI SCORE ----------
    for col in ["fsi_score", "fsi", "food_security_index"]:
        if col in df.columns:
            df["FSI"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["FSI"] = df["Yield"]

    # ---------- PROFIT MARGIN ----------
    for col in ["profit_margin_pct", "profit_margin", "margin"]:
        if col in df.columns:
            df["ProfitMargin"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["ProfitMargin"] = 0

    # ---------- LOSS RATE ----------
    for col in ["loss_rate_pct", "loss_rate", "loss"]:
        if col in df.columns:
            df["LossRate"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["LossRate"] = 0

    # ---------- MECHANIZATION ----------
    for col in ["mechanization_pct", "mechanization"]:
        if col in df.columns:
            df["Mechanization"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["Mechanization"] = 0

    # ---------- IRRIGATION ----------
    for col in ["irrigation_pct", "irrigation"]:
        if col in df.columns:
            df["Irrigation"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["Irrigation"] = 0

    # ---------- SOIL HEALTH ----------
    for col in ["soil_health_score", "soil_health"]:
        if col in df.columns:
            df["SoilHealth"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["SoilHealth"] = 0

    # ---------- PRICE ----------
    for col in ["price_usd_per_mt", "price_usd", "price"]:
        if col in df.columns:
            df["Price"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["Price"] = 0

    # ---------- TRADE BALANCE ----------
    for col in ["trade_balance_usd_m", "trade_balance"]:
        if col in df.columns:
            df["TradeBalance"] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            break
    else:
        df["TradeBalance"] = 0

    # ---------- DROUGHT RISK ----------
    for col in ["drought_risk", "drought"]:
        if col in df.columns:
            df["DroughtRisk"] = df[col].astype(str)
            break
    else:
        df["DroughtRisk"] = "Unknown"

    # ---------- FS STATUS ----------
    for col in ["fs_status", "food_security_status"]:
        if col in df.columns:
            df["FSStatus"] = df[col].astype(str)
            break
    else:
        df["FSStatus"] = "Unknown"

    return df