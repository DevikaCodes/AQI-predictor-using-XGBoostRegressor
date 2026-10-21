import os
import joblib
import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="AI Air Guard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 1. Load model artifacts
print("Loading model weights...")
model = joblib.load("aqi_model.pkl")
scaler = joblib.load("aqi_scaler.pkl")
ncols = joblib.load("aqi_ncols.pkl")
col_ranges = joblib.load("aqi_col_ranges.pkl")
print("Weights loaded successfully!")

API_KEY = os.getenv("OPENWEATHER_API_KEY", "6200ecd2ff47f6ad62e15613c166f8df")

def get_aqi_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Sensitive"
    elif aqi <= 200:
        return "Unhealthy"
    else:
        return "Hazardous"

def calculate_sub_index(conc: float, pollutant: str) -> float:
    """
    Computes standard CPCB AQI sub-index for each pollutant based on Indian NAAQS breakpoints.
    """
    breakpoints = {
        "PM2.5": [
            (0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
            (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)
        ],
        "PM10": [
            (0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
            (250, 350, 201, 300), (350, 430, 301, 400), (430, 500, 401, 500)
        ],
        "NO2": [
            (0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
            (180, 280, 201, 300), (280, 400, 301, 400), (400, 1000, 401, 500)
        ],
        "SO2": [
            (0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
            (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2000, 401, 500)
        ],
        "CO": [  # in mg/m³
            (0, 1.0, 0, 50), (1.0, 2.0, 51, 100), (2.0, 10.0, 101, 200),
            (10.0, 17.0, 201, 300), (17.0, 34.0, 301, 400), (34.0, 50.0, 401, 500)
        ],
        "O3": [
            (0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
            (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)
        ],
        "NH3": [
            (0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200),
            (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 2500, 401, 500)
        ]
    }

    if pollutant not in breakpoints:
        return 0.0

    for c_low, c_high, i_low, i_high in breakpoints[pollutant]:
        if c_low <= conc <= c_high:
            return round(((i_high - i_low) / (c_high - c_low)) * (conc - c_low) + i_low, 1)

    return 500.0 if conc > 0 else 0.0

@app.get("/api/predict")
def predict_city(city: str, country: str = "IN"):
    # Step A: Geocoding
    geo_url = f"https://api.openweathermap.org/geo/1.0/direct?q={city},{country}&limit=1&appid={API_KEY}"
    geo_res = requests.get(geo_url).json()

    if not geo_res:
        raise HTTPException(status_code=404, detail=f"City '{city}' not found.")

    lat, lon = geo_res[0]["lat"], geo_res[0]["lon"]
    display_name = f"{geo_res[0].get('name', city.title())}, {geo_res[0].get('country', country.upper())}"

    # Step B: OpenWeatherMap Air Pollution API
    poll_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"
    poll_res = requests.get(poll_url).json()

    if "list" not in poll_res or not poll_res["list"]:
        raise HTTPException(status_code=502, detail="No pollution telemetry returned.")

    record = poll_res["list"][0]
    components = record["components"]

    # Step C: Extract cyclical temporal features
    reading_date = pd.to_datetime(record["dt"], unit="s")
    month = reading_date.month
    m_sin = np.sin(2 * np.pi * month / 12)
    m_cos = np.cos(2 * np.pi * month / 12)

    def clip_val(val, col):
        col_min, col_max = col_ranges[col]
        return np.clip(val, col_min, col_max)

    # Step D: Match exact 8 features
    raw_features = {
        "PM2.5": clip_val(components.get("pm2_5", 0), "PM2.5"),
        "PM10":  clip_val(components.get("pm10", 0), "PM10"),
        "NO2":   clip_val(components.get("no2", 0), "NO2"),
        "CO":    clip_val(components.get("co", 0) / 1000.0, "CO"),  # µg/m³ to mg/m³
        "O3":    clip_val(components.get("o3", 0), "O3"),
        "NH3":   clip_val(components.get("nh3", 0), "NH3"),
        "month_sin": m_sin,
        "month_cos": m_cos,
    }

    input_df = pd.DataFrame([raw_features], columns=ncols)
    input_scaled = scaler.transform(input_df)
    predicted_aqi = float(model.predict(input_scaled)[0])

    # Step E: Sub-Index & Pollutant Metrics
    pollutant_readings = {
        "PM2.5": round(float(components.get("pm2_5", 0)), 2),
        "PM10": round(float(components.get("pm10", 0)), 2),
        "NO2": round(float(components.get("no2", 0)), 2),
        "SO2": round(float(components.get("so2", 0)), 2),
        "CO": round(float(components.get("co", 0) / 1000.0), 3),
        "O3": round(float(components.get("o3", 0)), 2),
        "NH3": round(float(components.get("nh3", 0)), 2)
    }

    sub_indices = {
        p: calculate_sub_index(val, p) for p, val in pollutant_readings.items()
    }

    dominant_pollutant = max(sub_indices, key=sub_indices.get)
    unit = "mg/m³" if dominant_pollutant == "CO" else "µg/m³"
    dom_val_str = f"{pollutant_readings[dominant_pollutant]} {unit} (Sub-Index: {sub_indices[dominant_pollutant]})"

    return {
        "name": display_name,
        "lat": lat,
        "lng": lon,
        "aqi": round(predicted_aqi, 1),
        "status": get_aqi_category(predicted_aqi),
        "dominant_pollutant": dominant_pollutant,
        "dominant_value": dom_val_str,
        "components": pollutant_readings,
        "sub_indices": sub_indices
    }

app.mount("/", StaticFiles(directory=".", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)