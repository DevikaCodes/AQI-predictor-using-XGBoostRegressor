"""
AQI Prediction Backend (Flask)
------------------------------
Loads the saved model artifacts (model, scaler, col_ranges, ncols) and
exposes a single endpoint that takes a city name and returns the
predicted AQI, using live pollution data from OpenWeatherMap.

Run with:
    pip install flask flask-cors joblib pandas numpy requests xgboost scikit-learn
    python app.py
"""

import joblib
import numpy as np
import pandas as pd
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # allow requests from your HTML frontend (different origin/port)

# ---------------------------------------------------------------------------
# Load artifacts once at startup
# ---------------------------------------------------------------------------
model = joblib.load('aqi_model.pkl')
scaler = joblib.load('aqi_scaler.pkl')
col_ranges = joblib.load('aqi_col_ranges.pkl')
ncols = joblib.load('aqi_ncols.pkl')

API_KEY = '6200ecd2ff47f6ad62e15613c166f8df'


def clip_to_range(value, col):
    lo, hi = col_ranges[col]
    return np.clip(value, lo, hi)


def predict_aqi_for_city(city_name, country_code="IN"):
    # 1. Geocode the city name
    geo_url = (
        f"https://api.openweathermap.org/geo/1.0/direct"
        f"?q={city_name},{country_code}&limit=1&appid={API_KEY}"
    )
    geo_data = requests.get(geo_url).json()

    if not geo_data:
        return {"error": f"Location '{city_name}' not found"}

    lat = geo_data[0]['lat']
    lon = geo_data[0]['lon']

    # 2. Fetch live pollution data
    pollution_url = (
        f"https://api.openweathermap.org/data/2.5/air_pollution"
        f"?lat={lat}&lon={lon}&appid={API_KEY}"
    )
    pollution_data = requests.get(pollution_url).json()

    if "list" not in pollution_data or not pollution_data["list"]:
        return {"error": f"No pollution data available for '{city_name}'"}

    components = pollution_data["list"][0]["components"]
    dt_unix = pollution_data["list"][0]["dt"]
    reading_date = pd.to_datetime(dt_unix, unit='s')
    month = reading_date.month
    m_sin = np.sin(2 * np.pi * month / 12)
    m_cos = np.cos(2 * np.pi * month / 12)

    # 3. Build feature row (order must match ncols)
    sample_df = pd.DataFrame([{
        'PM2.5': clip_to_range(components.get('pm2_5', 0), 'PM2.5'),
        'PM10':  clip_to_range(components.get('pm10', 0), 'PM10'),
        'NO2':   clip_to_range(components.get('no2', 0), 'NO2'),
        'CO':    clip_to_range(components.get('co', 0) / 1000, 'CO'),  # µg/m³ -> mg/m³
        'O3':    clip_to_range(components.get('o3', 0), 'O3'),
        'NH3':   clip_to_range(components.get('nh3', 0), 'NH3'),
        'month_sin': m_sin,
        'month_cos': m_cos,
    }], columns=ncols)

    # 4. Scale + predict
    sample_scaled = scaler.transform(sample_df)
    predicted_aqi = float(model.predict(sample_scaled)[0])

    return {
        "city": city_name,
        "predicted_aqi": round(predicted_aqi, 2),
        "components": components,
        "reading_time": str(reading_date),
    }


@app.route('/predict', methods=['GET'])
def predict():
    city_name = request.args.get('city', '').strip()

    if not city_name:
        return jsonify({"error": "Please provide a city name, e.g. /predict?city=kochi"}), 400

    result = predict_aqi_for_city(city_name)

    if "error" in result:
        return jsonify(result), 404

    return jsonify(result)


@app.route('/', methods=['GET'])
def health_check():
    return jsonify({"status": "AQI prediction backend is running"})


if __name__ == '__main__':
    app.run(debug=True, port=5000)