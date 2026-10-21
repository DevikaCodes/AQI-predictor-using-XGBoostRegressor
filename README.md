AI Air Guard: 3D Geospatial Air Quality Predictor
AI Air Guard is an end-to-end machine learning and geospatial visualization platform designed to predict and analyze real-time Air Quality Index (AQI) levels across global and regional locations. The system integrates real-time atmospheric telemetry with an optimized gradient-boosted regression pipeline, presenting telemetry across an interactive 3D WebGL globe and a detailed analytical reporting interface.

Key Features
3D Geospatial Globe Dashboard (index.html): Interactive WebGL Earth visualization built with Three.js and Globe.gl. Displays localized air quality through 3D hexagonal pillars (height and color indicate severity) alongside dynamic radar warning pulses for sensitive or hazardous zones.

XGBoost Inference Engine: Predicts continuous AQI scores using an 8-feature machine learning pipeline (PM2.5, PM10, NO2, SO2, CO, O3, NH3, and cyclical temporal encodings).

Automated Telemetry Retrieval: Direct geocoding and live pollutant data ingestion powered by OpenWeatherMap APIs.

Health-Standard Dominant Pollutant Identification: Evaluates atmospheric readings using standard Indian Central Pollution Control Board (CPCB) / National Ambient Air Quality Standards (NAAQS) sub-index breakpoints rather than raw mass comparisons.

Dedicated Analytics Report (details.html): Automatically routes queries to an analytics view containing categorical alert badges, primary pollutant attribution, and an interactive Chart.js sub-index breakdown chart.

Technology Stack
Backend: FastAPI, Uvicorn, Requests

Machine Learning & Data Processing: XGBoost, Scikit-Learn, Pandas, NumPy, Joblib

Frontend Visualization: JavaScript (ES6+), Three.js, Globe.gl, Chart.js, HTML5/CSS3

Project Structure
Plaintext


aqi_project/
├── aqi_model.pkl          # Pre-trained XGBoost Regressor model
├── aqi_scaler.pkl         # Fitted StandardScaler artifact
├── aqi_ncols.pkl          # Ordered training feature schema
├── aqi_col_ranges.pkl     # Feature clipping boundary ranges
├── app.py                 # FastAPI backend, CPCB engine & routing
├── index.html             # 3D geospatial globe & search portal
└── details.html           # Detailed city AQI & pollutant visualizer
Installation & Setup
1. Clone the Repository & Prepare Environment
Bash


git clone <repository-url>
cd <repository-directory>
pip install fastapi uvicorn requests pandas numpy scikit-learn xgboost joblib
2. Configure Environment Variables (Optional)
If you have a dedicated OpenWeather API key, set it as an environment variable:

Bash


export OPENWEATHER_API_KEY="your_api_key_here"
3. Launch the Application
Run the FastAPI application server:

Bash


python app.py
Alternatively, launch via Uvicorn:

Bash


uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Usage
Open your browser and navigate to http://localhost:8000 (or your cloud environment's forwarded port 8000 URL).

Explore pre-mapped regional hubs using the interactive 3D pillars or enter a target city (e.g., Kochi, Delhi, London) in the search interface.

Review the predicted AQI, alert level classification, primary driving pollutant, and relative sub-index impact chart.
