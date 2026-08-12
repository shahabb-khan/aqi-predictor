import streamlit as st
import pandas as pd
import numpy as np
import hopsworks
import joblib
import requests
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")

LAT = 33.6007
LON = 73.0679

st.set_page_config(page_title="Rawalpindi AQI Predictor", page_icon="🌫️", layout="centered")

# --- Custom styling ---
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #4a6fa5 0%, #2c4870 100%);
        padding: 28px 24px;
        border-radius: 14px;
        margin-bottom: 24px;
        text-align: center;
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 32px;
    }
    .main-header p {
        color: #dbe4f0;
        margin: 6px 0 0 0;
        font-size: 15px;
    }
    .aqi-card {
        border-radius: 14px;
        padding: 20px 16px;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        margin-bottom: 10px;
    }
    .aqi-card .label {
        font-size: 14px;
        font-weight: 600;
        opacity: 0.85;
        margin-bottom: 4px;
    }
    .aqi-card .value {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.1;
        margin: 4px 0;
    }
    .aqi-card .category {
        font-size: 13px;
        font-weight: 600;
    }
    .section-title {
        font-size: 20px;
        font-weight: 700;
        margin: 18px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>Rawalpindi AQI Predictor</h1>
    <p>Live air quality data and 3-day AQI forecast</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    mr = project.get_model_registry()

    models = {}
    feature_columns = None

    for day in [1, 2, 3]:
        model_meta = mr.get_model(f"aqi_predictor_day{day}")
        model_dir = model_meta.download()
        models[day] = joblib.load(os.path.join(model_dir, f"aqi_model_day{day}.pkl"))
        if feature_columns is None:
            feature_columns = joblib.load(os.path.join(model_dir, "feature_columns.pkl"))

    return models, feature_columns


def get_current_conditions():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
    response = requests.get(url)
    return response.json()


def calculate_aqi_from_pm25(pm25):
    breakpoints = [
        (0.0, 12.05, 0, 50),
        (12.05, 35.45, 51, 100),
        (35.45, 55.45, 101, 150),
        (55.45, 150.45, 151, 200),
        (150.45, 250.45, 201, 300),
        (250.45, 350.45, 301, 400),
        (350.45, 500.45, 401, 500),
    ]
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
            return round(aqi)
    return 500


def get_aqi_category(aqi):
    # CHANGED: each category now returns (name, emoji, text_color, background_tint)
    # text_color is a darker/readable shade — pure yellow text on white was invisible before.
    if aqi <= 50:
        return "Good", "🟢", "#2e7d32", "#e8f5e9"
    elif aqi <= 100:
        return "Moderate", "🟡", "#b28704", "#fff8e1"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups", "🟠", "#ef6c00", "#fff3e0"
    elif aqi <= 200:
        return "Unhealthy", "🔴", "#c62828", "#ffebee"
    elif aqi <= 300:
        return "Very Unhealthy", "🟣", "#6a1b9a", "#f3e5f5"
    else:
        return "Hazardous", "⚫", "#7e0023", "#fce4ec"


@st.cache_data(ttl=3600)
def get_recent_history():
    end_time = int(datetime.now().timestamp())
    start_time = end_time - (4 * 24 * 60 * 60)
    hist_url = f"http://api.openweathermap.org/data/2.5/air_pollution/history?lat={LAT}&lon={LON}&start={start_time}&end={end_time}&appid={OPENWEATHER_KEY}"
    hist_response = requests.get(hist_url)
    history_data = hist_response.json()
    return history_data.get("list", [])


@st.cache_data(ttl=3600)
def get_weather_trend():
    end_time = int(datetime.now().timestamp())
    start_time = end_time - (4 * 24 * 60 * 60)
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = datetime.fromtimestamp(start_time).strftime("%Y-%m-%d")

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        f"&timezone=UTC"
    )
    response = requests.get(url)
    return response.json()


def build_live_features():
    readings = get_recent_history()
    weather = get_weather_trend()

    df = pd.DataFrame([{
        "timestamp": r["dt"],
        "pm2_5": r["components"]["pm2_5"],
        "pm10": r["components"]["pm10"],
        "co": r["components"]["co"],
        "no2": r["components"]["no2"],
        "o3": r["components"]["o3"],
        "so2": r["components"]["so2"],
    } for r in readings])

    df["aqi"] = df["pm2_5"].apply(calculate_aqi_from_pm25)
    df["event_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    weather_df = pd.DataFrame({
        "event_time": pd.to_datetime(weather["hourly"]["time"], utc=True),
        "temperature": weather["hourly"]["temperature_2m"],
        "humidity": weather["hourly"]["relative_humidity_2m"],
        "wind_speed": weather["hourly"]["wind_speed_10m"],
        "precipitation": weather["hourly"]["precipitation"],
    })
    df = df.merge(weather_df, on="event_time", how="left")
    df = df.dropna()

    df["hour"] = df["event_time"].dt.hour
    df["day_of_week"] = df["event_time"].dt.dayofweek
    df["month"] = df["event_time"].dt.month

    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_lag_72h"] = df["aqi"].shift(72)
    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24, min_periods=1).mean()
    df["aqi_rolling_std_24h"] = df["aqi"].rolling(window=24, min_periods=1).std().fillna(0)
    df["humidity_change_24h"] = df["humidity"].diff(24).fillna(0)
    df["wind_speed_rolling_mean_24h"] = df["wind_speed"].rolling(window=24, min_periods=1).mean()
    df["precipitation_sum_24h"] = df["precipitation"].rolling(window=24, min_periods=1).sum()
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df = df.fillna(method="bfill").fillna(method="ffill")
    return df


def predict_all_days(models, feature_columns, df):
    latest_row = df.iloc[-1:][feature_columns]
    predictions = {}
    for day, model in models.items():
        predictions[day] = round(model.predict(latest_row)[0])
    return predictions


def render_aqi_card(label, aqi_value, category, emoji, text_color, bg_color, big=False):
    value_size = "56px" if big else "42px"
    st.markdown(f"""
    <div class="aqi-card" style="background-color:{bg_color}; border-left: 6px solid {text_color};">
        <div class="label" style="color:{text_color};">{emoji} {label}</div>
        <div class="value" style="color:{text_color}; font-size:{value_size};">{aqi_value}</div>
        <div class="category" style="color:{text_color};">{category}</div>
    </div>
    """, unsafe_allow_html=True)


# --- Load models (cached, only runs once) ---
with st.spinner("Loading models..."):
    models, feature_columns = load_models()

# --- Current conditions ---
current_data = get_current_conditions()
current_pm25 = current_data["list"][0]["components"]["pm2_5"]
current_aqi = calculate_aqi_from_pm25(current_pm25)
current_category, current_emoji, current_text, current_bg = get_aqi_category(current_aqi)

with st.spinner("Building forecast..."):
    live_df = build_live_features()
    forecasts = predict_all_days(models, feature_columns, live_df)

# --- Current AQI (big card) ---
render_aqi_card("Current AQI", current_aqi, current_category, current_emoji, current_text, current_bg, big=True)

# --- 3-day forecast cards ---
st.markdown('<div class="section-title">📅 3-Day Forecast</div>', unsafe_allow_html=True)

day_labels = {1: "Tomorrow", 2: "In 2 Days", 3: "In 3 Days"}
cols = st.columns(3)

for day, col in zip([1, 2, 3], cols):
    aqi_value = forecasts[day]
    category, emoji, text_color, bg_color = get_aqi_category(aqi_value)
    with col:
        render_aqi_card(day_labels[day], aqi_value, category, emoji, text_color, bg_color)

st.markdown("<br>", unsafe_allow_html=True)

# --- Health alert: based on the worst (highest AQI) day in the forecast ---
worst_day = max(forecasts, key=forecasts.get)
worst_aqi = forecasts[worst_day]
worst_category, _, _, _ = get_aqi_category(worst_aqi)

if worst_aqi > 150:
    st.error(f"⚠️ **Health Alert:** AQI is expected to reach '{worst_category}' levels "
              f"({day_labels[worst_day].lower()}, predicted AQI {worst_aqi}). "
              f"Consider limiting outdoor activity during this period.")
elif worst_aqi > 100:
    st.warning(f"⚠️ AQI is expected to reach '{worst_category}' levels "
               f"({day_labels[worst_day].lower()}, predicted AQI {worst_aqi}). "
               f"Sensitive groups should take precautions.")
else:
    st.success("✅ Air quality is expected to remain at safe levels over the next 3 days.")