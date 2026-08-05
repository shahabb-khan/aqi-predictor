import os
import requests
import pandas as pd
import hopsworks
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

LAT = 33.6007
LON = 73.0679


def fetch_raw_data():
    url = f"http://api.openweathermap.org/data/2.5/air_pollution?lat={LAT}&lon={LON}&appid={OPENWEATHER_KEY}"
    response = requests.get(url)
    return response.json()


def calculate_aqi_from_pm25(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]
    for c_low, c_high, aqi_low, aqi_high in breakpoints:
        if c_low <= pm25 <= c_high:
            aqi = ((aqi_high - aqi_low) / (c_high - c_low)) * (pm25 - c_low) + aqi_low
            return round(aqi)
    return 500

def get_last_aqi():
    try:
        project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
        fs = project.get_feature_store()
        feature_group = fs.get_feature_group(name="aqi_features", version=1)
        df = feature_group.read()

        if df.empty:
            return None

        df = df.sort_values("timestamp")
        last_row = df.iloc[-1]
        return last_row["aqi"]
    except Exception as e:
        print(f"Could not fetch last AQI (might be the very first run): {e}")
        return None

def compute_features(raw_data):
    reading = raw_data["list"][0]
    components = reading["components"]
    timestamp = reading["dt"]
    current_time = datetime.fromtimestamp(timestamp)
    aqi = calculate_aqi_from_pm25(components["pm2_5"])
    previous_aqi = get_last_aqi()

    if previous_aqi is not None:
        aqi_change_rate = aqi - previous_aqi
    else:
        aqi_change_rate = 0
    features = {
        "timestamp": timestamp,
        "aqi": aqi,
        "pm2_5": float(components["pm2_5"]),
        "pm10": float(components["pm10"]),
        "co": float(components["co"]),
        "no2": float(components["no2"]),
        "o3": float(components["o3"]),
        "so2": float(components["so2"]),
        "hour": current_time.hour,
        "day": current_time.day,
        "month": current_time.month,
        "day_of_week": current_time.weekday(),
        "aqi_change_rate": aqi_change_rate,    }
    return features


def save_to_feature_store(features):
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()

    df = pd.DataFrame([features])
    df["event_time"] = pd.to_datetime(df["timestamp"], unit="s")

    feature_group = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        description="Hourly AQI features for Rawalpindi",
        primary_key=["timestamp"],
        event_time="event_time",
        time_travel_format="HUDI",
    )
    feature_group.insert(df)
    print("Saved to Hopsworks successfully!")


if __name__ == "__main__":
    raw = fetch_raw_data()
    features = compute_features(raw)
    print(features)
    save_to_feature_store(features)