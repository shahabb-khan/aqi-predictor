import os
import requests
import pandas as pd
import numpy as np
import hopsworks
import joblib
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

load_dotenv()
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

LAT = 33.6007
LON = 73.0679


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


def fetch_training_data():
    print("Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()
    fg = fs.get_feature_group(name="aqi_features", version=2)

    print("Reading data from Feature Store...")
    df = fg.read()

    # Remove fake test rows
    df = df[df["timestamp"] > 1700000010]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Fix AQI using corrected formula, remove dead-sensor rows
    df = df[df["pm2_5"] > 0].copy()
    df["aqi"] = df["pm2_5"].apply(calculate_aqi_from_pm25)
    df["aqi_change_rate"] = df["aqi"].diff().fillna(0)

    print(f"Data loaded: {df.shape[0]} rows")
    return df


def add_weather_data(df):
    print("Fetching weather history from Open-Meteo...")
    start_date = pd.to_datetime(df["timestamp"].min(), unit="s").strftime("%Y-%m-%d")
    end_date = pd.to_datetime(df["timestamp"].max(), unit="s").strftime("%Y-%m-%d")

    url = (
        f"https://archive-api.open-meteo.com/v1/archive"
        f"?latitude={LAT}&longitude={LON}"
        f"&start_date={start_date}&end_date={end_date}"
        f"&hourly=temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation"
        f"&timezone=UTC"
    )
    response = requests.get(url)
    weather_data = response.json()

    weather_df = pd.DataFrame({
        "event_time": pd.to_datetime(weather_data["hourly"]["time"], utc=True),
        "temperature": weather_data["hourly"]["temperature_2m"],
        "humidity": weather_data["hourly"]["relative_humidity_2m"],
        "wind_speed": weather_data["hourly"]["wind_speed_10m"],
        "precipitation": weather_data["hourly"]["precipitation"],
    })

    df["event_time"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.merge(weather_df, on="event_time", how="left")
    df = df.dropna(subset=["temperature", "humidity", "wind_speed", "precipitation"])

    print(f"After weather merge: {df.shape[0]} rows")
    return df


def build_features(df):
    print("Building engineered features...")
    df = df.sort_values("timestamp").reset_index(drop=True)

    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_lag_72h"] = df["aqi"].shift(72)

    df["aqi_rolling_mean_24h"] = df["aqi"].rolling(window=24, min_periods=24).mean()
    df["aqi_rolling_std_24h"] = df["aqi"].rolling(window=24, min_periods=24).std()

    df["humidity_change_24h"] = df["humidity"].diff(24)
    df["wind_speed_rolling_mean_24h"] = df["wind_speed"].rolling(window=24, min_periods=24).mean()
    df["precipitation_sum_24h"] = df["precipitation"].rolling(window=24, min_periods=24).sum()

    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    # Target: average AQI 49-72 hours ahead (Day 3)
    df["future_aqi"] = df["aqi"].shift(-72).rolling(window=24, min_periods=24).mean()

    df = df.dropna().reset_index(drop=True)
    print(f"Final feature set: {df.shape[0]} rows")
    return df


def train_and_evaluate(df):
    feature_columns = ["pm2_5", "pm10", "co", "no2", "o3", "so2",
                        "day_of_week", "aqi_change_rate", "aqi",
                        "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
                        "aqi_rolling_mean_24h", "aqi_rolling_std_24h",
                        "month_sin", "month_cos", "hour_sin", "hour_cos",
                        "temperature", "humidity", "wind_speed", "precipitation",
                        "humidity_change_24h", "wind_speed_rolling_mean_24h",
                        "precipitation_sum_24h"]

    X = df[feature_columns].values
    y = df["future_aqi"].values

    split_idx = int(len(df) * 0.85)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"Training on {len(X_train)} rows, testing on {len(X_test)} rows...")

    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("=== Training Complete ===")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.3f}")

    return model, feature_columns, {"rmse": rmse, "mae": mae, "r2": r2}


def save_model_locally(model, feature_columns):
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, "models/aqi_model.pkl")
    joblib.dump(feature_columns, "models/feature_columns.pkl")
    print("Model saved locally to models/aqi_model.pkl")

def save_model_to_registry(model, feature_columns, metrics, max_retries=3):
    print("Connecting to Hopsworks Model Registry...")
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    mr = project.get_model_registry()

    aqi_model = mr.python.create_model(
        name="aqi_predictor",
        metrics=metrics,
        description="Random Forest model predicting AQI 72 hours ahead (3-day average) for Rawalpindi, using pollution + weather + engineered time-series features.",
    )

    for attempt in range(1, max_retries + 1):
        try:
            print(f"Uploading model to registry (attempt {attempt}/{max_retries})...")
            aqi_model.save("models")
            print("Model saved to Hopsworks Model Registry successfully!")
            return
        except Exception as e:
            print(f"Upload attempt {attempt} failed: {e}")
            if attempt == max_retries:
                print("All retry attempts failed. Model is still saved locally in models/aqi_model.pkl")
            else:
                print("Retrying...")

if __name__ == "__main__":
    df = fetch_training_data()
    df = add_weather_data(df)
    df = build_features(df)
    model, feature_columns, metrics = train_and_evaluate(df)
    save_model_locally(model, feature_columns)
    save_model_to_registry(model, feature_columns, metrics)