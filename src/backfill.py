import os
import time
import requests
import pandas as pd
import hopsworks
from datetime import datetime, timedelta
from dotenv import load_dotenv
from feature_pipeline import calculate_aqi_from_pm25  # reuse our existing function

load_dotenv()
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

LAT = 33.6007
LON = 73.0679


def fetch_historical_chunk(start_ts, end_ts):
    """Fetch one chunk of historical data between two dates (as unix timestamps)."""
    url = (
        f"http://api.openweathermap.org/data/2.5/air_pollution/history"
        f"?lat={LAT}&lon={LON}&start={start_ts}&end={end_ts}&appid={OPENWEATHER_KEY}"
    )
    response = requests.get(url)
    data = response.json()
    return data.get("list", [])


def fetch_all_historical_data(months_back=6):
    """Pull 6 months of data, one week at a time, to stay safe with API limits."""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=months_back * 30)

    all_readings = []
    current_start = start_time

    while current_start < end_time:
        current_end = min(current_start + timedelta(days=7), end_time)

        start_ts = int(current_start.timestamp())
        end_ts = int(current_end.timestamp())

        print(f"Fetching {current_start.date()} to {current_end.date()}...")
        chunk = fetch_historical_chunk(start_ts, end_ts)
        all_readings.extend(chunk)

        current_start = current_end
        time.sleep(1)  # small pause to be gentle with the API

    return all_readings


def build_feature_rows(raw_readings):
    """Turn raw historical readings into our feature format, with real aqi_change_rate."""
    # Sort by time first, oldest to newest
    raw_readings.sort(key=lambda r: r["dt"])

    rows = []
    previous_aqi = None

    for reading in raw_readings:
        components = reading["components"]
        timestamp = reading["dt"]
        current_time = datetime.fromtimestamp(timestamp)
        aqi = calculate_aqi_from_pm25(components["pm2_5"])

        aqi_change_rate = 0 if previous_aqi is None else aqi - previous_aqi
        previous_aqi = aqi

        rows.append({
            "timestamp": timestamp,
            "aqi": aqi,
            "pm2_5": components["pm2_5"],
            "pm10": components["pm10"],
            "co": components["co"],
            "no2": components["no2"],
            "o3": components["o3"],
            "so2": components["so2"],
            "hour": current_time.hour,
            "day": current_time.day,
            "month": current_time.month,
            "day_of_week": current_time.weekday(),
            "aqi_change_rate": aqi_change_rate,
        })

    return rows


def save_backfill_to_hopsworks(rows):
    project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
    fs = project.get_feature_store()

    df = pd.DataFrame(rows)
    df["event_time"] = pd.to_datetime(df["timestamp"], unit="s")

    feature_group = fs.get_or_create_feature_group(
        name="aqi_features",
        version=2,
        description="Hourly AQI features for Rawalpindi",
        primary_key=["timestamp"],
        event_time="event_time",
        time_travel_format="HUDI",
    )

    print("Uploading data (without starting the background job yet)...")
    job, _ = feature_group.insert(df, write_options={"start_offline_materialization": False})

    print("Data uploaded. Now running the job and WAITING for it to fully finish...")
    execution = job.run()

    print("Job finished. Success:", execution.success)

    feature_group.insert(df)
    print(f"Saved {len(df)} historical rows to Hopsworks successfully!")


if __name__ == "__main__":
    raw_readings = fetch_all_historical_data(months_back=36)
    print(f"Total readings fetched: {len(raw_readings)}")

    rows = build_feature_rows(raw_readings)
    save_backfill_to_hopsworks(rows)