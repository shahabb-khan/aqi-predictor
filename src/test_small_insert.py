import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="aqi_features",
    version=2,
    primary_key=["timestamp"],
    event_time="event_time",
    time_travel_format="HUDI",
)

# Just 5 fake test rows
test_df = pd.DataFrame([
    {"timestamp": 1700000001 + i, "aqi": 50, "pm2_5": 10.0, "pm10": 20.0,
     "co": 100.0, "no2": 1.0, "o3": 50.0, "so2": 1.0,
     "hour": 12, "day": 1, "month": 1, "day_of_week": 1,
     "aqi_change_rate": 0}
    for i in range(5)
])
test_df["event_time"] = pd.to_datetime(test_df["timestamp"], unit="s")

job, _ = fg.insert(test_df, write_options={"start_offline_materialization": False})
execution = job.run()
print("Test job finished. Success:", execution.success)