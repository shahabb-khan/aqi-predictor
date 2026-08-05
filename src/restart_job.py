import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features", version=2)

print("Current state before restart:", fg.materialization_job.get_state())

fg.materialization_job.run()

print("Restart triggered. New state:", fg.materialization_job.get_state())