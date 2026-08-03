import os
import hopsworks
from dotenv import load_dotenv

load_dotenv()
HOPSWORKS_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(api_key_value=HOPSWORKS_KEY)
fs = project.get_feature_store()

feature_group = fs.get_feature_group(name="aqi_features", version=2)
df = feature_group.read()

print(df)