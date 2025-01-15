import hopsworks
from dotenv import load_dotenv
import os

def fetch_data_from_hopsworks():
    """
    Fetch historical AQI data from Hopsworks from the latest version of the feature group.
    """
    try:
        # Load the API key from the .env file
        load_dotenv()
        hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")

        if not hopsworks_api_key:
            raise ValueError("HOPSWORKS_API_KEY not found in .env file!")

        # Login to Hopsworks
        project = hopsworks.login(api_key=hopsworks_api_key)
        fs = project.get_feature_store()

        # List all feature groups and find the latest version of the target group
        feature_groups = fs.get_feature_groups(name="historical_aqi_data")
        if not feature_groups:
            raise ValueError("Feature group 'historical_aqi_data' does not exist.")

        # Get the latest version of the feature group
        latest_version = max(fg.version for fg in feature_groups)

        # Fetch the data from the latest version
        feature_group = fs.get_feature_group(name="historical_aqi_data", version=latest_version)
        data_df = feature_group.read()
        print(f"Fetched {len(data_df)} records from Hopsworks (version {latest_version}).")
        return data_df
    except Exception as e:
        print(f"Error fetching data from Hopsworks: {e}")
        return None

if __name__ == "__main__":
    df = fetch_data_from_hopsworks()
