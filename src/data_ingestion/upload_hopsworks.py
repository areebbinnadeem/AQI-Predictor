import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

def upload_to_hopsworks(file_path, feature_group_name):
    try:
        # Get the API key from the environment variable
        api_key = os.getenv("HOPSWORKS_API_KEY")
        if not api_key:
            raise ValueError("Hopsworks API key not found. Set it in the .env file.")

        # Login to Hopsworks
        project = hopsworks.login(api_key_value=api_key)
        fs = project.get_feature_store()

        # Read the data from the CSV file
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")

        data_df = pd.read_csv(file_path)

        # Determine the latest version of the feature group
        existing_versions = [
            fg.version for fg in fs.get_feature_groups(name=feature_group_name)
        ]
        latest_version = max(existing_versions, default=0)

        # Check if the feature group already exists
        try:
            feature_group = fs.get_feature_group(feature_group_name, version=latest_version)
            feature_group.insert(data_df, overwrite=False)
            print(f"Data appended to the existing feature group '{feature_group_name}' (version {latest_version}).")
        except hopsworks.client.exceptions.RestAPIError:
            # Create a new feature group if it doesn't exist
            new_version = latest_version + 1
            feature_group = fs.create_feature_group(
                name=feature_group_name,
                version=new_version,
                description="Air Quality Index data",
                primary_key=["date"],  # Primary key column
                time_travel_format="NONE"  # Adjust based on your needs
            )
            feature_group.insert(data_df)
            print(f"New feature group '{feature_group_name}' (version {new_version}) created and data uploaded.")

    except Exception as e:
        print(f"Error uploading data to Hopsworks: {e}")


if __name__ == "__main__":
    # File path to the AQI CSV file
    file_path = "src/data_ingestion/historical_aqi.csv"

    # Feature group details
    feature_group_name = "historical_aqi_data"

    # Upload data to Hopsworks
    upload_to_hopsworks(file_path, feature_group_name)
