import os
import pandas as pd
import hopsworks
from dotenv import load_dotenv
from src.app.exception import AppException
from src.app.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def upload_to_hopsworks(file_path, feature_group_name):
    """
    Upload a dataset from a CSV file to a Hopsworks feature group.
    """
    try:
        logger.info("Loading Hopsworks API key from environment...")
        load_dotenv()
        api_key = os.getenv("HOPSWORKS_API_KEY")

        # Attempt to log in with cached credentials first
        try:
            logger.info("Attempting to log in with cached credentials...")
            project = hopsworks.login()
            logger.info("Successfully logged in using cached credentials.")
        except Exception as cached_login_error:
            logger.warning("Cached credentials not found. Falling back to API key login.")

            # If no API key is found in the environment, raise an exception
            if not api_key:
                raise AppException("Hopsworks API key not found. Set it in the .env file.")

            # Login using the API key
            try:
                project = hopsworks.login(api_key=api_key)
                logger.info("Successfully logged in using API key.")
            except Exception as api_login_error:
                logger.error(f"Login failed using API key: {api_login_error}")
                raise AppException("Failed to authenticate with Hopsworks. Please check your API key or credentials.")

        fs = project.get_feature_store()

        # Verify the file exists
        if not os.path.exists(file_path):
            raise AppException(f"The file '{file_path}' does not exist.")

        # Load the data
        logger.info(f"Reading data from file: {file_path}")
        data_df = pd.read_csv(file_path)

        # Retrieve existing feature groups to determine the latest version
        logger.info(f"Checking for existing feature groups named '{feature_group_name}'...")
        existing_versions = [
            fg.version for fg in fs.get_feature_groups(name=feature_group_name)
        ]
        latest_version = max(existing_versions, default=0)

        # Attempt to fetch the existing feature group
        try:
            logger.info(f"Fetching existing feature group '{feature_group_name}' (version {latest_version})...")
            feature_group = fs.get_feature_group(feature_group_name, version=latest_version)
            feature_group.insert(data_df, overwrite=False)
            logger.info(f"Data successfully appended to feature group '{feature_group_name}' (version {latest_version}).")
        except hopsworks.client.exceptions.RestAPIError:
            # If the feature group does not exist, create a new one
            new_version = latest_version + 1
            logger.info(f"Feature group not found. Creating new feature group '{feature_group_name}' (version {new_version})...")
            feature_group = fs.create_feature_group(
                name=feature_group_name,
                version=new_version,
                description="Air Quality Index data",
                primary_key=["date"],  # Adjust the primary key column(s) based on your data
                time_travel_format="NONE"  # Change based on requirements (e.g., "HUDI" or "NONE")
            )
            feature_group.insert(data_df)
            logger.info(f"New feature group '{feature_group_name}' (version {new_version}) created and data uploaded.")

    except AppException as e:
        logger.error(f"Application-level error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error during upload operation: {e}")
        raise AppException("Failed to upload data to Hopsworks.", e)


if __name__ == "__main__":
    try:
        logger.info("Starting data upload to Hopsworks...")

        # File path to the AQI CSV file
        file_path = "historical_aqi.csv"

        # Feature group details
        feature_group_name = "historical_aqi_data"

        # Upload data to Hopsworks
        upload_to_hopsworks(file_path, feature_group_name)
        logger.info("Data upload to Hopsworks completed successfully.")

    except AppException as e:
        logger.error(f"Application encountered an error: {e}")
    except Exception as e:
        logger.error(f"Unexpected exception encountered: {e}")
