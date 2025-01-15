import os
import pandas as pd
from dotenv import load_dotenv
import hopsworks
from app.exception import AppException
from app.logger import get_logger

# Initialize the logger
logger = get_logger(__name__)

def fetch_data_from_hopsworks():
    """
    Fetch historical AQI data from Hopsworks from the latest version of the feature group.
    """
    try:
        logger.info("Loading API key from the environment...")
        load_dotenv()
        hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")

        if not hopsworks_api_key:
            raise AppException("HOPSWORKS_API_KEY not found in the environment file!")

        # Login to Hopsworks
        logger.info("Logging into Hopsworks...")
        project = hopsworks.login(api_key=hopsworks_api_key)
        fs = project.get_feature_store()

        # List all feature groups and find the latest version of the target group
        logger.info("Fetching feature groups for 'historical_aqi_data'...")
        feature_groups = fs.get_feature_groups(name="historical_aqi_data")
        if not feature_groups:
            raise AppException("Feature group 'historical_aqi_data' does not exist.")

        # Get the latest version of the feature group
        latest_version = max(fg.version for fg in feature_groups)
        logger.info(f"Latest version of 'historical_aqi_data': {latest_version}")

        # Fetch the data from the latest version
        logger.info("Fetching data from the feature group...")
        feature_group = fs.get_feature_group(name="historical_aqi_data", version=latest_version)
        data_df = feature_group.read()

        logger.info(f"Successfully fetched {len(data_df)} records from Hopsworks (version {latest_version}).")
        return data_df

    except AppException as app_err:
        logger.error(f"Application Error: {app_err}")
        raise app_err

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        raise AppException("An error occurred while fetching data from Hopsworks.", e)

if __name__ == "__main__":
    try:
        logger.info("Fetching AQI data from Hopsworks...")
        df = fetch_data_from_hopsworks()
        if df is not None:
            logger.info("Data fetching process completed successfully.")
        else:
            logger.error("Data fetching process failed.")
    except AppException as e:
        logger.error(f"Application-level exception encountered: {e}")
    except Exception as e:
        logger.error(f"Unexpected exception encountered: {e}")
