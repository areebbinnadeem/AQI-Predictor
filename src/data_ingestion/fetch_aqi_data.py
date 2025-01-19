import os
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from src.app.exception import AppException
from src.app.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

def get_historical_aqi(lat, lon, start_date, end_date):
    """
    Fetch historical AQI data from the OpenWeather API for the specified coordinates and date range.
    """
    try:
        logger.info("Loading OpenWeather API key from environment...")
        load_dotenv()
        API_KEY = os.getenv("OPENWEATHER_API_KEY")

        if not API_KEY:
            raise AppException("OpenWeather API key not found. Set it in the .env file.")

        BASE_URL = "https://api.openweathermap.org/data/2.5/air_pollution/history"

        # Convert dates to UNIX timestamps
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
        end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

        params = {
            "lat": lat,
            "lon": lon,
            "start": start_timestamp,
            "end": end_timestamp,
            "appid": API_KEY
        }

        logger.info(f"Fetching AQI data from OpenWeather API for coordinates ({lat}, {lon}) "
                    f"from {start_date} to {end_date}...")
        response = requests.get(BASE_URL, params=params)

        if response.status_code == 200:
            try:
                return response.json()
            except requests.JSONDecodeError as e:
                logger.error("Failed to decode JSON response.")
                raise AppException("Failed to decode JSON response.", e)
        else:
            logger.error(f"API call failed with status {response.status_code}: {response.text}")
            raise AppException(f"API Error: {response.status_code} - {response.text}")

    except Exception as e:
        logger.error(f"An error occurred while fetching AQI data: {e}")
        raise AppException("Error occurred during API fetch operation.", e)


def create_dataframe(data):
    """
    Create a DataFrame from the AQI data JSON response.
    """
    try:
        logger.info("Transforming API data into a DataFrame...")
        records = []
        for entry in data.get("list", []):
            date = datetime.utcfromtimestamp(entry["dt"]).strftime('%Y-%m-%d %H:%M:%S')
            record = {
                "date": date,
                "aqi": entry["main"]["aqi"],
                **entry["components"]
            }
            records.append(record)

        df = pd.DataFrame(records)
        logger.info(f"Successfully created DataFrame with {len(df)} records.")
        return df

    except Exception as e:
        logger.error(f"Error while creating DataFrame: {e}")
        raise AppException("Failed to create DataFrame from API data.", e)


def save_to_csv(new_data, file_path):
    """
    Save the DataFrame to a CSV file, appending to existing data if the file exists.
    """
    try:
        if os.path.exists(file_path):
            logger.info(f"CSV file '{file_path}' found. Appending new data...")
            existing_data = pd.read_csv(file_path)
            updated_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=["date"]).reset_index(drop=True)
            updated_data.to_csv(file_path, index=False)
            logger.info("New data appended successfully.")
        else:
            logger.info(f"CSV file '{file_path}' not found. Creating a new file...")
            new_data.to_csv(file_path, index=False)
            logger.info("New CSV file created successfully.")
    except Exception as e:
        logger.error(f"Error while saving data to CSV: {e}")
        raise AppException("Failed to save data to CSV.", e)


if __name__ == "__main__":
    try:
        logger.info("Fetching historical AQI data for Karachi...")
        # Coordinates for Karachi
        lat, lon = 24.8607, 67.0011

        # Fixed start date and dynamic end date
        start_date = "2023-01-01"
        end_date = datetime.now().strftime("%Y-%m-%d")

        # Fetch data from OpenWeather API
        data = get_historical_aqi(lat, lon, start_date, end_date)

        if data:
            df = create_dataframe(data)
            save_to_csv(df, "historical_aqi.csv")
            logger.info("Historical AQI data fetching and saving completed successfully.")
        else:
            logger.warning("No data returned from API.")
    except AppException as e:
        logger.error(f"Application-level exception encountered: {e}")
    except Exception as e:
        logger.error(f"Unexpected exception encountered: {e}")
