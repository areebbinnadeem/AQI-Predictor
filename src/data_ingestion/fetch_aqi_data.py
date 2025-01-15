import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

def get_historical_aqi(lat, lon, start_date, end_date):
    # Fetch the API key from the environment variable
    API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if not API_KEY:
        raise ValueError("OpenWeather API key not found. Set it in the .env file.")
    
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

    response = requests.get(BASE_URL, params=params)

    if response.status_code == 200:
        try:
            return response.json()
        except requests.JSONDecodeError:
            print("Failed to decode JSON response")
            return None
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    

def create_dataframe(data):
    records = []
    for entry in data["list"]:
        date = datetime.utcfromtimestamp(entry["dt"]).strftime('%Y-%m-%d %H:%M:%S')
        record = {
            "date": date,
            "aqi": entry["main"]["aqi"],
            "co": entry["components"].get("co", None),
            "no": entry["components"].get("no", None),
            "no2": entry["components"].get("no2", None),
            "o3": entry["components"].get("o3", None),
            "so2": entry["components"].get("so2", None),
            "pm2_5": entry["components"].get("pm2_5", None),
            "pm10": entry["components"].get("pm10", None),
            "nh3": entry["components"].get("nh3", None)
        }
        records.append(record)

    return pd.DataFrame(records)


def save_to_csv(new_data, file_path):
    if os.path.exists(file_path):
        # Load the existing data
        existing_data = pd.read_csv(file_path)

        # Append new data and remove duplicates
        updated_data = pd.concat([existing_data, new_data]).drop_duplicates(subset=["date"]).reset_index(drop=True)
        updated_data.to_csv(file_path, index=False)
        print("New data appended to the existing CSV file.")
    else:
        # Save new data as a new file
        new_data.to_csv(file_path, index=False)
        print("New CSV file created.")

if __name__ == "__main__":
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
