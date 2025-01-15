import os
import requests
import hopsworks
import joblib
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

if not HOPSWORKS_API_KEY:
    raise ValueError("HOPSWORKS_API_KEY is not set in the .env file!")
if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY is not set in the .env file!")

# Step 1: Connect to Hopsworks and access the model registry
project = hopsworks.login(api_key=HOPSWORKS_API_KEY)
model_registry = project.get_model_registry()

# Retrieve the model by name and version
model_name = "XGB_Model"
model_metadata = model_registry.get_models(name=model_name)
latest_version = max(model_metadata.keys())
model_version = model_registry.get_model(name=model_name, version=latest_version)
model_dir = model_version.download()

# Find and load the model
model_file = [f for f in os.listdir(model_dir) if f.endswith('.pkl')][0]
model_path = os.path.join(model_dir, model_file)
xgb_model = joblib.load(model_path)


def get_historical_aqi(lat, lon, start_date, end_date):
    """
    Fetch historical AQI data from OpenWeather API.
    """
    base_url = "https://api.openweathermap.org/data/2.5/air_pollution/history"

    # Convert dates to UNIX timestamps
    start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
    end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())

    params = {
        "lat": lat,
        "lon": lon,
        "start": start_timestamp,
        "end": end_timestamp,
        "appid": OPENWEATHER_API_KEY
    }

    response = requests.get(base_url, params=params)
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
    """
    Create a DataFrame from historical AQI data.
    """
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


def predict_next_three_days_aqi(lat, lon):
    """
    Predict AQI for the next three days based on historical data.
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

    historical_aqi = get_historical_aqi(lat, lon, start_date, end_date)

    if not historical_aqi:
        print("Failed to fetch historical AQI data")
        return None

    # Create DataFrame and process data
    pollutants_data = create_dataframe(historical_aqi)
    pollutants_data['date'] = pd.to_datetime(pollutants_data['date'])
    pollutants_data.sort_values('date', inplace=True)

    # Generate lagged features
    for lag in range(1, 4):
        for col in ['aqi', 'co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']:
            pollutants_data[f'{col}_lag_{lag}'] = pollutants_data[col].shift(lag)

    recent_data = pollutants_data.dropna().iloc[-1]
    recent_data['hour'] = recent_data['date'].hour

    # Prepare input for the next three days
    today = datetime.today()
    next_three_days = [today + timedelta(days=i) for i in range(1, 4)]

    input_data = pd.DataFrame({
        'month': [date.month for date in next_three_days],
        'day': [date.day for date in next_three_days],
        'day_of_week': [date.weekday() for date in next_three_days]
    })
    input_data['hour'] = recent_data['hour']

    for col in recent_data.index:
        if 'lag' in col:
            input_data[col] = recent_data[col]

    input_data['is_weekend'] = input_data['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Add rolling averages
    for col in ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']:
        input_data[f'{col}_3hr_avg'] = (
            recent_data[f'{col}_lag_1'] + recent_data[f'{col}_lag_2'] + recent_data[f'{col}_lag_3']
        ) / 3

    input_data = input_data[xgb_model.feature_names_in_]

    # Predict AQI
    predicted_aqi = xgb_model.predict(input_data)

    return pd.DataFrame({
        'Date': [date.strftime('%Y-%m-%d') for date in next_three_days],
        'Predicted_AQI': predicted_aqi.round()
    }).to_dict(orient="records")


# For Testing
if __name__ == "__main__":
    lat, lon = 24.8607, 67.0011  # Coordinates for Karachi
    predictions = predict_next_three_days_aqi(lat, lon)
    if predictions:
        print("\nPredicted AQI for the next three days:")
        for pred in predictions:
            print(f"Date: {pred['Date']}, Predicted AQI: {pred['Predicted_AQI']}")
