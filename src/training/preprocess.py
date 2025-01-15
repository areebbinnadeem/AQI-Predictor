import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from feature_store.fetch_hopsworks_data import fetch_data_from_hopsworks

def remove_outliers(df, columns, factor=1.5):
    """
    Remove outliers from specified columns using the IQR method.
    """
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
    return df

def add_features(data):
    """
    Add new features for AQI prediction, including rolling averages, lags, and interactions.
    """
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'

    pollutant_columns = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']

    # Adding season and weekend flags
    data['season'] = data['date'].dt.month.apply(get_season)
    data['is_weekend'] = data['date'].dt.dayofweek.isin([5, 6]).astype(int)

    # Rolling averages
    for col in pollutant_columns:
        data[f'{col}_3hr_avg'] = data[col].rolling(window=3, min_periods=1).mean()
        data[f'{col}_6hr_avg'] = data[col].rolling(window=6, min_periods=1).mean()

    # Date-related features
    data['month'] = data['date'].dt.month
    data['day'] = data['date'].dt.day
    data['day_of_week'] = data['date'].dt.dayofweek
    data['hour'] = data['date'].dt.hour

    # Feature interactions
    data['co_pm2_5'] = data['co'] * data['pm2_5']
    data['no_no2'] = data['no'] * data['no2']
    data['o3_pm10'] = data['o3'] * data['pm10']
    data['so2_nh3'] = data['so2'] * data['nh3']

    # Lags
    for col in pollutant_columns:
        for lag in range(1, 4):
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)

    return data

def preprocess_data_with_lags(data):
    """
    Preprocess the data by encoding, scaling, and splitting features/target.
    """
    one_hot = pd.get_dummies(data['season'], prefix='season')
    data_encoded = pd.concat([data.drop(columns=['season', 'date']), one_hot], axis=1)
    scaler = MinMaxScaler()
    features = scaler.fit_transform(data_encoded.drop(columns=['aqi']))
    target = data_encoded['aqi'].values
    return pd.DataFrame(features, columns=data_encoded.drop(columns=['aqi']).columns), target, scaler

if __name__ == "__main__":
    # Fetch the data from Hopsworks
    data_df = fetch_data_from_hopsworks()

    if data_df is not None:
        data_df['date'] = pd.to_datetime(data_df['date'])

        # Outlier removal
        pollutant_columns = ['co', 'no', 'no2', 'o3', 'so2', 'pm2_5', 'pm10', 'nh3']
        data_df = remove_outliers(data_df, pollutant_columns)

        # Add features
        data_df = add_features(data_df)

        # Drop rows with NaN values
        data_df = data_df.dropna().sort_values(by='date').reset_index(drop=True)

        # Preprocess data
        X, y, scaler = preprocess_data_with_lags(data_df)

        print("Preprocessing completed.")
    else:
        print("Data fetch failed. Exiting.")
