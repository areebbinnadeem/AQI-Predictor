import joblib
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import hopsworks
from dotenv import load_dotenv
import os
import pandas as pd
from preprocess import load_and_preprocess
from feature_store.fetch_hopsworks_data import fetch_data_from_hopsworks
from src.app.exception import AppException
from src.app.logger import get_logger

logger = get_logger(__name__)

def evaluate_model(model, X_test, y_test, name="Model"):
    """
    Evaluate the model and print performance metrics.
    """
    try:
        preds = model.predict(X_test)
        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)
        logger.info(f"{name} - MSE: {mse:.2f}, R^2: {r2:.2%}")
        print(f"{name} - MSE: {mse:.2f}, R^2: {r2:.2%}")
        return preds
    except Exception as e:
        raise AppException(f"Error occurred while evaluating the model: {e}", e)

def train_xgb(X_train, y_train, X_test, y_test):
    """
    Train an XGBoost model and return the best model.
    """
    try:
        xgb_params = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.2],
        }
        best_mse = float("inf")
        best_model = None
        for n_estimators in xgb_params['n_estimators']:
            for max_depth in xgb_params['max_depth']:
                for learning_rate in xgb_params['learning_rate']:
                    model = XGBRegressor(
                        n_estimators=n_estimators,
                        max_depth=max_depth,
                        learning_rate=learning_rate,
                        random_state=42
                    )
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    mse = mean_squared_error(y_test, preds)
                    if mse < best_mse:
                        best_mse = mse
                        best_model = model
        logger.info(f"Best XGBoost model trained with MSE: {best_mse:.2f}")
        return best_model
    except Exception as e:
        raise AppException(f"Error occurred while training the XGBoost model: {e}", e)

if __name__ == "__main__":
    try:
        # Load the API key from the .env file
        load_dotenv()
        hopsworks_api_key = os.getenv("HOPSWORKS_API_KEY")

        if not hopsworks_api_key:
            raise AppException("HOPSWORKS_API_KEY not found in .env file!")

        # Fetch data from Hopsworks
        data_df = fetch_data_from_hopsworks()
        if data_df is None or data_df.empty:
            raise AppException("Fetched data from Hopsworks is empty or None.")

        data_df['date'] = pd.to_datetime(data_df['date'])

        # Preprocess the data
        X, y, scaler = load_and_preprocess(data_df)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the XGBoost model
        xgb_model = train_xgb(X_train, y_train, X_test, y_test)

        # Evaluate the model
        evaluate_model(xgb_model, X_test, y_test, name="XGBoost")

        # Save the model to Hopsworks
        project = hopsworks.login(api_key=hopsworks_api_key)
        model_registry = project.get_model_registry()
        model_name = "XGB_Model"
        description = "XGBoost model for AQI prediction"

        model_path = "xgb_model.pkl"
        joblib.dump(xgb_model, model_path)

        model = model_registry.python.create_model(name=model_name, description=description)
        model.save(model_path)

        logger.info("Model registered successfully.")
        print("Model registered successfully.")

    except AppException as e:
        logger.error(f"Application Error: {e}")
        print(f"Application Error: {e}")
    except Exception as e:
        logger.error(f"Unexpected Error: {e}")
        print(f"Unexpected Error: {e}")
