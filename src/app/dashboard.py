import os
import pandas as pd
import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
from dotenv import load_dotenv
from app.logger import get_logger
from app.exception import AppException
from src.prediction.predict_aqi import predict_next_three_days_aqi

# Load environment variables
load_dotenv()
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Initialize logger
logger = get_logger("Dashboard")

# Flask Backend
app = Flask(__name__)

@app.route("/predict_aqi", methods=["GET"])
def predict_aqi():
    try:
        lat = request.args.get("lat", type=float)
        lon = request.args.get("lon", type=float)

        if lat is None or lon is None:
            raise AppException("Latitude and Longitude are required!", status_code=400)

        logger.info(f"Received request for AQI prediction: lat={lat}, lon={lon}")
        predictions = predict_next_three_days_aqi(lat, lon)

        if predictions:
            logger.info(f"Predictions generated successfully for lat={lat}, lon={lon}")
            return jsonify(predictions)
        else:
            raise AppException("Failed to fetch AQI predictions!", status_code=500)

    except AppException as e:
        logger.error(f"AppException: {str(e)}")
        return jsonify({"error": str(e)}), e.status_code
    except Exception as e:
        logger.exception("Unexpected error occurred while predicting AQI!")
        return jsonify({"error": "An unexpected error occurred!"}), 500


# Streamlit Frontend
def streamlit_app():
    try:
        st.title("🌍 Air Quality Index (AQI) Predictor")
        st.markdown("Forecasted AQI data")
        
        # Fixed Location: Karachi
        st.sidebar.header("Location Information")
        st.sidebar.markdown("**City**: Karachi")
        st.sidebar.markdown("**Latitude**: 24.8607")
        st.sidebar.markdown("**Longitude**: 67.0011")
        
        # Latitude and Longitude for API
        lat = 24.8607
        lon = 67.0011

        if st.sidebar.button("Get AQI Forecast"):
            with st.spinner("Fetching AQI predictions..."):
                try:
                    # Fetch predictions
                    logger.info(f"Fetching AQI predictions for Karachi (lat={lat}, lon={lon})")
                    predictions = predict_next_three_days_aqi(lat, lon)
                    
                    if predictions:
                        # Display predictions
                        st.subheader("Forecasted AQI")
                        forecast_df = pd.DataFrame(predictions)
                        st.write(forecast_df)

                        # Generate Alerts
                        st.subheader("⚠️ Alerts")
                        for pred in predictions:
                            aqi = pred["Predicted_AQI"]
                            date = pred["Date"]
                            if aqi == 5:
                                st.error(f"🚨 {date}: AQI = {aqi} (Very Hazardous). Stay indoors and wear a mask!")
                            elif aqi == 4:
                                st.warning(f"⚠️ {date}: AQI = {aqi} (Unhealthy). Limit outdoor activities.")
                            elif aqi == 3:
                                st.warning(f"⚠️ {date}: AQI = {aqi} (Moderate). Sensitive groups should take precautions.")
                            elif aqi == 2:
                                st.info(f"🌬️ {date}: AQI = {aqi} (Good). Air quality is satisfactory.")
                            elif aqi == 1:
                                st.success(f"✅ {date}: AQI = {aqi} (Excellent). Enjoy the clean air!")
                    else:
                        raise AppException("Failed to fetch AQI predictions. Please try again!")
                except AppException as e:
                    logger.error(f"AppException: {str(e)}")
                    st.error(str(e))
                except Exception as e:
                    logger.exception("Unexpected error occurred while fetching AQI predictions!")
                    st.error("An unexpected error occurred! Please check the logs for more details.")
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.info("Powered by OpenWeather API and Hopsworks")

    except Exception as e:
        logger.exception("Unexpected error occurred in Streamlit app!")
        st.error("An unexpected error occurred! Please check the logs for more details.")


if __name__ == "__main__":
    try:
        # Running Flask in a separate thread
        thread = Thread(target=lambda: app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False))
        thread.start()

        # Launch Streamlit
        streamlit_app()
    except Exception as e:
        logger.exception("Unexpected error occurred while running the application!")
        raise AppException("Failed to launch the application!") from e
