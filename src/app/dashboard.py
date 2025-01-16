import os
import pandas as pd
import streamlit as st
from flask import Flask, request, jsonify
from threading import Thread
from dotenv import load_dotenv
from src.app.logger import get_logger
from src.app.exception import AppException
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
    """
    API Endpoint to get AQI predictions for the next 3 days.
    Expects lat, lon as query parameters.
    """
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
    """
    Streamlit App for AQI dashboard.
    """
    try:
        st.title("🌍 Air Quality Index (AQI) Predictor")
        st.markdown("Real-time and forecasted AQI data with alerts for hazardous levels.")
        
        # Input: Latitude and Longitude
        st.sidebar.header("Enter Location")
        lat = st.sidebar.number_input("Latitude", value=24.8607, format="%.4f")
        lon = st.sidebar.number_input("Longitude", value=67.0011, format="%.4f")

        if st.sidebar.button("Get AQI Forecast"):
            with st.spinner("Fetching AQI predictions..."):
                try:
                    # Fetch predictions
                    logger.info(f"Fetching AQI predictions for lat={lat}, lon={lon}")
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
                            if aqi >= 300:
                                st.error(f"🚨 {date}: Hazardous AQI ({aqi})")
                            elif 200 <= aqi < 300:
                                st.warning(f"⚠️ {date}: Very Unhealthy AQI ({aqi})")
                            elif 150 <= aqi < 200:
                                st.warning(f"⚠️ {date}: Unhealthy AQI ({aqi})")
                            elif 100 <= aqi < 150:
                                st.info(f"🌬️ {date}: Moderate AQI ({aqi})")
                            else:
                                st.success(f"✅ {date}: Good AQI ({aqi})")
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
