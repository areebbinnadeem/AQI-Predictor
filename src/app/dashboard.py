import os
import pandas as pd
import streamlit as st
from flask import Flask, request, jsonify
from ..prediction.predict_aqi import predict_next_three_days_aqi

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Flask Backend
app = Flask(__name__)

@app.route("/predict_aqi", methods=["GET"])
def predict_aqi():
    """
    API Endpoint to get AQI predictions for the next 3 days.
    Expects lat, lon as query parameters.
    """
    lat = request.args.get("lat", type=float)
    lon = request.args.get("lon", type=float)

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and Longitude are required!"}), 400

    predictions = predict_next_three_days_aqi(lat, lon)
    if predictions:
        return jsonify(predictions)
    else:
        return jsonify({"error": "Failed to fetch predictions"}), 500


# Streamlit Frontend
def streamlit_app():
    """
    Streamlit App for AQI dashboard.
    """
    st.title("🌍 Air Quality Index (AQI) Predictor")
    st.markdown("Real-time and forecasted AQI data with alerts for hazardous levels.")
    
    # Input: Latitude and Longitude
    st.sidebar.header("Enter Location")
    lat = st.sidebar.number_input("Latitude", value=24.8607, format="%.4f")
    lon = st.sidebar.number_input("Longitude", value=67.0011, format="%.4f")

    if st.sidebar.button("Get AQI Forecast"):
        with st.spinner("Fetching AQI predictions..."):
            # Fetch predictions
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
                st.error("Failed to fetch AQI predictions. Please try again.")

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info("Powered by OpenWeather API and Hopsworks")


if __name__ == "__main__":
    # Running Flask in a separate thread
    from threading import Thread
    thread = Thread(target=lambda: app.run(host="0.0.0.0", port=8000, debug=False, use_reloader=False))
    thread.start()

    # Launch Streamlit
    streamlit_app()
