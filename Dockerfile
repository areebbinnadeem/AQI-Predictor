# Base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy dependencies files
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app/

# Expose the port for Streamlit
EXPOSE 8501

# Environment variables
ENV OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}
ENV HOPSWORKS_API_KEY=${HOPSWORKS_API_KEY}

# Default command to run the Streamlit app
CMD ["streamlit", "run", "src/app/dashboard.py", "--server.port=8501", "--server.enableCORS=false"]
