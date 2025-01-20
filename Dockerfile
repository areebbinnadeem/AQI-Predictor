FROM python:3.9-slim

# Set the PYTHONPATH environment variable
ENV PYTHONPATH=/AQI-Predictor

# Set the working directory inside the container
WORKDIR /AQI-Predictor

# Copy only necessary files for dependency installation
COPY requirements.txt setup.py /AQI-Predictor/

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /AQI-Predictor/

# Expose port and specify default command
EXPOSE 8501
CMD ["streamlit", "run", "src/app/dashboard.py"]
