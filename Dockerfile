FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the application code to the correct location
COPY src /app/src

# Add src to PYTHONPATH
ENV PYTHONPATH="/app/src"

# Expose port and specify default command
EXPOSE 8501
CMD ["streamlit", "run", "/app/src/app/dashboard.py"]
