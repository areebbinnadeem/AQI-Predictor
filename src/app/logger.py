import logging
import os
from datetime import datetime

# Create logs directory if it doesn't exist
LOG_DIR = "logs"
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Define log file name
log_file = os.path.join(LOG_DIR, f"log_{datetime.now().strftime('%Y_%m_%d')}.log")

# Configure logging
logging.basicConfig(
    filename=log_file,
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

def get_logger(name: str):
    """
    Returns a logger instance for the specified name.
    """
    logger = logging.getLogger(name)
    return logger
