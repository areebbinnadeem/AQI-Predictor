import sys
import logging
from app.logger import get_logger

logger = get_logger(__name__)

class AppException(Exception):
    """
    Custom Exception Class for the AQI Predictor application.
    """

    def __init__(self, message: str, error: Exception = None):
        """
        Initialize the custom exception.
        
        Args:
        - message (str): Custom error message.
        - error (Exception): Original exception (if any).
        """
        super().__init__(message)
        self.message = message
        self.error = error

        # Log the error
        logger.error(self.__str__())

    def __str__(self):
        """
        String representation of the exception.
        """
        _, _, exc_tb = sys.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename if exc_tb else "Unknown file"
        line_number = exc_tb.tb_lineno if exc_tb else "Unknown line"
        return f"Error occurred in [{file_name}] at line [{line_number}]: {self.message}"
