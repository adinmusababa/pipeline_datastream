"""
Centralized logging configuration
"""

import logging
import sys
from datetime import datetime


def setup_logging(level=logging.INFO, log_file=None):
    """
    Setup logging configuration untuk seluruh aplikasi
    
    Args:
        level: logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: path ke log file (optional)
    """
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        root_logger.addHandler(file_handler)
    
    # Suppress noisy loggers
    logging.getLogger("pika").setLevel(logging.WARNING)
    logging.getLogger("pymongo").setLevel(logging.WARNING)
    
    logging.info("Logging configured")


def get_logger(name):
    """
    Get logger instance dengan nama spesifik
    
    Args:
        name: nama logger (biasanya __name__)
    
    Returns:
        logging.Logger instance
    """
    return logging.getLogger(name)