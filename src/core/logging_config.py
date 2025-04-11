"""
Logging configuration for the Financial Risk Analysis System
"""
import os
import logging
import logging.handlers
from pathlib import Path
from datetime import datetime
from .config import BASE_DIR


def setup_logging(log_level=logging.INFO):
    """Configure logging for the application"""
    # Create logs directory if it doesn't exist
    log_dir = BASE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    log_file = log_dir / f"risk_system_{timestamp}.log"
    
    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_format = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
    console_handler.setFormatter(console_format)
    
    # Create file handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10485760,  # 10MB
        backupCount=10
    )
    file_handler.setLevel(log_level)
    file_format = logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s')
    file_handler.setFormatter(file_format)
    
    # Add handlers to root logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Create specific loggers for different parts of the application
    create_module_logger("data", log_level)
    create_module_logger("models", log_level)
    create_module_logger("analysis", log_level)
    create_module_logger("visualization", log_level)
    create_module_logger("dashboard", log_level)
    
    return logger


def create_module_logger(module_name, log_level=logging.INFO):
    """Create logger for a specific module"""
    logger = logging.getLogger(f"risk_system.{module_name}")
    logger.setLevel(log_level)
    return logger 