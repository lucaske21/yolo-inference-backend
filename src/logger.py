"""
Logging module for YOLO Inference Backend.

This module provides centralized logging configuration and utilities
for consistent logging across the application.
"""

import logging
import sys
from typing import Optional


class LoggerConfig:
    """
    Logger configuration and management class.
    
    Provides centralized logging setup with consistent formatting
    and configurable log levels.
    """
    
    _configured = False
    
    @staticmethod
    def setup_logger(
        name: str = 'yolo-inference',
        level: str = 'INFO',
        format_string: Optional[str] = None
    ) -> logging.Logger:
        """
        Set up and configure a logger instance.
        
        Args:
            name: Logger name
            level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            format_string: Custom format string for log messages
            
        Returns:
            Configured logger instance
        """
        logger = logging.getLogger(name)
        
        # Only configure once to avoid duplicate handlers
        if LoggerConfig._configured and logger.handlers:
            return logger
        
        # Set log level
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Create formatter
        if format_string is None:
            format_string = (
                '%(asctime)s - %(name)s - %(levelname)s - '
                '%(filename)s:%(lineno)d - %(message)s'
            )
        formatter = logging.Formatter(format_string)
        console_handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(console_handler)
        
        LoggerConfig._configured = True
        
        return logger
    
    @staticmethod
    def get_logger(name: str = 'yolo-inference') -> logging.Logger:
        """
        Get a logger instance with default configuration.
        
        Args:
            name: Logger name (typically module name)
            
        Returns:
            Logger instance
        """
        return logging.getLogger(name)


def setup_logging(log_level: str = 'INFO') -> logging.Logger:
    """
    Convenience function to set up logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        
    Returns:
        Configured root logger instance
    """
    return LoggerConfig.setup_logger(level=log_level)


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a logger for a specific module.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        
    Returns:
        Logger instance
    """
    return LoggerConfig.get_logger(name)
