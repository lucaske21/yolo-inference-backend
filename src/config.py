"""
Configuration module for YOLO Inference Backend.

This module provides a centralized configuration management system using
the Singleton pattern to ensure consistent configuration across the application.
"""

from __future__ import annotations

import os
from typing import Optional


class Config:
    """
    Configuration class that encapsulates all application settings.
    
    Uses Singleton pattern to ensure only one configuration instance exists.
    All configuration values are loaded from environment variables with sensible defaults.
    
    Attributes:
        models_path (str): Base path to the models directory
        conf_thres (float): Confidence threshold for detection
        iou_thres (float): IoU threshold for detection
        input_size (int): Input size for detection models
        output_img_base_path (str): Base path to save output images
        host (str): Server host address
        port (int): Server port number
        log_level (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_lru_cache (bool): Enable LRU cache for model management
        max_memory_mb (int): Maximum memory limit in MB for LRU cache
        memory_check_interval (int): Number of requests between memory checks
    """
    
    _instance: Optional[Config] = None
    
    def __new__(cls):
        """Implement Singleton pattern."""
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize configuration from environment variables."""
        if self._initialized:
            return
            
        # Model configuration
        self.models_path = os.getenv('MODELS_PATH', './models')
        
        # Inference parameters
        self.conf_thres = float(os.getenv('CONF_THRES', '0.25'))
        self.iou_thres = float(os.getenv('IOU_THRES', '0.45'))
        self.input_size = int(os.getenv('INPUT_SIZE', '640'))
        
        # Output configuration
        self.output_img_base_path = os.getenv('OUTPUT_IMG_BASE_PATH', 'output_img')
        
        # Server configuration
        self.host = os.getenv('HOST', '0.0.0.0')
        self.port = int(os.getenv('PORT', '8000'))
        
        # Logging configuration
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        
        # LRU Cache configuration
        self.enable_lru_cache = os.getenv('ENABLE_LRU_CACHE', 'false').lower() == 'true'
        self.max_memory_mb = int(os.getenv('MAX_MEMORY_MB', '4096'))
        self.memory_check_interval = int(os.getenv('MEMORY_CHECK_INTERVAL', '10'))
        
        self._initialized = True
    
    def validate(self) -> bool:
        """
        Validate configuration values.
        
        Returns:
            bool: True if configuration is valid, raises ValueError otherwise
            
        Raises:
            ValueError: If any configuration value is invalid
        """
        if not 0.0 <= self.conf_thres <= 1.0:
            raise ValueError(f"conf_thres must be between 0 and 1, got {self.conf_thres}")
        
        if not 0.0 <= self.iou_thres <= 1.0:
            raise ValueError(f"iou_thres must be between 0 and 1, got {self.iou_thres}")
        
        if self.input_size <= 0:
            raise ValueError(f"input_size must be positive, got {self.input_size}")
        
        if not 1 <= self.port <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {self.port}")
        
        if self.max_memory_mb <= 0:
            raise ValueError(f"max_memory_mb must be positive, got {self.max_memory_mb}")
        
        if self.memory_check_interval <= 0:
            raise ValueError(f"memory_check_interval must be positive, got {self.memory_check_interval}")
        
        return True
    
    def __str__(self) -> str:
        """String representation of configuration."""
        return (
            f"Config(models_path={self.models_path}, "
            f"conf_thres={self.conf_thres}, "
            f"iou_thres={self.iou_thres}, "
            f"input_size={self.input_size}, "
            f"output_img_base_path={self.output_img_base_path}, "
            f"host={self.host}, "
            f"port={self.port}, "
            f"log_level={self.log_level}, "
            f"enable_lru_cache={self.enable_lru_cache}, "
            f"max_memory_mb={self.max_memory_mb}, "
            f"memory_check_interval={self.memory_check_interval})"
        )


# Global configuration instance
config = Config()
