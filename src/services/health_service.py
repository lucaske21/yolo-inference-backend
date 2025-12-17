"""
Health service module for system health checks.

This module provides the HealthService class that encapsulates
all health check logic for the YOLO inference backend.
"""

import os
import tempfile
from typing import Dict, Optional, Any
import torch

from logger import get_logger


logger = get_logger(__name__)


class HealthService:
    """
    Service class for performing system health checks.
    
    This class provides methods to check various aspects of system health
    including model status, device availability, and filesystem access.
    
    Attributes:
        output_img_base_path (str): Path to output image directory
        inference_ok (bool): Whether inference is functioning correctly
    """
    
    def __init__(self, output_img_base_path: str):
        """
        Initialize the HealthService.
        
        Args:
            output_img_base_path: Path to output image directory
        """
        self.output_img_base_path = output_img_base_path
        self.inference_ok = True
        logger.info("HealthService initialized")
    
    def get_device_info(self) -> tuple[str, Optional[Dict[str, Any]]]:
        """
        Get information about the computing device.
        
        Returns:
            Tuple of (device_name, gpu_memory_info)
            where gpu_memory_info is None if no GPU is available
        """
        device = "cpu"
        gpu_memory = None
        
        try:
            if torch.cuda.is_available():
                device = f"cuda:{torch.cuda.current_device()}"
                try:
                    allocated = torch.cuda.memory_allocated()
                    total = torch.cuda.get_device_properties(0).total_memory
                    free_ratio = (total - allocated) / total
                    gpu_memory = {
                        "allocated": allocated,
                        "free_ratio": round(free_ratio, 2)
                    }
                    logger.debug(f"GPU memory: {allocated}/{total} bytes allocated")
                except Exception as e:
                    logger.warning(f"Failed to get GPU memory info: {e}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"  # Apple Silicon GPU
                logger.debug("Using Apple Silicon MPS device")
        except Exception as e:
            logger.error(f"Error detecting device: {e}")
        
        logger.info(f"Current device: {device}")
        return device, gpu_memory
    
    def check_tmp_writable(self) -> bool:
        """
        Check if temporary directory is writable.
        
        Returns:
            bool: True if temporary directory is writable
        """
        try:
            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(b"test")
                logger.debug("Temporary directory is writable")
                return True
        except Exception as e:
            logger.error(f"Temporary directory is not writable: {e}")
            return False
    
    def check_output_writable(self) -> bool:
        """
        Check if output directory is writable.
        
        Returns:
            bool: True if output directory is writable
        """
        try:
            os.makedirs(self.output_img_base_path, exist_ok=True)
            test_file = os.path.join(self.output_img_base_path, '.write_test')
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.debug(f"Output directory {self.output_img_base_path} is writable")
            return True
        except Exception as e:
            logger.error(f"Output directory {self.output_img_base_path} is not writable: {e}")
            return False
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get comprehensive health status of the system.
        
        Returns:
            Dictionary containing health status information including:
            - inference_ok: Whether inference is working
            - device: Computing device being used
            - gpu_memory: GPU memory info (if available)
            - fs: Filesystem access status
        """
        logger.info("Performing health check")
        
        device, gpu_memory = self.get_device_info()
        tmp_writable = self.check_tmp_writable()
        output_writable = self.check_output_writable()
        
        health_status = {
            "inference_ok": self.inference_ok,
            "device": device,
            "fs": {
                "tmp_writable": tmp_writable,
                "output_writable": output_writable
            },
        }
        
        # Add GPU memory info if available
        if gpu_memory:
            health_status["gpu_memory"] = gpu_memory
        
        logger.info(f"Health check completed: inference_ok={self.inference_ok}, device={device}")
        return health_status
    
    def set_inference_status(self, status: bool) -> None:
        """
        Set the inference status.
        
        Args:
            status: True if inference is working, False otherwise
        """
        self.inference_ok = status
        logger.info(f"Inference status set to: {status}")
