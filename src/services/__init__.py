"""
Services package for YOLO Inference Backend.

This package contains service classes that encapsulate business logic
and provide a clean separation between route handlers and core functionality.
"""

from .health_service import HealthService
from .detection_service import DetectionService

__all__ = ['HealthService', 'DetectionService']
