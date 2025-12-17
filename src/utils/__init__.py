"""
Utils package for YOLO Inference Backend.

This package contains utility modules for data models and tools.
"""

from .dataModel import ModelInfo, Models, Metadata

# Import tools only if dependencies are available
try:
    from .tools import load_models, InferenceSessions
    __all__ = ['ModelInfo', 'Models', 'Metadata', 'load_models', 'InferenceSessions']
except ImportError:
    # If ultralytics/torch not available, only export data models
    __all__ = ['ModelInfo', 'Models', 'Metadata']
