"""
Tools module for model loading and inference session management.

This module provides utilities for loading YOLO models and managing
inference sessions with proper encapsulation and logging.
"""

from utils.dataModel import ModelInfo, Models
from ultralytics import YOLO
from typing import Dict, Optional

from logger import get_logger


logger = get_logger(__name__)


def load_models(models_base_path: str) -> Models:
    """
    Load models from the specified base path.
    
    Args:
        models_base_path: Base directory containing model subdirectories
        
    Returns:
        Models instance with loaded model information
        
    Raises:
        Exception: If models cannot be loaded
    """
    try:
        logger.info(f"Loading models from {models_base_path}")
        models = Models(models={})
        models.load_models_info(models_base_path)
        logger.info(f"Successfully loaded {len(models.models)} models")
        return models
    except Exception as e:
        logger.error(f"Failed to load models from {models_base_path}: {str(e)}")
        raise Exception(f"Failed to load models from {models_base_path}: {str(e)}")




class InferenceSessions:
    """
    Manage inference sessions for different YOLO models.
    
    This class provides encapsulated management of model inference sessions,
    including model loading, session storage, and label retrieval.
    Implements lazy loading to reduce startup memory consumption.
    
    Attributes:
        sessions (Dict[int, YOLO]): Dictionary mapping model IDs to YOLO instances
        label_names (Dict[int, Dict]): Dictionary mapping model IDs to label names
        models: Reference to Models instance for lazy loading
    """

    def __init__(self):
        """Initialize empty inference sessions."""
        self.sessions: Dict[int, YOLO] = {}
        self.label_names: Dict[int, Dict] = {}
        self.models: Optional[Models] = None
        logger.info("InferenceSessions initialized")

    def get_session(self, model_id: int) -> Optional[YOLO]:
        """
        Retrieve the inference session for the specified model ID.
        Implements lazy loading - loads model on first access.
        
        Args:
            model_id: ID of the model
            
        Returns:
            YOLO model instance or None if not found
        """
        # Check if session already exists
        if model_id in self.sessions:
            return self.sessions[model_id]
        
        # Lazy load the model if not already loaded
        if self.models is not None:
            try:
                logger.info(f"Lazy loading model {model_id} on first access")
                self.add_session_label(model_id, self.models)
                return self.sessions.get(model_id)
            except Exception as e:
                logger.error(f"Failed to lazy load model {model_id}: {e}")
                return None
        
        logger.warning(f"Session not found for model ID {model_id} and models not available")
        return None
    
    def get_label_names(self, model_id: int) -> Optional[Dict]:
        """
        Retrieve label names for the specified model ID.
        Implements lazy loading - loads model metadata on first access.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary of label names or None if not found
        """
        # Check if labels already exist
        if model_id in self.label_names:
            return self.label_names[model_id]
        
        # Lazy load the model if not already loaded
        if self.models is not None:
            try:
                logger.info(f"Lazy loading model labels {model_id} on first access")
                self.add_session_label(model_id, self.models)
                return self.label_names.get(model_id)
            except Exception as e:
                logger.error(f"Failed to lazy load model labels {model_id}: {e}")
                return None
        
        logger.warning(f"Labels not found for model ID {model_id} and models not available")
        return None

    def add_session_label(self, model_id: int, models: Models) -> None:
        """
        Create and add an inference session for a model.
        
        Args:
            model_id: ID to assign to the model
            models: Models instance containing model information
            
        Raises:
            ValueError: If model ID is not found in models
        """
        model_info = models.models.get(str(model_id))
        if model_info is None:
            error_msg = f"Model ID {model_id} not found in models"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            logger.info(f"Loading YOLO model from {model_info.model_path}")
            session = YOLO(model_info.model_path)
            
            self.sessions[model_id] = session
            self.label_names[model_id] = model_info.names
            
            logger.info(
                f"Initialized inference session for model ID {model_id} "
                f"with {len(model_info.names)} classes"
            )
            logger.debug(f"Label names for model {model_id}: {model_info.names}")
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def initialize_sessions(self, models: Models, top_n: int = 2) -> None:
        """
        Initialize inference sessions manager with models reference.
        Does NOT preload models - uses lazy loading instead.
        
        Args:
            models: Models instance containing model information
            top_n: Number of models to support (default: 2) - kept for backward compatibility
        """
        logger.info(f"Setting up lazy loading for up to {top_n} inference sessions")
        self.models = models
        logger.info(f"Lazy loading configured - models will be loaded on first use")
    
