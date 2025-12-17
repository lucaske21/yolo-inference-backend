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
    
    Attributes:
        sessions (Dict[int, YOLO]): Dictionary mapping model IDs to YOLO instances
        label_names (Dict[int, Dict]): Dictionary mapping model IDs to label names
    """

    def __init__(self):
        """Initialize empty inference sessions."""
        self.sessions: Dict[int, YOLO] = {}
        self.label_names: Dict[int, Dict] = {}
        logger.info("InferenceSessions initialized")

    def get_session(self, model_id: int) -> Optional[YOLO]:
        """
        Retrieve the inference session for the specified model ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            YOLO model instance or None if not found
        """
        session = self.sessions.get(model_id)
        if session is None:
            logger.warning(f"Session not found for model ID {model_id}")
        return session
    
    def get_label_names(self, model_id: int) -> Optional[Dict]:
        """
        Retrieve label names for the specified model ID.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary of label names or None if not found
        """
        labels = self.label_names.get(model_id)
        if labels is None:
            logger.warning(f"Labels not found for model ID {model_id}")
        return labels

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
        Initialize inference sessions for the top N models.
        
        Args:
            models: Models instance containing model information
            top_n: Number of models to initialize (default: 2)
        """
        logger.info(f"Initializing {top_n} inference sessions")
        for i in range(top_n):
            try:
                self.add_session_label(i, models)
            except Exception as e:
                logger.error(f"Failed to initialize session for model {i}: {e}")
        logger.info(f"Initialized {len(self.sessions)} inference sessions")
    
