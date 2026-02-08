"""
Tools module with LRU cache for model loading and inference session management.

This module provides utilities for loading YOLO models and managing
inference sessions with LRU (Least Recently Used) cache strategy for
memory-constrained environments.
"""

from collections import OrderedDict
from utils.dataModel import ModelInfo, Models
from ultralytics import YOLO
from typing import Dict, Optional
import psutil
import os

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


class InferenceSessionsWithLRU:
    """
    Manage inference sessions for different YOLO models with LRU cache.
    
    This class provides encapsulated management of model inference sessions
    with intelligent memory management. It implements:
    - Lazy loading: Models load on first access
    - LRU eviction: Automatically unloads least recently used models
    - Memory monitoring: Tracks memory usage and enforces limits
    
    Attributes:
        sessions (OrderedDict[int, YOLO]): OrderedDict mapping model IDs to YOLO instances
        label_names (Dict[int, Dict]): Dictionary mapping model IDs to label names
        models: Reference to Models instance for lazy loading
        max_memory_mb: Maximum memory limit in MB
        memory_check_interval: Number of requests between memory checks
        request_count: Counter for memory check intervals
    """

    def __init__(self, max_memory_mb: int = 4096, memory_check_interval: int = 10):
        """
        Initialize LRU-based inference sessions.
        
        Args:
            max_memory_mb: Maximum memory limit in MB (default: 4096)
            memory_check_interval: Check memory every N requests (default: 10)
        """
        self.sessions: OrderedDict[int, YOLO] = OrderedDict()
        self.label_names: Dict[int, Dict] = {}
        self.models: Optional[Models] = None
        self.max_memory_mb = max_memory_mb
        self.memory_check_interval = memory_check_interval
        self.request_count = 0
        
        logger.info(
            f"InferenceSessionsWithLRU initialized with max_memory={max_memory_mb}MB, "
            f"check_interval={memory_check_interval}"
        )

    def get_current_memory_mb(self) -> float:
        """
        Get current process memory usage in MB.
        
        Returns:
            Memory usage in megabytes
        """
        process = psutil.Process(os.getpid())
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb

    def check_memory_and_evict(self) -> None:
        """
        Check current memory usage and evict least recently used models if needed.
        
        This method is called periodically based on memory_check_interval.
        If memory exceeds max_memory_mb, it evicts models until memory is under control.
        """
        current_memory = self.get_current_memory_mb()
        
        if current_memory > self.max_memory_mb:
            logger.warning(
                f"Memory usage {current_memory:.2f}MB exceeds limit {self.max_memory_mb}MB"
            )
            
            # Evict models until memory is acceptable
            while len(self.sessions) > 0 and current_memory > self.max_memory_mb * 0.9:
                # Remove least recently used (first item in OrderedDict)
                model_id, model = self.sessions.popitem(last=False)
                
                # Also remove label names
                if model_id in self.label_names:
                    del self.label_names[model_id]
                
                # Delete model to free memory
                del model
                
                logger.info(f"Evicted model {model_id} (LRU) to free memory")
                
                # Check memory again
                current_memory = self.get_current_memory_mb()
                logger.info(f"Memory after eviction: {current_memory:.2f}MB")
            
            if len(self.sessions) == 0:
                logger.warning(
                    f"All models evicted but memory still high: {current_memory:.2f}MB"
                )

    def get_session(self, model_id: int) -> Optional[YOLO]:
        """
        Retrieve the inference session for the specified model ID.
        
        Implements lazy loading with LRU cache:
        1. Check if model is already loaded (cache hit)
        2. If cached, move to end (mark as recently used)
        3. If not cached, check memory and evict if needed
        4. Load model on first access
        
        Args:
            model_id: ID of the model
            
        Returns:
            YOLO model instance or None if not found
        """
        # Increment request counter
        self.request_count += 1
        
        # Periodic memory check
        if self.request_count % self.memory_check_interval == 0:
            self.check_memory_and_evict()
        
        # Check if session already exists (cache hit)
        if model_id in self.sessions:
            # Move to end (mark as recently used)
            self.sessions.move_to_end(model_id)
            logger.debug(f"Cache hit for model {model_id}")
            return self.sessions[model_id]
        
        # Cache miss - need to load model
        logger.info(f"Cache miss for model {model_id}, lazy loading...")
        
        # Check memory before loading
        current_memory = self.get_current_memory_mb()
        if current_memory > self.max_memory_mb * 0.8:
            logger.info(
                f"Memory at {current_memory:.2f}MB (80% of limit), "
                "checking if eviction needed before loading new model"
            )
            self.check_memory_and_evict()
        
        # Lazy load the model if not already loaded
        if self.models is not None:
            try:
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
        Also updates LRU order if model is already loaded.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary of label names or None if not found
        """
        # Check if labels already exist
        if model_id in self.label_names:
            # If model is loaded, update LRU order
            if model_id in self.sessions:
                self.sessions.move_to_end(model_id)
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
        
        Loads the model and adds it to the LRU cache (at the end, as most recently used).
        
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
            
            # Add to OrderedDict at the end (most recently used)
            self.sessions[model_id] = session
            self.label_names[model_id] = model_info.names
            
            # Log memory usage after loading
            current_memory = self.get_current_memory_mb()
            logger.info(
                f"Initialized inference session for model ID {model_id} "
                f"with {len(model_info.names)} classes. "
                f"Current memory: {current_memory:.2f}MB"
            )
            logger.debug(f"Label names for model {model_id}: {model_info.names}")
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {e}")
            raise

    def initialize_sessions(self, models: Models, top_n: int = 2) -> None:
        """
        Initialize inference sessions manager with models reference.
        
        Does NOT preload models - uses lazy loading with LRU cache instead.
        
        Args:
            models: Models instance containing model information
            top_n: Number of models to support (default: 2) - kept for backward compatibility
        """
        logger.info(
            f"Setting up LRU cache for up to {top_n} inference sessions "
            f"(max memory: {self.max_memory_mb}MB)"
        )
        self.models = models
        logger.info("LRU cache configured - models will be loaded on first use and evicted when memory is low")
    
    def get_cache_stats(self) -> Dict:
        """
        Get statistics about the current cache state.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "loaded_models": list(self.sessions.keys()),
            "num_loaded_models": len(self.sessions),
            "current_memory_mb": self.get_current_memory_mb(),
            "max_memory_mb": self.max_memory_mb,
            "memory_usage_percent": (self.get_current_memory_mb() / self.max_memory_mb) * 100,
            "request_count": self.request_count,
            "memory_check_interval": self.memory_check_interval
        }
