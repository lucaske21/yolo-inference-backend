"""
YOLO Inference Backend API.

This module provides a FastAPI application for object detection using YOLO models.
It implements OOP principles with service-oriented architecture, proper logging,
and configuration management.
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any

from config import config
from logger import setup_logging, get_logger
from utils.tools import load_models, InferenceSessions
from utils.tools_lru import InferenceSessionsWithLRU
from services import HealthService, DetectionService


# Initialize logging
setup_logging(log_level=config.log_level)
logger = get_logger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="YOLO Inference Backend",
    description="Object detection API using YOLO models",
    version="2.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ApplicationState:
    """
    Application state management class.
    
    Encapsulates all global application state including models,
    inference sessions, and services.
    
    Attributes:
        models: Models instance with loaded model information
        inference_sessions: InferenceSessions manager instance
        health_service: HealthService instance for health checks
        detection_service: DetectionService instance for object detection
    """
    
    def __init__(self):
        """Initialize application state."""
        logger.info("Initializing application state")
        
        # Validate configuration
        try:
            config.validate()
            logger.info(f"Configuration validated: {config}")
        except ValueError as e:
            logger.error(f"Invalid configuration: {e}")
            raise
        
        # Load models
        try:
            self.models = load_models(config.models_path)
            logger.info(f"Loaded {len(self.models.models)} models")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
        
        # Initialize inference sessions (with or without LRU cache)
        if config.enable_lru_cache:
            logger.info("Using LRU cache mode for inference sessions")
            self.inference_sessions = InferenceSessionsWithLRU(
                max_memory_mb=config.max_memory_mb,
                memory_check_interval=config.memory_check_interval
            )
        else:
            logger.info("Using lazy loading mode for inference sessions")
            self.inference_sessions = InferenceSessions()
        
        try:
            self.inference_sessions.initialize_sessions(self.models, top_n=2)
            logger.info("Inference sessions initialized")
        except Exception as e:
            logger.error(f"Failed to initialize inference sessions: {e}")
            raise
        
        # Initialize services
        self.health_service = HealthService(
            output_img_base_path=config.output_img_base_path
        )
        
        self.detection_service = DetectionService(
            inference_sessions=self.inference_sessions,
            conf_thres=config.conf_thres,
            iou_thres=config.iou_thres,
            output_img_base_path=config.output_img_base_path
        )
        
        logger.info("Application state initialized successfully")


# Initialize application state
try:
    app_state = ApplicationState()
except Exception as e:
    logger.critical(f"Failed to initialize application: {e}")
    raise


@app.get("/api/v2/models")
async def get_models() -> Dict[str, Any]:
    """
    Get list of available models.
    
    Returns:
        Dictionary containing model information
    """
    logger.info("GET /api/v2/models")
    try:
        models_dict = app_state.models.to_dict()
        logger.debug(f"Returning {len(models_dict)} models")
        return models_dict
    except Exception as e:
        logger.error(f"Error getting models: {e}")
        return {"error": str(e)}


@app.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint that returns system status.
    
    Returns comprehensive health status information including:
    - inference_ok: Whether inference is working
    - device: Computing device being used (cpu/cuda/mps)
    - gpu_memory: GPU memory information (if available)
    - fs: Filesystem access status
    
    Returns:
        Dictionary containing health status information
    """
    logger.info("GET /api/v1/health")
    try:
        health_status = app_state.health_service.get_health_status()
        logger.debug(f"Health status: {health_status}")
        return health_status
    except Exception as e:
        logger.error(f"Error during health check: {e}")
        return {
            "inference_ok": False,
            "device": "unknown",
            "error": str(e)
        }


@app.post("/api/v2/detect")
async def predict(
    file: UploadFile = File(...),
    model_id: int = Form(0)
) -> Dict[str, Any]:
    """
    Detect objects in an uploaded image using YOLO model.
    
    Args:
        file: The uploaded image file
        model_id: ID of the model to use (default: 0)
        
    Returns:
        Dictionary containing predictions with bounding boxes and class information
        
    Example usage with curl:
        curl --location 'http://localhost:8000/api/v2/detect' \\
        --form 'file=@"path/to/image.jpg"' \\
        --form 'model_id="0"'
    """
    logger.info(f"POST /api/v2/detect - model_id={model_id}, filename={file.filename}")
    
    try:
        # Read image bytes
        contents = await file.read()
        logger.debug(f"Received image file: {file.filename}, size: {len(contents)} bytes")
        
        # Perform detection using service
        result = app_state.detection_service.detect_objects(
            image_bytes=contents,
            model_id=model_id,
            filename=file.filename
        )
        
        if "error" in result:
            logger.warning(f"Detection failed: {result['error']}")
        else:
            logger.info(
                f"Detection completed: {len(result.get('predictions', []))} objects detected"
            )
        
        return result
        
    except Exception as e:
        logger.error(f"Unexpected error during detection: {e}", exc_info=True)
        return {"error": f"Internal server error: {str(e)}"}


@app.get("/api/v2/cache/stats")
async def get_cache_stats() -> Dict[str, Any]:
    """
    Get LRU cache statistics (only available when LRU cache is enabled).
    
    Returns cache statistics including loaded models, memory usage,
    and eviction information.
    
    Returns:
        Dictionary containing cache statistics or error message
    """
    logger.info("GET /api/v2/cache/stats")
    
    if not config.enable_lru_cache:
        return {
            "error": "LRU cache is not enabled",
            "message": "Set ENABLE_LRU_CACHE=true to use LRU cache mode"
        }
    
    try:
        # Check if the inference_sessions has get_cache_stats method
        if hasattr(app_state.inference_sessions, 'get_cache_stats'):
            stats = app_state.inference_sessions.get_cache_stats()
            logger.debug(f"Cache stats: {stats}")
            return stats
        else:
            return {"error": "Cache statistics not available for this session type"}
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        return {"error": str(e)}


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting server on {config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)
