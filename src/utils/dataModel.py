"""
Data models for YOLO Inference Backend.

This module provides data classes for representing model information,
metadata, and collections of models with validation and loading capabilities.
"""

from dataclasses import dataclass
from typing import Dict
import os
import sys

# Add parent directory to path for logger import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logger import get_logger


logger = get_logger(__name__)

# Below is the sample of model info in the labels.yaml file
# model_name: "fire-smoke-p-tfc-exca-roller"
# model_family: "YOLOv8"     # YOLOv8 / YOLO11
# version: "1.0.0"
# model_path: "./best.onnx"
# task: "detection"   # 分类：classification / detection / segmentation
# input_size: "640x640"

# description: "施工现场安全防护检测模型，用于明火、烟雾、人员、警示桩、挖掘机和压路机识别，适用于白天。"

# metadata:
#   training_data: "NA"
#   date_trained: "2025-11-10"
#   maintainers: "Detection Team"
#   note: "optimized for GPU, ORT < 1.20"
# # Classes
# names:
#   0: 明火
#   1: 烟雾
#   2: 人员
#   3: 警示桩
#   4: 挖掘机
#   5: 压路机





@dataclass
class Metadata:
    """
    Metadata information for YOLO models.
    
    Attributes:
        training_data (str): Training dataset information
        date_trained (str): Date when model was trained
        maintainers (str): Model maintainers
        note (str): Additional notes about the model
    """
    training_data: str
    date_trained: str
    maintainers: str
    note: str


@dataclass
class ModelInfo:
    """
    Data model for storing model information loaded from YAML file.
    
    Attributes:
        model_name (str): Name of the model
        model_family (str): Family of the model (e.g., YOLOv8, YOLO11)
        version (str): Version of the model
        model_path (str): Path to the model file
        task (str): Task type (e.g., classification, detection, segmentation)
        input_size (str): Input size for the model
        description (str): Description of the model
        metadata (Metadata): Metadata information about the model
        names (dict): Dictionary mapping class indices to class names
        
    Methods:
        from_dict(data: dict) -> ModelInfo:
            Creates a ModelInfo instance from a dictionary
        is_complete() -> bool:
            Checks if all required fields are present
        validate_task() -> None:
            Validates the task type
        validate_model_path() -> None:
            Validates that the model path exists
        load_model_info_from_yaml(yaml_path: str) -> ModelInfo:
            Loads model information from a YAML file
            
    Example usage:
        model_info = ModelInfo.load_model_info_from_yaml('path/to/labels.yaml')
        logger.info(f"Loaded model: {model_info}")
    """
    model_name: str
    model_family: str
    version: str
    model_path: str
    task: str
    input_size: str
    description: str
    metadata: Metadata
    names: dict

    def from_dict(data: dict):
        """
        Create ModelInfo instance from dictionary.
        
        Args:
            data: Dictionary containing model information
            
        Returns:
            ModelInfo instance
        """
        return ModelInfo(
            model_name=data.get("model_name", ""),
            model_family=data.get("model_family", ""),
            version=data.get("version", ""),
            model_path=data.get("model_path", ""),
            task=data.get("task", ""),
            input_size=data.get("input_size", ""),
            description=data.get("description", ""),
            metadata=data.get("metadata", Metadata(
                training_data=data.get("metadata", {}).get("training_data", ""),
                date_trained=data.get("metadata", {}).get("date_trained", ""),
                maintainers=data.get("metadata", {}).get("maintainers", ""),
                note=data.get("metadata", {}).get("note", "")
            )),
            names=data.get("names", {})
        )
    

    def is_complete(self):
        """
        Check if all required fields are present.
        
        Returns:
            bool: True if all required fields are present
        """
        required_fields = [
            self.model_name,
            self.model_family,
            self.version,
            self.model_path,
            self.task,
            self.input_size,
            self.description,
            self.metadata,
            self.names
        ]
        return all(required_fields)
    
    def __str__(self):
        """String representation of ModelInfo."""
        return (
            f"ModelInfo(name={self.model_name}, family={self.model_family}, "
            f"version={self.version}, model_path={self.model_path}, "
            f"task={self.task}, input_size={self.input_size}, "
            f"description={self.description}, num_classes={len(self.names)}, "
            f"metadata={self.metadata})"
        )
    
    def validate_task(self):
        """
        Validate the task type.
        
        Raises:
            ValueError: If task is not valid
        """
        valid_tasks = ["classification", "detection", "segmentation"]
        if self.task not in valid_tasks:
            error_msg = f"Invalid task: {self.task}. Must be one of {valid_tasks}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
    def validate_model_path(self):
        """
        Validate that the model path exists.
        
        Raises:
            FileNotFoundError: If model path does not exist
        """
        if not os.path.exists(self.model_path):
            error_msg = f"Model path does not exist: {self.model_path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

    def load_model_info_from_yaml(yaml_path: str):
        """
        Load model information from a YAML file.
        
        Args:
            yaml_path: Path to the YAML file
            
        Returns:
            ModelInfo instance with loaded information
            
        Raises:
            ValueError: If model info is incomplete
            FileNotFoundError: If model path does not exist
        """
        import yaml
        logger.info(f"Loading model info from {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            model_info = ModelInfo.from_dict(data)
            # The model file is in the same folder as the yaml file
            model_info.model_path = os.path.join(
                os.path.dirname(yaml_path),
                model_info.model_path
            )

            if not model_info.is_complete():
                error_msg = f"Incomplete model info in {yaml_path}"
                logger.error(error_msg)
                raise ValueError(error_msg)
            
            # Validate task
            model_info.validate_task()
            # Validate model path
            model_info.validate_model_path()
            
            logger.info(f"Successfully loaded model: {model_info.model_name}")
            return model_info
        

@dataclass
class Models:
    """
    Data model for storing multiple model information.
    
    This class manages a collection of ModelInfo instances and provides
    methods for loading models from a directory structure.
    
    Attributes:
        models (Dict[str, ModelInfo]): Dictionary mapping model IDs to ModelInfo instances
        
    Methods:
        add_model(model_info: ModelInfo) -> None:
            Adds a ModelInfo instance to the models Dictionary
        load_models_info(models_base_path: str) -> None:
            Loads all models from the base path directory
        to_dict() -> dict:
            Returns JSON-serializable representation of models
            
    Example usage:
        models = Models(models={})
        models.load_models_info('./models')
        logger.info(f"Loaded {len(models.models)} models")
    """
    models: Dict[str, ModelInfo]

    def load_models_info(self, models_base_path: str):
        """
        Load models information from the models base path.
        
        The path should contain multiple model folders, each with a labels.yaml file.
        This method scans the base path, loads each labels.yaml file, and adds
        models to the collection.
        
        Args:
            models_base_path: Base directory containing model subdirectories
        """
        logger.info(f"Scanning for models in {models_base_path}")
        i = 0
        
        for model_folder in os.listdir(models_base_path):
            model_path = os.path.join(models_base_path, model_folder)
            if os.path.isdir(model_path):
                labels_file = os.path.join(model_path, 'labels.yaml')

                if os.path.exists(labels_file):
                    try:
                        model_info = ModelInfo.load_model_info_from_yaml(labels_file)
                        self.models[str(i)] = model_info
                        logger.info(
                            f"Loaded model {i}: {model_info.model_name} "
                            f"(version {model_info.version})"
                        )
                        i += 1
                    except Exception as e:
                        logger.error(f"Failed to load model info from {labels_file}: {e}")
        
        logger.info(f"Successfully loaded {len(self.models)} models")

    def to_dict(self):
        """
        Convert Models instance to dictionary representation.
        
        Returns dictionary containing only essential information about each model
        for easy serialization and API responses.
        
        Returns:
            dict: Dictionary with model information (name, version, task, description)
        """
        models_dict = {}
        
        for idx, model_info in self.models.items():
            models_dict[idx] = {
                "model_name": model_info.model_name,
                "version": model_info.version,
                "task": model_info.task,
                "description": model_info.description
            }
        
        logger.debug(f"Converted {len(models_dict)} models to dictionary")
        return models_dict

if __name__ == "__main__":
    # Test loading multiple models info from base path
    models = Models(models={})
    models.load_models_info('../models')
    logger.info(f"Loaded {len(models.models)} models:")
    for idx, model_info in models.models.items():
        logger.info(f"Model {idx}: {model_info}")