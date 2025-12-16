from dataclasses import dataclass
from typing import Dict
import os

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
    training_data: str
    date_trained: str
    maintainers: str
    note: str


@dataclass
class ModelInfo:
    '''
    Data model for storing model information loaded from YAML file.
    Attributes:
    ----------
    model_name (str): Name of the model.
    model_family (str): Family of the model (e.g., YOLOv8, YOLO11).
    version (str): Version of the model.
    model_path (str): Path to the model file.
    task (str): Task type (e.g., classification, detection, segmentation).
    input_size (str): Input size for the model.
    description (str): Description of the model.
    metadata (Metadata): Metadata information about the model.
    names (dict): Dictionary mapping class indices to class names.
    Methods:
    -------
    from_dict(data: dict) -> ModelInfo:
        Creates a ModelInfo instance from a dictionary.
    is_complete() -> bool:
        Checks if all required fields are present.
    validate_task() -> None:
        Validates the task type.
    validate_model_path() -> None:
        Validates that the model path exists.
    load_model_info_from_yaml(yaml_path: str) -> ModelInfo:
        Loads model information from a YAML file.

    Example usage:
    -------------
        model_info = ModelInfo.load_model_info_from_yaml('path/to/labels.yaml')
        print(model_info)
    '''
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
    

    # check the info completeness
    def is_complete(self):
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
        return f"ModelInfo(name={self.model_name}, family={self.model_family}, version={self.version}, model_path={self.model_path}, task={self.task}, input_size={self.input_size}, description={self.description}, num_classes={len(self.names)}), metadata={self.metadata})"
    
    # The input validation checking 
    def validate_task(self):
        valid_tasks = ["classification", "detection", "segmentation"]
        if self.task not in valid_tasks:
            raise ValueError(f"Invalid task: {self.task}. Must be one of {valid_tasks}")
        
    def validate_model_path(self):
        import os
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
        


    # load the model info from a yaml file
    def load_model_info_from_yaml(yaml_path: str):
        import yaml
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
            model_info = ModelInfo.from_dict(data)
            # the model file is in the same folder as the yaml file
            model_info.model_path = os.path.join(os.path.dirname(yaml_path), model_info.model_path)

            if not model_info.is_complete():
                raise ValueError(f"Incomplete model info in {yaml_path}")
            
            # validate task
            model_info.validate_task()
            # validate model path
            model_info.validate_model_path()

            return model_info
        

@dataclass
class Models:
    '''
    Data model for storing multiple model information.
    Attributes:
    ----------
    models (Dict[str, ModelInfo]): Dictionary mapping model IDs to ModelInfo instances.
    Methods:
    -------
    add_model(model_info: ModelInfo) -> None:
        Adds a ModelInfo instance to the models Dictionary.

    Example usage:
    -------------
    models = Models()
    '''
    models: Dict[str, ModelInfo]

    # Add models info from the models base path 
    # The path contains multiple model folders, each folder has a labels.yaml file
    # The function will scan the base path, load each labels.yaml file, and add to the models list
    def load_models_info(self, models_base_path: str):
        import os
        # scan the subfolder under the models base path directory to get the model folders
        # and put into the models list
        i = 0
        for model_folder in os.listdir(models_base_path):
            model_path = os.path.join(models_base_path, model_folder)
            if os.path.isdir(model_path):
                labels_file = os.path.join(model_path, 'labels.yaml')

                if os.path.exists(labels_file):
                    try:
                        model_info = ModelInfo.load_model_info_from_yaml(labels_file)

                        self.models[str(i)] = model_info
                        i += 1
                    except Exception as e:
                        print(f"Failed to load model info from {labels_file}: {e}")


    # json representation of the models, to respond to API request
    # just contain the model_name, version, task and description
    def to_dict(self):
        '''
        This method converts the Models instance into a dictionary representation.
        It includes only essential information about each model for easy serialization.
        Returns:
        dict: A dictionary containing model information.
        '''
        models_dict = {}
        
        for idx, model_info in self.models.items():
            models_dict[idx] = {
                "model_name": model_info.model_name,
                "version": model_info.version,
                "task": model_info.task,
                "description": model_info.description
            }
        return models_dict


if __name__ == "__main__":

    # test loading multiple models info from base path
    models = Models(models={})
    models.load_models_info('../models')
    print(f"Loaded {len(models.models)} models:")
    for idx, model_info in models.models.items():
        print(f"Model {idx}: {model_info}")