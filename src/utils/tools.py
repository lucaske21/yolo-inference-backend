from utils.dataModel import ModelInfo, Models
from ultralytics import YOLO
from typing import Dict
from dataclasses import dataclass


# loading the models from the models base path
def load_models(models_base_path: str) -> Models:
    try:
        models = Models(models={})
        models.load_models_info(models_base_path)
        return models
    except Exception as e:
        raise Exception(f"Failed to load models from {models_base_path}: {str(e)}")




class InferenceSessions:
    '''
    Class to manage inference sessions for different models.
    Attributes:
    ----------
    sessions (Dict[str, Any]): Dictionary mapping model IDs to their inference sessions.
    Methods:
    -------
    get_session(model_id: int) -> YOLO:
        Retrieves the inference session for the specified model ID.
    add_session(model_id: int, session: YOLO) -> None:
        Adds an inference session for the specified model ID.
    '''


    def __init__(self):
        self.sessions: Dict[int, YOLO] = {}
        self.label_names: Dict[int, Dict] = {}

    def get_session(self, model_id: int) -> YOLO:
        return self.sessions.get(model_id)
    
    def get_label_names(self, model_id: int) -> Dict:
        return self.label_names.get(model_id)

    def add_session_label(self, model_id: int, models:Models) -> None:
        # create a YOLO inference session for the model id and add to the sessions dictionary
        model_info = models.models.get(str(model_id))
        if model_info is None:
            raise ValueError(f"Model ID {model_id} not found in models.")
        session = YOLO(model_info.model_path)
        

        self.sessions[model_id] = session
        self.label_names[model_id] = model_info.names

        print(f"Initialized inference session for model ID {model_id} from {model_info.model_path}")
        print(f"Label names: {model_info.names}")


    # added top 2 models to the sessions
    def initialize_sessions(self, models:Models, top_n:int=2) -> None:
        for i in range(top_n):
            self.add_session_label(i, models)
    
