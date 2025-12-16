from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from ultralytics import YOLO
import os
import yaml
import torch
import tempfile
from utils.tools import load_models
from utils.tools import InferenceSessions


app = FastAPI()

inf_sessions = InferenceSessions()


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration from environment variables
MODELS_BASE_PATH = os.getenv('MODELS_PATH', '../models')
CONF_THRES = float(os.getenv('CONF_THRES', '0.25'))
IOU_THRES = float(os.getenv('IOU_THRES', '0.45'))
INPUT_SIZE = int(os.getenv('INPUT_SIZE', '640'))
OUTPUT_IMG_BASE_PATH = os.getenv('OUTPUT_IMG_BASE_PATH', 'output_img')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))

models = load_models(MODELS_BASE_PATH)
inf_sessions.initialize_sessions(models, top_n=2)
model_loaded = True
inference_ok = True



# GET /api/v2/models
@app.get("/api/v2/models")
async def get_models():
    global models

    return models.to_dict()

@app.get("/api/v1/health")
async def health_check():
    """
    Health check endpoint that returns system status, model status, and resource information.
    
    Returns:
        dict: A dictionary containing comprehensive health status information.
    """
    global model_loaded, inference_ok
    
    # Detect device
    device = "cpu"
    gpu_memory = None
    
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
        except Exception as e:
            print(f"Failed to get GPU memory info: {e}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"  # Apple Silicon GPU
    
    # Check filesystem access
    # model_path_access = False
    # model_path_access = os.path.exists(model_file_path) and os.access(model_file_path, os.R_OK)
    tmp_writable = False
    try:
        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(b"test")
            tmp_writable = True
    except Exception:
        pass
    
    # Check output directory
    output_writable = False
    try:
        os.makedirs(OUTPUT_IMG_BASE_PATH, exist_ok=True)
        test_file = os.path.join(OUTPUT_IMG_BASE_PATH, '.write_test')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        output_writable = True
    except Exception:
        pass
    
    health_status = {
        # "status": "ready" if (model_loaded and inference_ok and model_path_access) else "not_ready",
        "inference_ok": inference_ok,
        "device": device,
        "fs": {
            # "model_path_access": model_path_access,
            "tmp_writable": tmp_writable,
            "output_writable": output_writable
        },
    }
    
    # Add GPU memory info if available
    if gpu_memory:
        health_status["gpu_memory"] = gpu_memory
    
    return health_status


@app.post("/api/v2/detect")
async def predict(file: UploadFile = File(...), model_id: int = Form(0)):
    """
    Predict objects in an image using YOLOv11 model.
    
    Args:
        file (UploadFile): The uploaded image file.
        
    Returns:
        dict: A dictionary containing the prediction results.


    example usage in curl:
        curl --location 'http://localhost:8000/api/v2/detect' \
        --form 'file=@"ds-yolo/nc3-v3-inc_bg-multi-person-yolo/val/images/bdf62324-4709-42ac-8be8-f65f09789bf7.jpg"' \
        --form 'model_id="1"'
    """
    # Read the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(f"Received image with shape: {img.shape}, model_id: {model_id}")

    global inf_sessions

    model = inf_sessions.get_session(model_id)
    model_labels = inf_sessions.get_label_names(model_id)
    if model is None:
        return {"error": f"Model ID {model_id} not found."}
    if model_labels is None:
        return {"error": f"Labels for Model ID {model_id} not found."}


    # Perform prediction
    results = model.predict(source=img, conf=CONF_THRES, iou=IOU_THRES, show_labels=True, show_conf=True)

    # Process results
    predictions = []
    for result in results:
        for box in result.boxes.data.tolist():
            predictions.append({
                "class_id": int(box[5]),
                "class_name": model_labels[int(box[5])],
                "confidence": float(box[4]),
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3])
            })
    # draw the boxes with labels on the image
    # and save it to the output folder using ultralytics' built-in function
    result_img = results[0].plot()
    # img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_IMG_BASE_PATH, exist_ok=True)
    output_path = os.path.join(OUTPUT_IMG_BASE_PATH, file.filename)
    cv2.imwrite(output_path, result_img)

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
