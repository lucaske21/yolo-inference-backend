from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from ultralytics import YOLO
import os
import yaml
import torch
import tempfile



app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins. Change to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configuration from environment variables
MODEL_BASE_PATH = os.getenv('MODEL_PATH', './config')
CONF_THRES = float(os.getenv('CONF_THRES', '0.25'))
IOU_THRES = float(os.getenv('IOU_THRES', '0.45'))
INPUT_SIZE = int(os.getenv('INPUT_SIZE', '640'))
OUTPUT_IMG_BASE_PATH = os.getenv('OUTPUT_IMG_BASE_PATH', 'output_img')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))

# Construct paths for model and labels
model_file_path = os.path.join(MODEL_BASE_PATH, 'best.onnx')
labels_file_path = os.path.join(MODEL_BASE_PATH, 'labels.yaml')

# Validate that required files exist
if not os.path.exists(model_file_path):
    raise FileNotFoundError(f"Model file not found: {model_file_path}")
if not os.path.exists(labels_file_path):
    raise FileNotFoundError(f"Labels file not found: {labels_file_path}")

# Load the YOLO model
model = YOLO(model_file_path)

# get model version from onnx model metadata if available
model_version = MODEL_BASE_PATH


# Load labels from YAML file
with open(labels_file_path, 'r') as f:
    labels_data = yaml.safe_load(f)
    model_labels = labels_data.get('names', {})
    
print(f"Model loaded from: {model_file_path}")
print(f"Labels loaded from: {labels_file_path}")
print(f"Available classes: {model_labels}")

# Global state for health checks
model_loaded = True
inference_ok = True

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
    model_path_access = os.path.exists(model_file_path) and os.access(model_file_path, os.R_OK)
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
        "status": "ready" if (model_loaded and inference_ok and model_path_access) else "not_ready",
        "model_loaded": model_loaded,
        "inference_ok": inference_ok,
        "device": device,
        "fs": {
            "model_path_access": model_path_access,
            "tmp_writable": tmp_writable,
            "output_writable": output_writable
        },
        "version": model_version
    }
    
    # Add GPU memory info if available
    if gpu_memory:
        health_status["gpu_memory"] = gpu_memory
    
    return health_status


@app.post("/api/v1/detect")
async def predict(file: UploadFile = File(...)):
    """
    Predict objects in an image using YOLOv11 model.
    
    Args:
        file (UploadFile): The uploaded image file.
        
    Returns:
        dict: A dictionary containing the prediction results.
    """
    # Read the image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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
    img = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_IMG_BASE_PATH, exist_ok=True)
    output_path = os.path.join(OUTPUT_IMG_BASE_PATH, file.filename)
    cv2.imwrite(output_path, img)

    return {"predictions": predictions}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)
