from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import cv2
import numpy as np
from ultralytics import YOLO
import os



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
MODEL_PATH = os.getenv('MODEL_PATH', './config/best.onnx')
CONF_THRES = float(os.getenv('CONF_THRES', '0.25'))
IOU_THRES = float(os.getenv('IOU_THRES', '0.45'))
INPUT_SIZE = int(os.getenv('INPUT_SIZE', '640'))
OUTPUT_IMG_BASE_PATH = os.getenv('OUTPUT_IMG_BASE_PATH', 'output_img')
HOST = os.getenv('HOST', '0.0.0.0')
PORT = int(os.getenv('PORT', '8000'))

# Load the YOLO model
model = YOLO(MODEL_PATH)
# get the labels from the onnx model
def get_model_labels(model: YOLO):
    labels = model.names
    return labels

model_labels = get_model_labels(model)


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
