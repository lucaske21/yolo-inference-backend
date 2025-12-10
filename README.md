# yolo-inference-backend
This repo is the backend of the yolo models

## Configuration

The application can be configured using environment variables:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `MODEL_PATH` | Path to the ONNX model file | `./config/best.onnx` |
| `CONF_THRES` | Confidence threshold for detection | `0.25` |
| `IOU_THRES` | IoU threshold for detection | `0.45` |
| `INPUT_SIZE` | Input size for detection models | `640` |
| `OUTPUT_IMG_BASE_PATH` | Base path to save output images | `output_img` |
| `HOST` | Server host address | `0.0.0.0` |
| `PORT` | Server port number | `8000` |

## Usage

### Basic Usage

```bash
python src/app.py
```

### Using Environment Variables

You can set environment variables before running the application:

```bash
export MODEL_PATH=./config/best.onnx
export CONF_THRES=0.25
export IOU_THRES=0.45
export OUTPUT_IMG_BASE_PATH=/user/output_img
export HOST=0.0.0.0
export PORT=8000
python src/app.py
```

Or set them inline:

```bash
MODEL_PATH=./config/best.onnx CONF_THRES=0.3 IOU_THRES=0.5 HOST=127.0.0.1 PORT=9000 python src/app.py
```

## API Endpoints

### POST /api/v1/detect

Detect objects in an uploaded image using the YOLO model.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Image file

**Response:**
```json
{
  "predictions": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.95,
      "x1": 100,
      "y1": 150,
      "x2": 300,
      "y2": 400
    }
  ]
}
```
