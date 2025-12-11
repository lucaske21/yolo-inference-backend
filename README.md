# yolo-inference-backend
This repo is the backend of the yolo models

## Model Structure

The application expects the model files to be organized in a specific directory structure:

```
config/
├── best.onnx      # YOLO model file
└── labels.yaml    # Class labels configuration
```

### labels.yaml Format

The `labels.yaml` file should contain the class names for your model:

```yaml
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  # ... add your classes here
```

## Configuration

The application can be configured using environment variables:

| Variable | Description | Default Value |
|----------|-------------|---------------|
| `MODEL_PATH` | Base path to the directory containing model files (best.onnx and labels.yaml) | `./config` |
| `CONF_THRES` | Confidence threshold for detection | `0.25` |
| `IOU_THRES` | IoU threshold for detection | `0.45` |
| `INPUT_SIZE` | Input size for detection models | `640` |
| `OUTPUT_IMG_BASE_PATH` | Base path to save output images | `output_img` |
| `HOST` | Server host address | `0.0.0.0` |
| `PORT` | Server port number | `8000` |
| `MODEL_VERSION` | Model version identifier for health checks | `yolo11m-{current-date}` |

## Usage

### Prerequisites

Before running the application, ensure you have:
1. A trained YOLO model exported to ONNX format (`best.onnx`)
2. A `labels.yaml` file with your model's class names
3. Both files placed in the model directory (default: `./config/`)

### Basic Usage

With default configuration (model files in `./config/`):

```bash
python src/app.py
```

### Using Environment Variables

You can set environment variables before running the application:

```bash
export MODEL_PATH=./config
export CONF_THRES=0.25
export IOU_THRES=0.45
export OUTPUT_IMG_BASE_PATH=/user/output_img
export HOST=0.0.0.0
export PORT=8000
python src/app.py
```

Or set them inline:

```bash
MODEL_PATH=./my_models CONF_THRES=0.3 IOU_THRES=0.5 HOST=127.0.0.1 PORT=9000 python src/app.py
```

**Note:** `MODEL_PATH` should point to the directory containing both `best.onnx` and `labels.yaml` files.

### Docker Usage

#### Prerequisites for Docker

Ensure your model directory structure is ready:

```
your-model-directory/
├── best.onnx
└── labels.yaml
```

**For GPU Support (NVIDIA):**
- Install [NVIDIA Docker runtime](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- Ensure you have compatible NVIDIA drivers (supports RTX 4060 Ti and other modern GPUs)
- CUDA 12.1 or compatible version

#### Pull the Image

```bash
docker pull ghcr.io/lucaske21/yolo-inference-backend:main-latest
```

Or pull a specific version:

```bash
docker pull ghcr.io/lucaske21/yolo-inference-backend:main-abc1234
```

#### Run the Container

**With GPU support (NVIDIA RTX 4060 Ti or other CUDA GPUs):**

```bash
docker run -d \
  --name yolo-backend \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/output_img:/app/output_img \
  ghcr.io/lucaske21/yolo-inference-backend:main-latest
```

**Basic CPU run** (mount your model directory to `/app/config`):

```bash
docker run -d \
  --name yolo-backend \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/output_img:/app/output_img \
  ghcr.io/lucaske21/yolo-inference-backend:main-latest
```

**With custom model path:**

If your model files are in a different directory (e.g., `./my_models`):

```bash
docker run -d \
  --name yolo-backend \
  --gpus all \
  -p 8000:8000 \
  -e MODEL_PATH=/app/models \
  -v $(pwd)/my_models:/app/models \
  -v $(pwd)/output_img:/app/output_img \
  ghcr.io/lucaske21/yolo-inference-backend:main-latest
```

**With custom environment variables and GPU:**

```bash
docker run -d \
  --name yolo-backend \
  --gpus all \
  -p 9000:9000 \
  -e MODEL_PATH=/app/config \
  -e CONF_THRES=0.3 \
  -e IOU_THRES=0.5 \
  -e HOST=0.0.0.0 \
  -e PORT=9000 \
  -e MODEL_VERSION=yolo11m-v1.0 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/output_img:/app/output_img \
  ghcr.io/lucaske21/yolo-inference-backend:main-latest
```

**Important Notes:**
- The mounted directory must contain both `best.onnx` and `labels.yaml` files
- `MODEL_PATH` environment variable should point to the container path where you mount your model directory
- Output images will be saved to the mounted output directory
- Use `--gpus all` flag to enable GPU acceleration (requires NVIDIA Docker runtime)
- The image is built with CUDA 12.1 support for NVIDIA GPUs including RTX 4060 Ti

#### Docker Compose Example

**With GPU Support:**

Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  yolo-backend:
    image: ghcr.io/lucaske21/yolo-inference-backend:main-latest
    container_name: yolo-backend
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/config
      - CONF_THRES=0.25
      - IOU_THRES=0.45
      - OUTPUT_IMG_BASE_PATH=/app/output_img
      - HOST=0.0.0.0
      - PORT=8000
      - MODEL_VERSION=yolo11m-2025-12-11
    volumes:
      - ./config:/app/config
      - ./output_img:/app/output_img
    restart: unless-stopped
```

**CPU Only:**

```yaml
version: '3.8'

services:
  yolo-backend:
    image: ghcr.io/lucaske21/yolo-inference-backend:main-latest
    container_name: yolo-backend
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/config
      - CONF_THRES=0.25
      - IOU_THRES=0.45
      - OUTPUT_IMG_BASE_PATH=/app/output_img
      - HOST=0.0.0.0
      - PORT=8000
    volumes:
      - ./config:/app/config
      - ./output_img:/app/output_img
    restart: unless-stopped
```

Run with:

```bash
docker-compose up -d
```

#### Build Locally

```bash
docker build -t yolo-inference-backend .
```

Run your locally built image with GPU:

```bash
docker run -d \
  --name yolo-backend \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/config:/app/config \
  -v $(pwd)/output_img:/app/output_img \
  yolo-inference-backend
```

#### Verify GPU Support

After starting the container with GPU support, check the health endpoint:

```bash
curl http://localhost:8000/api/v1/health
```

You should see `"device": "cuda:0"` in the response if GPU is properly detected.

### GitHub Actions CI/CD

This repository includes a GitHub Actions workflow that automatically builds and pushes Docker images to GitHub Container Registry (ghcr.io).

**To trigger a build:**
1. Go to the "Actions" tab in your GitHub repository
2. Select "Build and Push Docker Image" workflow
3. Click "Run workflow" button
4. Select the branch and click "Run workflow"

The Docker image will be tagged with the format: `{branch}-{commit-hash}` (e.g., `main-abc1234`)

## API Endpoints

### GET /api/v1/health

Health check endpoint that returns comprehensive system status.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
```json
{
  "status": "ready",
  "model_loaded": true,
  "inference_ok": true,
  "device": "cuda:0",
  "gpu_memory": {
    "allocated": 512123123,
    "free_ratio": 0.74
  },
  "fs": {
    "model_path_access": true,
    "tmp_writable": true,
    "output_writable": true
  },
  "version": "yolo11m-2025-12-11"
}
```

**Response Fields:**
- `status`: Overall status (`ready` or `not_ready`)
- `model_loaded`: Whether the model is successfully loaded
- `inference_ok`: Whether inference is working correctly
- `device`: Computing device being used (`cpu`, `cuda:0`, or `mps` for Apple Silicon)
- `gpu_memory`: GPU memory information (only present if CUDA GPU is available)
  - `allocated`: Currently allocated GPU memory in bytes
  - `free_ratio`: Ratio of free GPU memory (0.0 to 1.0)
- `fs`: Filesystem access checks
  - `model_path_access`: Whether model file is accessible
  - `tmp_writable`: Whether temporary directory is writable
  - `output_writable`: Whether output directory is writable
- `version`: Model version identifier (set via `MODEL_VERSION` env var)

**Example Usage:**
```bash
curl http://localhost:8000/api/v1/health
```

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

**Example Usage:**
```bash
curl -X POST http://localhost:8000/api/v1/detect \
  -F "file=@/path/to/image.jpg"
```
