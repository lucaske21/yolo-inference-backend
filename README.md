# yolo-inference-backend
This repo is the backend of the yolo models

## What's New in v2.0 🎉

The latest release includes a major architectural refactoring with new features and improvements:

### 🏗️ OOP Architecture & Service Layer

The application has been completely refactored using **Object-Oriented Programming principles** with a clean service-oriented architecture:

- **Service Layer**: Business logic is now encapsulated in dedicated service classes
  - `HealthService`: Manages all health check operations
  - `DetectionService`: Handles object detection and inference
- **ApplicationState**: Centralized state management replacing global variables
- **Better Separation of Concerns**: HTTP handlers are separated from business logic
- **Dependency Injection**: Services receive dependencies through constructors for better testability

**Benefits**:
- More maintainable and testable code
- Easier to add new features
- Clear responsibilities for each component
- Better error handling and debugging

### ⚙️ Configuration Management

Introduced a new **Singleton-based configuration system** (`Config` class) that:

- Centralizes all configuration in one place
- Provides type-safe access to configuration values
- Validates configuration on startup
- Supports all environment variables with sensible defaults

**New Environment Variables**:
```bash
LOG_LEVEL=INFO           # Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
MODELS_PATH=./models     # Path to models directory (replaces MODEL_PATH)
```

**Configuration Validation**: The application now validates all configuration values on startup and fails fast with clear error messages if configuration is invalid.

### 📋 Structured Logging

Replaced all `print()` statements with **professional structured logging**:

- Configurable log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Timestamps on all log messages
- Module-aware logging (shows which file/line generated the log)
- Production-ready logging infrastructure
- Easy to redirect logs to files or external systems

**Example logs**:
```
2025-12-17 03:00:01 - app - INFO - app.py:56 - Initializing application state
2025-12-17 03:00:01 - app - INFO - app.py:61 - Configuration validated: Config(models_path=./models, ...)
2025-12-17 03:00:02 - app - INFO - app.py:138 - GET /api/v1/health
```

**Control logging level** via environment variable:
```bash
LOG_LEVEL=DEBUG python src/app.py  # Verbose output for debugging
LOG_LEVEL=ERROR python src/app.py  # Only show errors
```

### 🏥 Enhanced Health Checks

The `/api/v1/health` endpoint now provides **comprehensive system status**:

**New health check features**:
- **Device Detection**: Automatically detects CPU, CUDA GPU (including RTX 4060 Ti), or Apple Silicon MPS
- **GPU Memory Monitoring**: Shows allocated GPU memory and free memory ratio (CUDA only)
- **Filesystem Checks**: Validates access to model files, temporary directory, and output directory
- **Model Status**: Confirms model is loaded and inference is working
- **Version Information**: Returns model version identifier

**Enhanced response structure**:
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

**Use cases**:
- Monitor GPU memory usage in production
- Verify GPU acceleration is working
- Check filesystem permissions
- Kubernetes/Docker health probes
- Debugging deployment issues

### 🎯 Detection Service Improvements

The detection functionality has been enhanced with:

- **Structured Results**: New `DetectionResult` class for type-safe detection results
- **Better Error Handling**: Clear error messages for common issues (invalid model_id, corrupted images)
- **Detailed Logging**: Track inference performance and detect objects
- **Output Image Management**: Automatic output directory creation and management

### 🚀 API v2 Endpoints

Introduced **new v2 API endpoints** with improved functionality:

#### `GET /api/v2/models`
Lists all available models with their metadata.

**Response**:
```json
{
  "models": [
    {
      "id": 0,
      "name": "yolo11m",
      "path": "/app/config/best.onnx",
      "classes": ["person", "bicycle", "car", ...]
    }
  ]
}
```

**Example**:
```bash
curl http://localhost:8000/api/v2/models
```

#### `POST /api/v2/detect`
Enhanced detection endpoint with model selection.

**Parameters**:
- `file`: Image file (multipart/form-data)
- `model_id`: Model to use for detection (default: 0)

**Response**:
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
  ],
  "output_image": "/app/output_img/image_123456.jpg"
}
```

**Example with model selection**:
```bash
curl -X POST http://localhost:8000/api/v2/detect \
  -F "file=@/path/to/image.jpg" \
  -F "model_id=0"
```

### 🧪 Testing Infrastructure

Added comprehensive unit tests:
- Configuration management tests
- Logger functionality tests
- Data model tests
- Service layer tests
- 10+ unit tests covering core functionality

### 🔒 Security & Quality

- **CodeQL Integration**: Automated security scanning (0 vulnerabilities)
- **Type Hints**: Full type annotations throughout codebase
- **Documentation**: Comprehensive docstrings for all classes and methods
- **Code Review**: Automated code review integration

### 📦 Backward Compatibility

**100% backward compatible** - No breaking changes:
- All existing environment variables work as before
- Original API endpoints (`/api/v1/health`, `/api/v1/detect`) still supported
- Docker containers work without changes
- Model directory structure unchanged

**Migration Notes**:
- You can continue using your existing setup without any changes
- To use new features (like structured logging), simply set the `LOG_LEVEL` environment variable
- Consider migrating to v2 endpoints for new applications to get enhanced features

### 📚 Additional Documentation

New documentation files added:
- `REFACTORING.md`: Detailed refactoring documentation with OOP principles and patterns
- `SUMMARY.md`: High-level summary of changes and improvements

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

The application uses a centralized **configuration system** (new in v2.0) that validates all settings on startup. All settings can be configured using environment variables:

| Variable | Description | Default Value | Validation |
|----------|-------------|---------------|------------|
| `MODELS_PATH` | Base path to the directory containing model files (best.onnx and labels.yaml) | `./models` | Must be accessible directory |
| `CONF_THRES` | Confidence threshold for detection | `0.25` | Must be between 0.0 and 1.0 |
| `IOU_THRES` | IoU threshold for detection | `0.45` | Must be between 0.0 and 1.0 |
| `INPUT_SIZE` | Input size for detection models | `640` | Must be positive integer |
| `OUTPUT_IMG_BASE_PATH` | Base path to save output images | `output_img` | Directory will be created if needed |
| `HOST` | Server host address | `0.0.0.0` | Any valid IP address or hostname |
| `PORT` | Server port number | `8000` | Must be between 1 and 65535 |
| `LOG_LEVEL` | Logging verbosity level | `INFO` | DEBUG, INFO, WARNING, ERROR, CRITICAL |
| `MODEL_VERSION` | Model version identifier for health checks | `yolo11m-{current-date}` | Any string |

**Note**: For backward compatibility, `MODEL_PATH` is also supported and maps to `MODELS_PATH`.

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
export MODELS_PATH=./models
export CONF_THRES=0.25
export IOU_THRES=0.45
export OUTPUT_IMG_BASE_PATH=/user/output_img
export HOST=0.0.0.0
export PORT=8000
export LOG_LEVEL=INFO
python src/app.py
```

Or set them inline:

```bash
MODELS_PATH=./my_models CONF_THRES=0.3 IOU_THRES=0.5 LOG_LEVEL=DEBUG python src/app.py
```

**Logging Examples**:

Enable debug logging for verbose output:
```bash
LOG_LEVEL=DEBUG python src/app.py
```

Show only errors and critical messages:
```bash
LOG_LEVEL=ERROR python src/app.py
```

**Note:** `MODELS_PATH` should point to the directory containing both `best.onnx` and `labels.yaml` files. For backward compatibility, `MODEL_PATH` is also supported.

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
  -e MODELS_PATH=/app/config \
  -e CONF_THRES=0.3 \
  -e IOU_THRES=0.5 \
  -e LOG_LEVEL=DEBUG \
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
      - MODELS_PATH=/app/config
      - CONF_THRES=0.25
      - IOU_THRES=0.45
      - OUTPUT_IMG_BASE_PATH=/app/output_img
      - HOST=0.0.0.0
      - PORT=8000
      - LOG_LEVEL=INFO
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
      - MODELS_PATH=/app/config
      - CONF_THRES=0.25
      - IOU_THRES=0.45
      - OUTPUT_IMG_BASE_PATH=/app/output_img
      - HOST=0.0.0.0
      - PORT=8000
      - LOG_LEVEL=INFO
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

### GET /api/v2/models (New in v2.0)

List all available models with their metadata.

**Request:**
- Method: `GET`
- No parameters required

**Response:**
```json
{
  "models": [
    {
      "id": 0,
      "name": "yolo11m",
      "path": "/app/config/best.onnx",
      "classes": ["person", "bicycle", "car", "motorcycle", ...]
    }
  ]
}
```

**Response Fields:**
- `models`: Array of available model objects
  - `id`: Model identifier (used in detection requests)
  - `name`: Model name
  - `path`: Path to model file
  - `classes`: List of class names the model can detect

**Example Usage:**
```bash
curl http://localhost:8000/api/v2/models
```

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

### POST /api/v2/detect (New in v2.0)

Detect objects in an uploaded image using the YOLO model with model selection support.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Parameters:
  - `file`: Image file (required)
  - `model_id`: Model ID to use for detection (optional, default: 0)

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
  ],
  "output_image": "/app/output_img/image_20251217_030001.jpg"
}
```

**Response Fields:**
- `predictions`: Array of detected objects
  - `class_id`: Detected class ID
  - `class_name`: Detected class name
  - `confidence`: Detection confidence score (0.0 to 1.0)
  - `x1, y1, x2, y2`: Bounding box coordinates (top-left and bottom-right)
- `output_image`: Path to saved output image with drawn bounding boxes

**Example Usage:**
```bash
# Use default model (model_id=0)
curl -X POST http://localhost:8000/api/v2/detect \
  -F "file=@/path/to/image.jpg"

# Select specific model
curl -X POST http://localhost:8000/api/v2/detect \
  -F "file=@/path/to/image.jpg" \
  -F "model_id=0"
```

**Error Responses:**
```json
{
  "error": "Model with id 1 not found"
}
```

### POST /api/v1/detect (Legacy)

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

**Note**: Consider migrating to `/api/v2/detect` for enhanced features like model selection and output image paths.
