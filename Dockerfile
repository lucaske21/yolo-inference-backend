# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set working directory
WORKDIR /app

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Create symbolic links for python
RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install PyTorch with CUDA 12.1 support first (this will install compatible numpy)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Ensure opencv-python is not installed (use headless version only for server)
RUN pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless || true

# Install opencv-python-headless explicitly for server environments
# This will use numpy already installed by PyTorch (compatible version)
RUN pip install --no-cache-dir opencv-python-headless==4.8.0.74

# Install other Python dependencies (requirements.txt uses flexible numpy version)
RUN pip install --no-cache-dir -r requirements.txt

# Verify installations and show versions
RUN python -c "import torch; import cv2; import numpy; print(f'PyTorch: {torch.__version__}'); print(f'OpenCV: {cv2.__version__}'); print(f'NumPy: {numpy.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"

# Copy application code
COPY src/ ./src/

# Create directories for model and output
RUN mkdir -p /app/config /app/output_img

# Environment variables with default values
ENV MODEL_PATH=/app/config \
    CONF_THRES=0.25 \
    IOU_THRES=0.45 \
    INPUT_SIZE=640 \
    OUTPUT_IMG_BASE_PATH=/app/output_img \
    HOST=0.0.0.0 \
    PORT=8000

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "src/app.py"]
