# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Set timezone
ENV TZ=Europe/Copenhagen
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    git \
    wget \
    supervisor \
    nginx \
    curl \
    redis-server \
    && rm -rf /var/lib/apt/lists/*

# Add deadsnakes PPA and install Python 3.11
RUN add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
        python3.11 \
        python3.11-dev \
        python3.11-venv \
        libgl1-mesa-glx \
        libglib2.0-0 \
        ffmpeg \
        libxext6 \
        libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Install pip for Python 3.11
RUN wget https://bootstrap.pypa.io/get-pip.py -O /tmp/get-pip.py \
    && python3.11 /tmp/get-pip.py --force-reinstall \
    && rm /tmp/get-pip.py

# Upgrade pip
RUN python3.11 -m pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN python3.11 -m pip install torch==2.1.0+cu121 torchvision==0.16.0+cu121 torchaudio==2.1.0 -f https://download.pytorch.org/whl/cu121/torch_stable.html

# Set working directory
WORKDIR /app

# Install RoboPoint first (with its pinned older dependencies)
RUN python3.11 -m pip install git+https://github.com/wentaoyuan/RoboPoint.git

# Copy requirements and install/upgrade all dependencies
COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir --upgrade -r requirements.txt

# Making sure correct versions of these packages are applied AFTER requirements 
RUN python3.11 -m pip install --upgrade --force-reinstall \
    transformers==4.51.1 \
    safetensors==0.5.3 \
    accelerate==1.7.0 \
    "protobuf<=3.20.3"

# Create necessary directories
RUN mkdir -p /app/logs /var/log/supervisor

# Copy application files (excluding Models for now)
COPY LLM_endpoints/ ./LLM_endpoints/
COPY VLM_endpoints/ ./VLM_endpoints/
COPY main.py .
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY nginx.conf /etc/nginx/nginx.conf

# Copy Models folder (this is the critical part for your local models)
COPY Models/ ./Models/

# Remove default nginx config
RUN rm -f /etc/nginx/sites-enabled/default

# Make scripts executable
RUN chmod +x /app/*.py

# Test RoboPoint installation
RUN python3.11 -c "from robopoint.constants import IMAGE_TOKEN_INDEX; from robopoint.conversation import conv_templates; print('✓ RoboPoint installed successfully')"

# Expose ports
EXPOSE 80 8000 8001 8002 8003

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]