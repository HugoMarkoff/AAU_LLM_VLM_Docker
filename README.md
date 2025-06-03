# AAU_LLM_VLM_Docker

Server code for LLMs and VLMs used for robot interaction at the materials and production department at AAU.

## Environment Variables

Update your `.env` file with the following configuration:

```env
NGROK_AUTHTOKEN=your_actual_ngrok_auth_token_here
NGROK_DOMAIN=YOUR_DOMAIN.ngrok-free.app
NGROK_ENABLED=true
```

## Model Installation

### Qwen3-8B

1. Install Git LFS (only needed once):
   ```bash
   sudo apt install git-lfs
   git lfs install
   ```

2. Clone the repository:
   ```bash
   git clone https://huggingface.co/Qwen/Qwen3-8B
   ```

### Robopoint

1. Install Git LFS (only needed once):
   ```bash
   git lfs install
   ```

2. Clone the repository:
   ```bash
   git clone https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-13b
   ```

Note: The models may be large, so please be patient during the download process.

## Docker Setup

### First-Time Installation

1. Install Docker:
   ```bash
   https://docs.docker.com/engine/install/ubuntu/
   ```

2. Configure NVIDIA Docker:

   ```bash
   # Remove any malformed entries
   sudo rm -f /etc/apt/sources.list.d/nvidia-docker.list

   # Set up NVIDIA Docker repository
   distribution="ubuntu22.04"
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-docker.gpg
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
     sed 's#^deb https://#deb [signed-by=/usr/share/keyrings/nvidia-docker.gpg] https://#' | \
     sed 's/\$(ARCH)/amd64/g' | \
     sudo tee /etc/apt/sources.list.d/nvidia-docker.list

   # Install NVIDIA container toolkit
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker

   # Test GPU access
   sudo docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04 nvidia-smi
   ```

## Docker Commands

### Rebuilding the Image

```bash
# Completely rebuilding the image (purge all existing containers, images, and volumes)
sudo docker stop $(sudo docker ps -aq)
sudo docker rm $(sudo docker ps -aq)
sudo docker rmi $(sudo docker images -q) --force
sudo docker volume rm $(sudo docker volume ls -q)
sudo docker network rm $(sudo docker network ls -q)
sudo docker system prune -a --volumes -f

# Build fresh image
sudo docker build --no-cache -t ai-model-server .  # "." means you build it from the current directory
```

### Running the Server

```bash
# Run the production container
sudo docker run -d \
  --name ai_model_server_container \
  --gpus all \
  --env-file .env \
  -p 8080:80 \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  ai-model-server
```

### Accessing the Container

```bash
# Enter the container to make changes
sudo docker exec -ti ai_model_server_container /bin/bash
apt-get update
apt-get install nano
```

## Example Usage

### Regular Chat

```bash
# Simple chat request
curl -X POST https://YOUR_DOMAIN.ngrok-free.app/qwen3/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "none",
    "message": "What is the capital of France?"
  }'
```

### Custom Chat

```bash
# Custom instructions request
curl -X POST https://YOUR_DOMAIN.ngrok-free.app/qwen3/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "custom",
    "message": "Pick up the red ball",
    "custom_instructions": "You are a helpful robot assistant. Always respond with enthusiasm and provide clear step-by-step instructions."
  }'
```

### Streaming Chat

```bash
# Streaming chat request
curl -s -X POST https://YOUR_DOMAIN.ngrok-free.app/qwen3/chat-stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -N --no-buffer \
  -d '{
    "instructions": "custom",
    "message": "Count from 1 to 10 slowly",
    "custom_instructions": "Count slowly, one number at a time with spaces between"
  }' | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      printf "%s" "${line#data: }"
    fi
  done
```

## Notes

- Be patient when working with large models, as they may take significant time to download and initialize.
- Ensure you have proper GPU support configured for optimal performance.
- Regularly clean up Docker resources to avoid running out of disk space.