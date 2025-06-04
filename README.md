# AAU_LLM_VLM_Docker

Server code for LLMs and VLMs used for robot interaction at the materials and production department at AAU.

## Environment Variables

Update your `.env` file with the following configuration:

```env
NGROK_AUTHTOKEN=your_actual_ngrok_auth_token_here
NGROK_DOMAIN=YOUR_DOMAIN.ngrok-free.app
NGROK_ENABLED=true
DEEPSEEK_API_KEY=your_actual_key_here
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
sudo docker build --no-cache -t ai-server .  # "." means you build it from the current directory
```

### Running the Server

```bash
# Run the production container
sudo docker run -d \
  --name ai_container \
  --gpus all \
  --env-file .env \
  -p 8080:80 \
  -v $(pwd)/logs:/app/logs \
  ai-server
```

### Accessing the Container

```bash
# Enter the container to make changes
sudo docker exec -ti ai_container /bin/bash
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

```bash
curl -X POST https://YOUR_DOMAIN.ngrok-free.app/robopoint/predict \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "default",
    "message": "Find the red cup",
    "image": "'$(base64 -w 0 /path/to/your/image.jpg)'"
  }'
```

## How to Set Up New Online LLM Endpoints

This guide explains how to add new online LLM services (like Claude, GPT, etc.) to your server following the DeepSeek implementation pattern.

### Step 1: Environment Variables

Add your API key to the `.env` file:
```env
YOUR_SERVICE_API_KEY=your_actual_api_key_here
```

### Step 2: Create the Endpoint File

Create a new file in `LLM_endpoints/your_service_endpoint.py` following this structure:

```python
###############################################################################
# your_service_endpoint.py – Your Service Online LLM endpoint
###############################################################################
import os
from openai import OpenAI  # or the appropriate SDK
from dotenv import load_dotenv

# Configuration
YOUR_SERVICE_PORT = 8004  # Choose next available port
load_dotenv()
API_KEY = os.getenv("YOUR_SERVICE_API_KEY")

# Initialize client (adapt to your service's SDK)
client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.your-service.com"  # Your service's base URL
)

# Follow the same FastAPI structure as deepseek_online_endpoint.py
# Include: ChatRequest, ChatResponse, chat_endpoint, chat_stream_endpoint, health_check
```

### Step 3: Update Configuration Files

1. **requirements.txt** - Add necessary SDK:
   ```
   your-service-sdk>=1.0.0
   ```

2. **supervisord.conf** - Add program section:
   ```ini
   [program:your_service_endpoint]
   command=python3.11 /app/LLM_endpoints/your_service_endpoint.py
   directory=/app
   autostart=true
   autorestart=true
   startsecs=10
   stdout_logfile=/app/logs/your_service_stdout.log
   stderr_logfile=/app/logs/your_service_stderr.log
   environment=PYTHONPATH="/app"
   priority=3
   ```

3. **nginx.conf** - Add routing:
   ```nginx
   location /your-service/chat-stream {
       proxy_pass http://localhost:8004/chat-stream;
       proxy_buffering off;
       proxy_cache off;
       proxy_set_header X-Accel-Buffering no;
       proxy_read_timeout 600s;
   }

   location /your-service/ {
       rewrite ^/your-service/(.*) /$1 break;
       proxy_pass http://localhost:8004;
   }
   ```

4. **main.py** - Update service listings and port references

5. **Dockerfile** - Add new port to EXPOSE line

### Step 4: API Usage Examples

Once deployed, you can use your new endpoint:

```bash
# Non-streaming chat
curl -X POST https://YOUR_DOMAIN.ngrok-free.app/your-service/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "default",
    "message": "Pick up the red ball",
    "model": "your-model-name",
    "stream": false
  }'

# Streaming chat
curl -s -X POST https://YOUR_DOMAIN.ngrok-free.app/your-service/chat-stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -N --no-buffer \
  -d '{
    "instructions": "custom",
    "message": "Count from 1 to 10",
    "model": "your-model-name",
    "custom_instructions": "Count slowly with explanations"
  }'
```

### Step 5: Common Service Configurations

#### OpenAI GPT
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Models: "gpt-4", "gpt-3.5-turbo", etc.
```

#### Anthropic Claude
```python
import anthropic
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
# Models: "claude-3-opus-20240229", "claude-3-sonnet-20240229", etc.
# Note: Claude has different API structure, adapt accordingly
```

#### Google Gemini
```python
import google.generativeai as genai
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# Models: "gemini-pro", "gemini-pro-vision", etc.
```

### Service-Specific Considerations

- **Rate Limits**: Each service has different rate limits, consider adding rate limiting
- **Token Limits**: Adjust max_tokens based on service capabilities
- **Streaming Support**: Not all services support streaming in the same way
- **Error Handling**: Each API has different error response formats
- **Model Names**: Update available models list for each service

### Testing Your New Endpoint

1. **Health Check**: `GET /your-service/health`
2. **Models List**: `GET /your-service/models` (if implemented)
3. **Simple Chat**: Test with basic message
4. **Streaming**: Test streaming functionality
5. **Error Handling**: Test with invalid requests

## DeepSeek Usage Examples

### Basic Chat Request
```bash
curl -X POST https://YOUR_DOMAIN.ngrok-free.app/deepseek/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "default",
    "message": "Pick up the red ball and place it on the table",
    "model": "deepseek-chat",
    "stream": false
  }'
```

### Reasoning Model
```bash
curl -X POST https://YOUR_DOMAIN.ngrok-free.app/deepseek/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "none",
    "message": "Explain the physics behind throwing a ball",
    "model": "deepseek-reasoner",
    "stream": false
  }'
```

### Streaming Chat
```bash
curl -s -X POST https://YOUR_DOMAIN.ngrok-free.app/deepseek/chat-stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -N --no-buffer \
  -d '{
    "instructions": "custom",
    "message": "Count from 1 to 5 slowly",
    "model": "deepseek-chat",
    "custom_instructions": "Count one number at a time with a brief pause description"
  }' | while IFS= read -r line; do
    if [[ $line == data:* ]]; then
      printf "%s" "${line#data: }"
    fi
  done
```

### Check Available Models
```bash
curl https://YOUR_DOMAIN.ngrok-free.app/deepseek/models
```

### Health Check
```bash
curl https://YOUR_DOMAIN.ngrok-free.app/deepseek/health
```