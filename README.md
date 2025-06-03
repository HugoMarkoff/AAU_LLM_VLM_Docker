# AAU_LLM_VLM_Docker
Server code for LLMs and VLMs used for robot interaction at material and production at AAU. 


# .env
NGROK_AUTHTOKEN=your_actual_ngrok_auth_token_here
NGROK_DOMAIN=YOUR_DOMAIN.ngrok-free.app
NGROK_ENABLED=true


For models - cd into model folder

Find models on huggingface or elsewhere and get the whole model including safetensors:

Examples: 

For Qwen3-8B

sudo apt install git-lfs # only needed once
git lfs install
git clone https://huggingface.co/Qwen/Qwen3-8B


For Eobopoint

git lfs install
git clone https://huggingface.co/wentao-yuan/robopoint-v1-vicuna-v1.5-13b

The models may be large so have patience :) 


To install docker and configure GPU for the first time 

Follow instructions here:

https://docs.docker.com/engine/install/ubuntu/

THEN: 

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


Usefull Docker commands:

Compleately rebuilding the image #Make sure you dont have several dockers - will purge all

sudo docker stop $(sudo docker ps -aq)
sudo docker rm $(sudo docker ps -aq)
sudo docker rmi $(sudo docker images -q) --force
sudo docker volume rm $(sudo docker volume ls -q)
sudo docker network rm $(sudo docker network ls -q)
sudo docker system prune -a --volumes -f

Building fresh:

sudo docker build --no-cache -t ai-model-server .  # "." meanins you build it from the path you are in, so make sure to be in ~/AAU_LLM_VLM_Docker where you have the Dockerfile

Start docker:

# Run the production container
sudo docker run -d \
  --name ai_model_server_container \
  --gpus all \
  --env-file .env \
  -p 8080:80 \
  -v $(pwd)/logs:/app/logs \
  --restart unless-stopped \
  ai-model-server

Enter and edit ode inside of the docker:

sudo docker exec -ti ai_model_server_container /bin/bash 
apt-get update 
apt-get install nano

Some example commands:



# Regular chat:

# none
  curl -X POST https://YOUR_DOMAIN.ngrok-free.app/qwen3/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "none",
    "message": "What is the capital of France?"
  }'

# custom
curl -X POST https://YOUR_DOMAIN.ngrok-free.app/qwen3/chat \
  -H "Content-Type: application/json" \
  -d '{
    "instructions": "custom",
    "message": "Pick up the red ball",
    "custom_instructions": "You are a helpful robot assistant. Always respond with enthusiasm and provide clear step-by-step instructions."
  }'

# streaming chat:
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