###############################################################################
# main.py – Main orchestrator for AI model server with ngrok tunnel
###############################################################################
import asyncio
import logging
import signal
import sys
import time
import os
import threading
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import uvicorn
import ngrok  # Using the newer ngrok SDK

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/main.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

###############################################################################
# CONFIGURATION
###############################################################################
MAIN_PORT = 8000
QWEN3_PORT = 8001
ROBOPOINT_PORT = 8002

# Ngrok Configuration
load_dotenv()
NGROK_AUTHTOKEN = os.getenv("NGROK_AUTHTOKEN", "")
NGROK_DOMAIN = os.getenv("NGROK_DOMAIN", "")
NGROK_ENABLED = os.getenv("NGROK_ENABLED", "true").lower() == "true"

###############################################################################
# NGROK TUNNEL MANAGEMENT
###############################################################################
def setup_ngrok_tunnel():
    """Setup ngrok tunnel using newer ngrok SDK"""
    if not NGROK_ENABLED:
        logger.info("Ngrok disabled")
        return None
        
    if not NGROK_AUTHTOKEN:
        logger.error("NGROK_AUTHTOKEN environment variable not set")
        return None
    
    try:
        logger.info("Setting up ngrok tunnel...")
        logger.info(f"Domain: {NGROK_DOMAIN}")
        logger.info("Tunneling port 80 (nginx)")
        
        # Establish connectivity with custom domain
        listener = ngrok.forward(
            80,  # nginx port
            authtoken_from_env=True,
            domain=NGROK_DOMAIN if NGROK_DOMAIN else None
        )
        
        logger.info(f"🌐 Ngrok tunnel established: {listener.url()}")
        return listener
        
    except Exception as e:
        logger.error(f"Error setting up ngrok tunnel: {e}")
        return None

# Global tunnel reference
ngrok_listener = None
ngrok_thread = None
ngrok_running = False

def keep_ngrok_alive():
    """Keep ngrok listener alive in background thread"""
    global ngrok_running
    try:
        while ngrok_running:
            time.sleep(1)  # Keep the listener alive
    except KeyboardInterrupt:
        logger.info("Ngrok thread interrupted")
    except Exception as e:
        logger.error(f"Ngrok thread error: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global ngrok_listener, ngrok_thread, ngrok_running
    logger.info("Starting AI Model Server Orchestrator...")
    
    # Wait for services to start
    await asyncio.sleep(10)
    
    # Setup ngrok tunnel after services are ready
    if NGROK_ENABLED:
        ngrok_listener = setup_ngrok_tunnel()
        if ngrok_listener:
            # Start background thread to keep ngrok alive
            ngrok_running = True
            ngrok_thread = threading.Thread(target=keep_ngrok_alive, daemon=True)
            ngrok_thread.start()
    
    logger.info("✓ Orchestrator ready")
    
    yield
    
    # Shutdown
    logger.info("Shutting down orchestrator")
    ngrok_running = False
    if ngrok_listener:
        try:
            ngrok_listener.close()
            logger.info("Ngrok tunnel closed")
        except:
            pass

# Create FastAPI app
app = FastAPI(
    title="AI Model Server Orchestrator",
    description="Main orchestrator for AI model endpoints",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# ENDPOINTS
###############################################################################
@app.get("/")
async def root():
    tunnel_info = {}
    if ngrok_listener:
        tunnel_info["ngrok_url"] = str(ngrok_listener.url())
    elif NGROK_DOMAIN:
        tunnel_info["ngrok_url"] = f"https://{NGROK_DOMAIN}"
    
    return {
        "message": "AI Model Server is running",
        "services": {
            "qwen3": f"Port {QWEN3_PORT}",
            "robopoint": f"Port {ROBOPOINT_PORT}"
        },
        "endpoints": {
            "qwen3_chat": "/qwen3/chat",
            "qwen3_chat_stream": "/qwen3/chat-stream", 
            "robopoint_predict": "/robopoint/predict"
        },
        **tunnel_info
    }

@app.get("/health")  
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "tunnel_active": ngrok_listener is not None,
        "ngrok_url": str(ngrok_listener.url()) if ngrok_listener else None
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=MAIN_PORT,
        log_level="info",
        reload=False
    )