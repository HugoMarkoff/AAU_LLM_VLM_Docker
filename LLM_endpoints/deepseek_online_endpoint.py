###############################################################################
# deepseek_online_endpoint.py – DeepSeek Online LLM endpoint
# 
# USAGE:
# POST /chat
# {
#   "instructions": "default",  // "default", "none", or "custom"
#   "message": "Hello, how are you?",
#   "model": "deepseek-chat", // or "deepseek-reasoner"
#   "stream": false,
#   "custom_instructions": "Your custom system prompt here" // only if instructions="custom"
# }
# 
# POST /chat-stream (for streaming responses)
# Same as above but with stream=true by default
###############################################################################
import os
import json
import logging
import time
from typing import Dict, List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

from openai import OpenAI
from dotenv import load_dotenv

###############################################################################
# CONFIGURATION
###############################################################################
DEEPSEEK_PORT = 8003

# Load environment variables
load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

if not DEEPSEEK_API_KEY:
    raise ValueError("DEEPSEEK_API_KEY environment variable not set")

# Initialize OpenAI client for DeepSeek
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# Available models
AVAILABLE_MODELS = ["deepseek-chat", "deepseek-reasoner"]

# Default system prompt (when instructions="default")
DEFAULT_SYSTEM_PROMPT = "You are the AAU Robot Agent called MAX, you are friendly :)"

###############################################################################
# LOGGING SETUP
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/deepseek.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('deepseek_endpoint')

###############################################################################
# REQUEST SCHEMAS
###############################################################################
class ChatRequest(BaseModel):
    instructions: str = "default"  # "default", "none", or "custom"
    message: str
    model: str = "deepseek-chat"  # "deepseek-chat" or "deepseek-reasoner"
    stream: bool = False
    custom_instructions: Optional[str] = None
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    result: str
    model_used: str

###############################################################################
# CONVERSATION MEMORY
###############################################################################
_session_history: Dict[str, List[Dict[str, str]]] = {}

def get_history(session_id: str) -> List[Dict[str, str]]:
    return _session_history.setdefault(session_id, [])

###############################################################################
# INSTRUCTION PROCESSING
###############################################################################
def get_system_prompt(instructions: str, custom_instructions: Optional[str] = None) -> Optional[str]:
    """Get system prompt based on instruction type"""
    if instructions == "default":
        return DEFAULT_SYSTEM_PROMPT
    elif instructions == "none":
        return None  # No system prompt, use model's default behavior
    elif instructions == "custom":
        if custom_instructions:
            return custom_instructions
        else:
            # Fallback to default if custom requested but not provided
            return DEFAULT_SYSTEM_PROMPT
    else:
        # Invalid instruction type, fallback to default
        return DEFAULT_SYSTEM_PROMPT

###############################################################################
# FASTAPI APP
###############################################################################
app = FastAPI(
    title="DeepSeek Online LLM Endpoint",
    description="DeepSeek online language model for general conversation",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################################################
# STARTUP EVENT
###############################################################################
@app.on_event("startup")
async def startup_event():
    """Test API connection on startup"""
    try:
        logger.info("Starting DeepSeek endpoint...")
        # Test API connection
        test_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        logger.info("✓ DeepSeek API connection successful")
        logger.info("DeepSeek endpoint ready")
    except Exception as e:
        logger.error(f"Failed to connect to DeepSeek API: {str(e)}")

###############################################################################
# ENDPOINTS
###############################################################################
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Synchronous chat endpoint"""
    try:
        # Validate model
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model. Available models: {AVAILABLE_MODELS}"
            )
        
        history = get_history(request.session_id)
        history.append({"role": "user", "content": request.message})
        
        # Get system prompt based on instructions
        system_prompt = get_system_prompt(request.instructions, request.custom_instructions)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        
        # Call DeepSeek API
        response = client.chat.completions.create(
            model=request.model,
            messages=messages,
            stream=request.stream,
            temperature=0.7,
            max_tokens=1024
        )
        
        resp = response.choices[0].message.content.strip()
        
        history.append({"role": "assistant", "content": resp})
        
        return ChatResponse(result=resp, model_used=request.model)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint"""
    try:
        # Validate model
        if request.model not in AVAILABLE_MODELS:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid model. Available models: {AVAILABLE_MODELS}"
            )
        
        history = get_history(request.session_id)
        history.append({"role": "user", "content": request.message})
        
        system_prompt = get_system_prompt(request.instructions, request.custom_instructions)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        
        # Call DeepSeek API with streaming
        stream = client.chat.completions.create(
            model=request.model,
            messages=messages,
            stream=True,
            temperature=0.7,
            max_tokens=1024
        )
        
        collected = ""
        
        def event_stream():
            nonlocal collected
            try:
                for chunk in stream:
                    if chunk.choices[0].delta.content is not None:
                        token = chunk.choices[0].delta.content
                        collected += token
                        # Yield each token immediately
                        yield f"data: {token}\n\n"
                                
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
        
        def finish_chat():
            """Save conversation to history after streaming completes"""
            history.append({"role": "assistant", "content": collected.strip()})
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",
            },
            background=finish_chat
        )
        
    except Exception as e:
        logger.error(f"Stream endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test API connection
        test_response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1
        )
        return {
            "status": "healthy",
            "service": "deepseek",
            "available_models": AVAILABLE_MODELS,
            "api_status": "connected",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "deepseek", 
            "available_models": AVAILABLE_MODELS,
            "api_status": f"error: {str(e)}",
            "timestamp": time.time()
        }

@app.get("/models")
async def list_models():
    """List available models"""
    return {
        "available_models": AVAILABLE_MODELS,
        "default_model": "deepseek-chat"
    }

@app.post("/clear-history")
async def clear_history(session_id: str = "default"):
    """Clear conversation history for a session"""
    if session_id in _session_history:
        del _session_history[session_id]
        return {"message": f"History cleared for session: {session_id}"}
    return {"message": f"No history found for session: {session_id}"}

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    uvicorn.run(
        "deepseek_online_endpoint:app",  
        host="0.0.0.0",
        port=DEEPSEEK_PORT,
        log_level="info",
        reload=False
    )