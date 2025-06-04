###############################################################################
# qwen3.py – QWen3 LLM endpoint for robot command conversion
# 
# USAGE:
# POST /chat
# {
#   "instructions": "default",  // "default", "none", or "custom"
#   "message": "Pick the white cup and place it on the black case",
#   "custom_instructions": "Your custom system prompt here" // only if instructions="custom"
# }
# 
# POST /chat-stream (for streaming responses)
# {
#   "instructions": "default",  // "default", "none", or "custom"
#   "message": "Pick the white cup and place it on the black case",
#   "custom_instructions": "Your custom system prompt here" // only if instructions="custom"
# }
###############################################################################
import threading
import torch
import re
import json
import logging
import time
from typing import Dict, List, Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import uvicorn

# Transformers imports
try:
    from transformers.streamers import TextIteratorStreamer
except ImportError:
    from transformers import TextIteratorStreamer

from transformers import AutoModelForCausalLM, AutoTokenizer

###############################################################################
# CONFIGURATION
###############################################################################
QWEN3_PORT = 8001

# Use the correct model from your original code
DEFAULT_LLM_NAME = "/app/Models/Qwen3-8B"

LLM_KWARGS = {
    "torch_dtype": "auto",
    "device_map": "auto",
    "trust_remote_code": True,  
}

# Default system prompt (when instructions="default")
DEFAULT_SYSTEM_PROMPT = r"""
You are the AAU Robot Agent. Your job is to convert user commands into robot actions.

IMPORTANT:
- Do NOT output any reasoning, explanation, or extra text.
- Do NOT use <think> tags.
- Do NOT output [DONE].
- Output ONLY the [ACTION] block or a clarifying question.
- Each line in the [ACTION] block must be on its own line.
- There must be a line break between "RoboPoint Request: ..." and "Action Request: ...".

[ACTION]
RoboPoint Request: object1; object2; object3
Action Request: Pick; Place; Pick

- If you do NOT have enough information, output ONLY a clarifying question (no reasoning, no explanation).

EXAMPLES:

User: "Pick the white cup and place it on the black case"
You:
[ACTION]
RoboPoint Request: white cup; black case
Action Request: Pick; Place

User: "Pick up the blue bottle"
You:
[ACTION]
RoboPoint Request: blue bottle
Action Request: Pick

User: "Move something"
You:
Which object should I move?

User: "Pick the red block on the left and put it on the table"
You:
[ACTION]
RoboPoint Request: red block on the left; table
Action Request: Pick; Place

IMPORTANT:
- Do NOT output any reasoning, explanation, or extra text.
- Do NOT use <think> tags.
- Do NOT output [DONE].
- Output ONLY the [ACTION] block or a clarifying question.
"""

###############################################################################
# LOGGING SETUP
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/qwen3.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('qwen3_endpoint')

###############################################################################
# REQUEST SCHEMAS
###############################################################################
class ChatRequest(BaseModel):
    instructions: str = "default"  # "default", "none", or "custom"
    message: str
    custom_instructions: Optional[str] = None
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    result: str
    parsed: Optional[Dict] = None

###############################################################################
# MODEL MANAGEMENT
###############################################################################
class QWen3ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.model_loaded = False
        self.loading_lock = threading.Lock()
    def load_model(self):
        """Load QWen3 model and tokenizer"""
        with self.loading_lock:
            if self.model_loaded:
                return  
            logger.info(f"Loading QWen3 model '{DEFAULT_LLM_NAME}'...")
            start_time = time.time()
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(DEFAULT_LLM_NAME, trust_remote_code=True)
                self.model = AutoModelForCausalLM.from_pretrained(DEFAULT_LLM_NAME, **LLM_KWARGS).eval()  
                load_time = time.time() - start_time
                logger.info(f"✓ QWen3 model loaded successfully in {load_time:.2f}s")
                self.model_loaded = True
            except Exception as e:
                logger.error(f"✗ Failed to load QWen3 model: {str(e)}")
                raise e
    def is_ready(self):
        return self.model_loaded

# Global model manager
model_manager = QWen3ModelManager()

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
# ACTION PARSING
###############################################################################
ACTION_BLOCK_RE = re.compile(
    r"\[ACTION\]\s*"                       
    r"RoboPoint\s*Request:\s*(?P<reqs>.*?)\s*"   
    r"Action\s*Request:\s*(?P<acts>.*)",          
    flags=re.IGNORECASE | re.DOTALL,
)

def parse_action_block(text: str) -> Optional[Dict[str, List[str]]]:
    """Extract the 2 aligned lists from an [ACTION] block"""
    m = ACTION_BLOCK_RE.search(text)
    if not m:
        return None

    reqs = [r.strip() for r in re.split(r"[;|]", m["reqs"]) if r.strip()]
    acts = [a.strip() for a in re.split(r"[;|]", m["acts"]) if a.strip()]

    if len(reqs) != len(acts) or not reqs:
        return None
    return {"requests": reqs, "actions": acts}

###############################################################################
# FASTAPI APP
###############################################################################
app = FastAPI(
    title="QWen3 LLM Endpoint",
    description="QWen3 language model for robot command conversion",
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
    """Load model on startup"""
    try:
        logger.info("Starting QWen3 endpoint...")
        model_manager.load_model()
        logger.info("QWen3 endpoint ready")
    except Exception as e:
        logger.error(f"Failed to start QWen3 endpoint: {str(e)}")

###############################################################################
# ENDPOINTS
###############################################################################
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Synchronous chat endpoint"""
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        history = get_history(request.session_id)
        history.append({"role": "user", "content": request.message})
        
        # Get system prompt based on instructions
        system_prompt = get_system_prompt(request.instructions, request.custom_instructions)
        
        # Build messages
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        
        prompt_text = model_manager.tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = model_manager.tokenizer([prompt_text], return_tensors="pt").to(model_manager.model.device)
        
        with torch.inference_mode():
            gen_ids = model_manager.model.generate(
                **inputs, 
                max_new_tokens=1024,
                temperature=0.7, 
                top_p=0.95, 
                do_sample=True
            )
        
        resp = model_manager.tokenizer.decode(
            gen_ids[0][len(inputs.input_ids[0]):],
            skip_special_tokens=True
        ).strip()
        
        history.append({"role": "assistant", "content": resp})
        
        block = parse_action_block(resp)
        if block:
            _session_history[request.session_id] = []  # Clear memory after action
        
        # Clean up
        del inputs, gen_ids
        torch.cuda.empty_cache()
        
        return ChatResponse(result=resp, parsed=block)
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat-stream")
async def chat_stream_endpoint(request: ChatRequest):
    """Streaming chat endpoint"""
    if not model_manager.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    try:
        history = get_history(request.session_id)
        history.append({"role": "user", "content": request.message})
        
        system_prompt = get_system_prompt(request.instructions, request.custom_instructions)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history)
        
        prompt_text = model_manager.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        
        model_inputs = model_manager.tokenizer([prompt_text], return_tensors="pt").to(model_manager.model.device)
        
        # CRITICAL: Configure for single token streaming
        streamer = TextIteratorStreamer(
            model_manager.tokenizer, 
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=None
        )
        
        def _worker():
            with torch.inference_mode():
                model_manager.model.generate(
                    **model_inputs,
                    streamer=streamer,
                    max_new_tokens=1024,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    # IMPORTANT: These ensure single token output
                    pad_token_id=model_manager.tokenizer.eos_token_id,
                    eos_token_id=model_manager.tokenizer.eos_token_id,
                )

        threading.Thread(target=_worker, daemon=True).start()
        
        collected = ""
        action_sent = False
        
        def event_stream():
            nonlocal collected, action_sent
            try:
                for token in streamer:
                    collected += token
                    # Yield each token immediately
                    yield f"data: {token}\n\n"
                    
                    if not action_sent and "[ACTION]" in collected:
                        blk = parse_action_block(collected)
                        if blk:
                            yield f"data: \n{json.dumps(blk)}\n\n"
                            _session_history[request.session_id] = []
                            action_sent = True
                            
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
        
        def finish_chat():
            history.append({"role": "assistant", "content": collected.strip()})
        
        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "X-Accel-Buffering": "no",  # CRITICAL: Disable nginx buffering
            },
            background=finish_chat
        )
        
    except Exception as e:
        logger.error(f"Stream endpoint error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_manager.is_ready() else "loading",
        "model": "qwen3",
        "model_path": DEFAULT_LLM_NAME,
        "timestamp": time.time()
    }

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    uvicorn.run(
        "qwen3_endpoint:app",  
        host="0.0.0.0",
        port=QWEN3_PORT,
        log_level="info",
        reload=False
    )