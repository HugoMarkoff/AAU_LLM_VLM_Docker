###############################################################################
# robopoint.py – RoboPoint VLM endpoint for coordinate detection
# 
# USAGE:
# POST /predict
# {
#   "instructions": "default",  // "default", "none", or "custom"
#   "message": "Find the red cup",
#   "image": "base64_encoded_image_string",
#   "custom_instructions": "Custom instructions here" // only if instructions="custom"
# }
###############################################################################
import io
import base64
import os
import threading
import torch
import logging
import time
from typing import Optional
from pydantic import BaseModel

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import uvicorn

# RoboPoint imports
try:
    from robopoint.constants import (
        IMAGE_TOKEN_INDEX,
        DEFAULT_IMAGE_TOKEN,
        DEFAULT_IM_START_TOKEN,
        DEFAULT_IM_END_TOKEN
    )
    from robopoint.conversation import conv_templates
    from robopoint.model.builder import load_pretrained_model
    from robopoint.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
    ROBOPOINT_AVAILABLE = True
except ImportError:
    ROBOPOINT_AVAILABLE = False

###############################################################################
# CONFIGURATION
###############################################################################
ROBOPOINT_PORT = 8002

# Path to your RoboPoint model (from your original code)
MODEL_PATH = "/app/Models/robopoint-v1-vicuna-v1.5-13b"
MODEL_BASE_PATH = None
CONV_MODE = "llava_v1"

# Default instructions (when instructions="default")
DEFAULT_INSTRUCTIONS = (
    "Your answer should be formatted as a list of tuples, i.e. [(x1, y1), (x2, y2), ...], "
    "where each tuple contains the x and y coordinates of a point satisfying the conditions above. "
    "The coordinates should be between 0 and 1, indicating the normalized pixel locations of the points in the image."
)

TEMPERATURE = 0.2
TOP_P = 0.9
NUM_BEAMS = 1
MAX_NEW_TOKENS = 1024

###############################################################################
# LOGGING SETUP
###############################################################################
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/robopoint.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('robopoint_endpoint')

###############################################################################
# REQUEST SCHEMAS
###############################################################################
class PredictionRequest(BaseModel):
    instructions: str = "default"  # "default", "none", or "custom" 
    message: str
    image: str  # base64 encoded
    custom_instructions: Optional[str] = None

class PredictionResponse(BaseModel):
    result: str

###############################################################################
# MODEL MANAGEMENT
###############################################################################
class RoboPointModelManager:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.context_len = None
        self.model_loaded = False
        self.loading_lock = threading.Lock()
    def load_model(self):
        """Load RoboPoint vision-language model"""
        with self.loading_lock:
            if self.model_loaded:
                return
            logger.info(f"Loading RoboPoint model from {MODEL_PATH}...")
            start_time = time.time()
            try:
                if not ROBOPOINT_AVAILABLE:
                    logger.warning("RoboPoint package not available, using mock model")
                    self.model_loaded = True
                    return
                if not os.path.exists(MODEL_PATH):
                    raise FileNotFoundError(f"Model path {MODEL_PATH} does not exist")  
                model_name = get_model_name_from_path(MODEL_PATH)
                self.tokenizer, self.model, self.image_processor, self.context_len = load_pretrained_model(
                    model_path=MODEL_PATH,
                    model_base=MODEL_BASE_PATH,
                    model_name=model_name
                )
                self.model.cuda().eval()
                # Check vision tower
                vision_tower = getattr(self.model.model, "vision_tower", None)
                if vision_tower is None or (isinstance(vision_tower, list) and all(v is None for v in vision_tower)):
                    logger.warning("vision_tower is None (not fully multimodal?)")
                else:
                    logger.info("vision_tower present")
                load_time = time.time() - start_time
                logger.info(f"✓ RoboPoint model loaded successfully in {load_time:.2f}s")
                self.model_loaded = True
            except Exception as e:
                logger.error(f"✗ Failed to load RoboPoint model: {str(e)}")
                raise e

    def is_ready(self):
        return self.model_loaded

# Global model manager
model_manager = RoboPointModelManager()

###############################################################################
# INSTRUCTION PROCESSING
###############################################################################
def get_instructions(instruction_type: str, custom_instructions: Optional[str] = None) -> str:
    """Get instructions based on type"""
    if instruction_type == "default":
        return DEFAULT_INSTRUCTIONS
    elif instruction_type == "none":
        return ""  # No additional instructions
    elif instruction_type == "custom":
        if custom_instructions:
            return custom_instructions
        else:
            # Fallback to default if custom requested but not provided
            return DEFAULT_INSTRUCTIONS
    else:
        # Invalid instruction type, fallback to default
        return DEFAULT_INSTRUCTIONS

###############################################################################
# IMAGE PROCESSING
###############################################################################
def process_base64_image(base64_string: str) -> Image.Image:
    """Process base64 encoded image"""
    try:
        image_bytes = base64.b64decode(base64_string)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return pil_image
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError("Invalid image data")

###############################################################################
# FASTAPI APP
###############################################################################
app = FastAPI(
    title="RoboPoint VLM Endpoint",
    description="RoboPoint vision-language model for coordinate detection",
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
        logger.info("Starting RoboPoint endpoint...")
        model_manager.load_model()
        logger.info("RoboPoint endpoint ready")
    except Exception as e:
        logger.error(f"Failed to start RoboPoint endpoint: {str(e)}")

###############################################################################
# ENDPOINTS
###############################################################################
@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(request: PredictionRequest):
    """RoboPoint prediction endpoint for coordinate detection"""
    if not model_manager.is_ready():
        if ROBOPOINT_AVAILABLE:
            raise HTTPException(status_code=503, detail="Model not loaded yet")
        else:
            # Mock response when RoboPoint is not available
            return PredictionResponse(result="[(0.5, 0.5), (0.3, 0.7)]")
    
    try:
        # Process image
        pil_image = process_base64_image(request.image)
        
        # Get instructions based on type
        instructions = get_instructions(request.instructions, request.custom_instructions)
        
        # Combine user message with instructions
        if instructions:
            user_input = request.message.strip() + "\n" + instructions
        else:
            user_input = request.message.strip()
        
        # Insert image token if missing
        if DEFAULT_IMAGE_TOKEN not in user_input:
            if getattr(model_manager.model.config, 'mm_use_im_start_end', False):
                user_input = (
                    DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN +
                    "\n" + user_input
                )
            else:
                user_input = DEFAULT_IMAGE_TOKEN + "\n" + user_input
        
        # Build conversation
        conv = conv_templates[CONV_MODE].copy()
        conv.append_message(conv.roles[0], user_input)
        conv.append_message(conv.roles[1], None)
        prompt_text = conv.get_prompt()
        
        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_text,
            model_manager.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).cuda()
        
        # Process image
        image_tensor = process_images([pil_image], model_manager.image_processor, model_manager.model.config)[0]
        image_tensor = image_tensor.unsqueeze(0).half().cuda()
        
        # Generate
        with torch.inference_mode():
            outputs = model_manager.model.generate(
                input_ids,
                images=image_tensor,
                image_sizes=[pil_image.size],
                do_sample=True,
                temperature=TEMPERATURE,
                top_p=TOP_P,
                num_beams=NUM_BEAMS,
                max_new_tokens=MAX_NEW_TOKENS,
                use_cache=True
            )
        
        # Decode
        result_text = model_manager.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        
        # Clean up
        del input_ids, image_tensor, outputs
        torch.cuda.empty_cache()
        
        return PredictionResponse(result=result_text)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model_manager.is_ready() else "loading",
        "model": "robopoint",
        "model_path": MODEL_PATH,
        "robopoint_available": ROBOPOINT_AVAILABLE,
        "timestamp": time.time()
    }

###############################################################################
# MAIN
###############################################################################
if __name__ == "__main__":
    uvicorn.run(
        "robopoint:app",
        host="0.0.0.0",
        port=ROBOPOINT_PORT,
        log_level="info",
        reload=False
    )