"""
Scorer API Service for Alignment Task

This service runs independently and provides scoring functionality via REST API.
It loads reward and cost models and scores prompt-response pairs.
"""

import os
import sys
import argparse
from typing import List, Tuple
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Ensure local safe-rlhf package is importable (same as classify_reward_cost_unified.py)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'safe-rlhf'))
from transformers import AutoTokenizer
from safe_rlhf.models import AutoModelForScore

# Request/Response models for API
class ScoreRequest(BaseModel):
    prompts: List[str]
    responses: List[str]

class ScoreResponse(BaseModel):
    rewards: List[float]
    costs: List[float]

# Global variables for models (loaded once at startup)
reward_model = None
reward_tokenizer = None
cost_model = None
cost_tokenizer = None
device = None
max_length = None
r_eos = None
c_eos = None

def build_text(prompt: str, response: str, eos: str) -> str:
    """Build text in the format expected by the models (same as classify_reward_cost_unified.py)"""
    conv = f"BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}"
    return conv if conv.endswith(eos) else conv + eos

# Create FastAPI app
app = FastAPI(title="Alignment Scorer API", version="1.0.0")

@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    global reward_model, reward_tokenizer, cost_model, cost_tokenizer, device, max_length, r_eos, c_eos
    
    # Get configuration from environment variables
    reward_model_name = os.environ.get("REWARD_MODEL", "PKU-Alignment/beaver-7b-v1.0-reward")
    cost_model_name = os.environ.get("COST_MODEL", "PKU-Alignment/beaver-7b-v1.0-cost")
    max_length = int(os.environ.get("MAX_LENGTH", "2048"))
    use_bf16 = os.environ.get("USE_BF16", "false").lower() == "true"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading models on device: {device}")
    
    # Determine dtype (same as classify_reward_cost_unified.py)
    dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float16
    
    # Load models using the same pattern as classify_reward_cost_unified.py
    print(f"Loading reward model: {reward_model_name}")
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    r_eos = reward_tokenizer.eos_token or '</s>'
    reward_model = AutoModelForScore.from_pretrained(reward_model_name, torch_dtype=dtype).to(device).eval()
    
    print(f"Loading cost model: {cost_model_name}")
    cost_tokenizer = AutoTokenizer.from_pretrained(cost_model_name)
    c_eos = cost_tokenizer.eos_token or '</s>'
    cost_model = AutoModelForScore.from_pretrained(cost_model_name, torch_dtype=dtype).to(device).eval()
    
    print("Models loaded successfully!")

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "device": str(device)}

@app.post("/score", response_model=ScoreResponse)
async def score_responses(request: ScoreRequest):
    """Score prompt-response pairs."""
    if len(request.prompts) != len(request.responses):
        raise HTTPException(status_code=400, detail="Number of prompts and responses must match")
    
    if len(request.prompts) == 0:
        return ScoreResponse(rewards=[], costs=[])
    
    try:
        rewards = []
        costs = []
        
        with torch.no_grad():
            for prompt, response in zip(request.prompts, request.responses):
                # Build text in the same format as classify_reward_cost_unified.py
                r_text = build_text(prompt, response, r_eos)
                c_text = build_text(prompt, response, c_eos)
                
                # Score with reward model
                reward_input = reward_tokenizer(
                    r_text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(device)
                
                reward_output = reward_model(**reward_input)
                reward_score = reward_output.end_scores.squeeze(-1).float().item()
                rewards.append(float(reward_score))
                
                # Score with cost model
                cost_input = cost_tokenizer(
                    c_text,
                    return_tensors="pt",
                    max_length=max_length,
                    truncation=True,
                    padding=True
                ).to(device)
                
                cost_output = cost_model(**cost_input)
                cost_score = cost_output.end_scores.squeeze(-1).float().item()
                costs.append(float(cost_score))
        
        return ScoreResponse(rewards=rewards, costs=costs)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring responses: {str(e)}")

@app.post("/score_batch", response_model=ScoreResponse)
async def score_responses_batch(request: ScoreRequest):
    """Score prompt-response pairs in batch (more efficient for large batches)."""
    if len(request.prompts) != len(request.responses):
        raise HTTPException(status_code=400, detail="Number of prompts and responses must match")
    
    if len(request.prompts) == 0:
        return ScoreResponse(rewards=[], costs=[])
    
    try:
        # Build texts in the same format as classify_reward_cost_unified.py
        r_texts = [build_text(p, r, r_eos) for p, r in zip(request.prompts, request.responses)]
        c_texts = [build_text(p, r, c_eos) for p, r in zip(request.prompts, request.responses)]
        
        with torch.no_grad():
            # Batch process with reward model
            reward_inputs = reward_tokenizer(
                r_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(device)
            
            reward_outputs = reward_model(**reward_inputs)
            reward_scores = reward_outputs.end_scores.squeeze(-1).cpu().tolist()
            
            # Ensure it's a list even for single item
            if not isinstance(reward_scores, list):
                reward_scores = [reward_scores]
            
            # Batch process with cost model
            cost_inputs = cost_tokenizer(
                c_texts,
                return_tensors="pt",
                max_length=max_length,
                truncation=True,
                padding=True
            ).to(device)
            
            cost_outputs = cost_model(**cost_inputs)
            cost_scores = cost_outputs.end_scores.squeeze(-1).cpu().tolist()
            
            # Ensure it's a list even for single item
            if not isinstance(cost_scores, list):
                cost_scores = [cost_scores]
        
        return ScoreResponse(rewards=reward_scores, costs=cost_scores)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error scoring responses: {str(e)}")

if __name__ == "__main__":
    # Run the service
    uvicorn.run(app, host="0.0.0.0", port=8000)
