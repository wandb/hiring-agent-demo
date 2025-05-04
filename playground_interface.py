import time
import json
import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import weave
from weave.trace import urls
from typing import Dict, Any, Optional, Union, List
import openai
from dotenv import load_dotenv

# Import directly from server.py
from server import (
    HiringAgent, GraphState, DecisionScorer, ReasonScorer, 
    pre_process_eval, context_prompt, guardrail_prompt,
    extract_offer_prompt, extract_application_prompt, compare_offer_application_prompt
)

# Load environment variables from .env file
load_dotenv("utils/.env")

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define request and response models (based on standard request from playground)
class GenerateOptions(BaseModel):
    temperature: float = 1.0
    top_p: float = 1.0
    num_predict: int = 2048
    repeat_penalty: float = 0.0

class GenerateRequest(BaseModel):
    model: str
    prompt: str
    options: GenerateOptions
    stream: bool = False

# Default W&B settings
wandb_entity = os.environ.get("WANDB_ENTITY", "wandb-smle")
wandb_project = os.environ.get("WANDB_PROJECT", "e2e-hiring-assistant-test")

# Initialize OpenAI client
openai_client = openai.OpenAI()

# Initialize Weave client once globally
weave_client = weave.init(f"{wandb_entity}/{wandb_project}")

# Use newer model for intent detection
INTENT_MODEL = "gpt-4o"

@weave.op
def detect_intent(prompt: str) -> Dict[str, Any]:
    """
    Detect user intent from the prompt
    
    Returns a dictionary with intent and parameters
    """
    system_message = """
    Your task is to detect the user's intent from their message. Classify into one of these categories:
    1. get_prompt - User wants to see a prompt
    2. update_prompt - User wants to update a prompt
    3. run_evaluation - User wants to run an evaluation
    4. other - Any other request
    
    For get_prompt, identify which prompt they want to see
    For update_prompt, identify which prompt they want to update and the new content
    For run_evaluation, identify any specific parameters mentioned
    
    Return a JSON object with:
    - intent: the intent category
    - prompt_name: (for get_prompt and update_prompt) the name of the prompt 
    - prompt_content: (for update_prompt) the new prompt content
    - model: (for run_evaluation) the model to use
    - dataset: (for run_evaluation) the dataset to use
    """
    
    try:
        response = openai_client.chat.completions.create(
            model=INTENT_MODEL,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": prompt}
            ]
        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        print(f"Error detecting intent: {str(e)}")
        return {"intent": "other", "error": str(e)}

@weave.op
def get_prompt(prompt_name: str) -> Dict[str, Any]:
    """Get the latest version of a prompt"""
    try:
        latest_prompt = weave.ref(f"weave:///{wandb_entity}/{wandb_project}/object/{prompt_name}:latest").get()
        
        # Handle different prompt object types
        content = latest_prompt.content if hasattr(latest_prompt, 'content') else latest_prompt
        
        prompt_url = urls.object_version_path(
            wandb_entity,
            wandb_project,
            prompt_name,
            "latest"
        )
        
        return {
            "status": "success",
            "prompt_name": prompt_name,
            "prompt_content": content,
            "prompt_url": prompt_url
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }
    
@weave.op
def update_prompt(prompt_name: str, prompt_content: str) -> Dict[str, Any]:
    """Update a prompt with new content"""
    try:
        weave_prompt = weave.StringPrompt(prompt_content)
        ref = weave.publish(weave_prompt, name=prompt_name)
        
        prompt_url = urls.object_version_path(
            ref.entity,
            ref.project,
            ref.name,
            ref.digest
        )
        
        return {
            "status": "success",
            "prompt_name": prompt_name,
            "prompt_url": prompt_url,
            "message": f"Successfully updated {prompt_name}"
        }
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

@weave.op
async def run_evaluation(model_name: str = "HiringAgent:latest", dataset_name: str = "evaluation_dataset:latest") -> Dict[str, Any]:
    """Run an actual evaluation using the specified model and dataset"""
    try:
        # Get dataset reference (assume format is already "name:version")
        dataset_ref = f"weave:///{wandb_entity}/{wandb_project}/object/{dataset_name}"
        dataset = weave.ref(dataset_ref).get()
        
        # Get model reference (assume format is already "name:version")
        model_ref = f"weave:///{wandb_entity}/{wandb_project}/object/{model_name}"
        model = weave.ref(model_ref).get()
        
        # Extract arguments from the model
        extraction_model = model.extraction_model
        comparison_model = model.comparison_model
        guardrail_model = model.guardrail_model
        hitl_always_on = model.hitl_always_on
        disable_expert_review = getattr(model, 'disable_expert_review', False)
        
        # Create a new HiringAgent with the extracted arguments
        hiring_agent = HiringAgent(
            extraction_model=extraction_model,
            comparison_model=comparison_model,
            guardrail_model=guardrail_model,
            hitl_always_on=hitl_always_on,
            disable_expert_review=disable_expert_review,
            wandb_entity=wandb_entity,
            wandb_project=wandb_project
        )
        
        # Set up the evaluation
        benchmark = weave.Evaluation(
            dataset=dataset,
            scorers=[DecisionScorer(), ReasonScorer(model_id='gpt-4o-mini')],
            preprocess_model_input=pre_process_eval,
            trials=1  # Just one trial for now
        )
        
        try:
            # Properly await the evaluation instead of using asyncio.run()
            results = await benchmark.evaluate(hiring_agent)
            
            # Get the run URL (simplified)
            run_url = f"https://wandb.ai/{wandb_entity}/{wandb_project}/runs/latest"
            
            return {
                "status": "success",
                "message": f"Evaluation completed with model {model_name} on dataset {dataset_name}",
                "results": results,
                "eval_url": run_url
            }
        except Exception as eval_error:
            return {
                "status": "failed",
                "error": f"Evaluation error: {str(eval_error)}"
            }
            
    except Exception as e:
        return {
            "status": "failed",
            "error": str(e)
        }

# Define the generate endpoint
@app.post("/api/generate")
async def generate(request: GenerateRequest):
    # Extract intent from prompt
    intent_data = detect_intent(request.prompt)
    intent = intent_data.get("intent", "other")
    
    # Check for error in intent detection
    if "error" in intent_data and intent == "other":
        return {
            "model": request.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
            "response": json.dumps({
                "status": "error",
                "message": f"Error detecting intent: {intent_data['error']}"
            }),
            "done": True,
            "done_reason": "error"
        }
    
    response_data = {}
    
    # Route to appropriate handler based on intent
    if intent == "get_prompt":
        prompt_name = intent_data.get("prompt_name")
        response_data = get_prompt(prompt_name)
    elif intent == "update_prompt":
        prompt_name = intent_data.get("prompt_name")
        prompt_content = intent_data.get("prompt_content")
        response_data = update_prompt(prompt_name, prompt_content)
    elif intent == "run_evaluation":
        model_name = intent_data.get("model", "gpt-4o")
        dataset = intent_data.get("dataset", "evaluation_dataset")
        response_data = await run_evaluation(model_name, dataset)
    else:
        # Default response for other intents
        response_data = {
            "status": "info",
            "message": "I can help with prompt management and evaluations. Try asking me to show a prompt, update a prompt, or run an evaluation."
        }
    
    # Format response in the expected format
    response = {
        "model": request.model,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
        "response": json.dumps(response_data, indent=2),
        "done": True,
        "done_reason": "stop",
        "context": [
            106, 1645, 108, 6176, 1479, 235292, 108, 2045, 708, 671, 16481, 20409, 
            6869, 577, 5422, 6211, 731, 9151, 3110, 235269, 66004, 235269, 578, 
            10055, 15641, 235265, 109, 6176, 4926, 235292, 108, 2151, 235269, 1212, 
            798, 692, 749, 235336, 109, 107, 108, 106, 2516, 108, 116546, 43624, 
            25957, 235482, 140, 235307, 235248, 97781, 236931, 146909, 577, 577, 
            2045, 731, 2151, 2151
        ],
        "total_duration": 5148599166,
        "load_duration": 3405613458,
        "prompt_eval_count": 45,
        "prompt_eval_duration": 1327218375,
        "eval_count": 17,
        "eval_duration": 414738458
    }
    
    return response

if __name__ == "__main__":
    # Run the server
    uvicorn.run(app, host="0.0.0.0", port=8000)