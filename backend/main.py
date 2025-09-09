from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from codecarbon import OfflineEmissionsTracker
from typing import Optional, List, Dict, Any, AsyncGenerator
import aiohttp
import json
import time
import yaml
import lib.db
import os


app = FastAPI(
    title="LLM Proxy API",
    description="Proxy API for Large Language Models with authentication and rate limiting",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Load configuration
with open("/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)


def get_model_config(model_name: str, user_key: Dict[str, Any]) -> Dict[str, Any]:
    if model_name not in user_key['models']:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to the model is forbidden for this user",
        )
    for model in CONFIG['model_list']:
        if model['model_name'] == model_name:
            return model['params']
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Model not found",
    )


# verify user token and model access
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    for key in CONFIG['keys']:
        if key['token'] == token:
            return key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing token or insufficient permissions",
        headers={"WWW-Authenticate": "Bearer"},
    )

def get_user_from_token(token: str) -> Optional[str]:
    for key in CONFIG['keys']:
        if key['token'] == token:
            return key['name']
    return None

# chat completion request and response models
class UsageDetails(BaseModel):
    reasoning_tokens: Optional[int] = None
    accepted_prediction_tokens: Optional[int] = None
    rejected_prediction_tokens: Optional[int] = None

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_tokens_details: Optional[UsageDetails] = None
    completion_tokens_details: Optional[UsageDetails] = None
class Message(BaseModel):
    role: str
    content: str
class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    max_tokens: Optional[int] = Field(None, alias="max_tokens")
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    usage: Optional[Usage] = None
class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None
class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Optional[Usage] = None 

# fetch chat completion from the model API streaming depends on verify_token
async def fetch_chat_completion_stream(model_config: Dict[str, Any], request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
    url = f"{model_config['api_base']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if model_config.get('api_key') and model_config['api_key'] != "no_token":
        headers["Authorization"] = f"Bearer {model_config['api_key']}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
            async for line in resp.content:
                if line:
                    yield line.decode('utf-8')
# fetch chat completion from the model API non-streaming
async def fetch_chat_completion(model_config: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{model_config['api_base']}/chat/completions"
    headers = {
        "Content-Type": "application/json",
    }
    if model_config.get('api_key') and model_config['api_key'] != "no_token":
        headers["Authorization"] = f"Bearer {model_config['api_key']}"
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
            return await resp.json()


# /chat/completions endpoint
@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest, user_key = Depends(verify_token)):
    model_config = get_model_config(request.model, user_key)
    request_data = request.dict(by_alias=True)
    if model_config.get('drop_params'):
        # keep only model and messages
        request_data = {
            "model": request_data["model"],
            "messages": request_data["messages"]
        }
    if model_config.get('max_input_tokens'):
        # truncate messages to fit max_input_tokens
        total_tokens = sum(len(msg['content'].split()) for msg in request_data['messages'])
        while total_tokens > model_config['max_input_tokens'] and len(request_data['messages']) > 1:
            removed_msg = request_data['messages'].pop(0)
            total_tokens -= len(removed_msg['content'].split())
    
    # start time for CO2 calculation
    tracker = OfflineEmissionsTracker(country_iso_code="FRA")
    tracker.start()
    if request.stream:
        # streaming response
        async def stream_generator():
            async for chunk in fetch_chat_completion_stream(model_config, request_data):
                yield chunk
        return stream_generator()
    else:
        # non-streaming response
        response_data = await fetch_chat_completion(model_config, request_data)
        end_time = time.time()
        # log the request in the database
        co2_params = model_config.get('co2', {'watt': 0, 'gram_per_kwh': 0})
        lib.db.create_request(
            user_name=get_user_from_token(user_key['token']),
            model_name=request.model,
            prompt=json.dumps([msg.dict() for msg in request.messages]),
            response=json.dumps(response_data),
            co2=tracker.stop()
        )
        return response_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)