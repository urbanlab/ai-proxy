from fastapi import FastAPI, HTTPException, Depends, Security, status,File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from codecarbon import OfflineEmissionsTracker
import tempfile
import io
from lib.types import ChatCompletionRequest, EmbeddingInput, SpeechRequest
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
           return model  # Return the complete model config, not just params
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

# Add this function after your other fetch functions
async def fetch_speech(model_config: Dict[str, Any], request_data: Dict[str, Any]) -> bytes:
    url = f"{model_config['params']['api_base']}/audio/speech"
    headers = {
        "Content-Type": "application/json",
    }
    
    if model_config['params'].get('api_key') and model_config['params']['api_key'] != "no_token":
        headers["Authorization"] = f"Bearer {model_config['params']['api_key']}"
    
    print(f"Making speech request to: {url}")  # Debug log
    print(f"Request data: {request_data}")  # Debug log
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"Error response: {text}")  # Debug log
                raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
            
            print(f"Response status: {resp.status}")  # Debug log
            print(f"Response content type: {resp.headers.get('content-type')}")  # Debug log
            
            # Return the audio bytes
            return await resp.read()

# Add this function after your other fetch functions
async def fetch_transcription(model_config: Dict[str, Any], file_path: str, request_data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{model_config['params']['api_base']}/audio/transcriptions"
    headers = {}
    
    if model_config['params'].get('api_key') and model_config['params']['api_key'] != "no_token":
        headers["Authorization"] = f"Bearer {model_config['params']['api_key']}"
    
    # Prepare form data
    form_data = aiohttp.FormData()
    
    # Add the audio file
    with open(file_path, 'rb') as f:
        form_data.add_field('file', f, filename=os.path.basename(file_path), content_type='audio/mpeg')
        
        # Add other parameters
        for key, value in request_data.items():
            if value is not None:
                form_data.add_field(key, str(value))
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=headers, data=form_data) as resp:
                if resp.status != 200:
                    text = await resp.text()
                    raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
                
                # Handle different response formats
                if request_data.get('response_format') == 'text':
                    return {"text": await resp.text()}
                else:
                    return await resp.json()

# fetch chat completion from the model API streaming depends on verify_token
async def fetch_chat_completion_stream(model_config: Dict[str, Any], request_data: Dict[str, Any]) -> AsyncGenerator[str, None]:
   url = f"{model_config['params']['api_base']}/chat/completions"
   headers = {
       "Content-Type": "application/json",
   }
   if model_config['params'].get('api_key') and model_config['params']['api_key'] != "no_token":
       headers["Authorization"] = f"Bearer {model_config['params']['api_key']}"

   print(f"Making request to: {url}")  # Debug log
   print(f"Request data: {request_data}")  # Debug log
   
   async with aiohttp.ClientSession() as session:
       async with session.post(url, headers=headers, json=request_data) as resp:
           if resp.status != 200:
               text = await resp.text()
               print(f"Error response: {text}")  # Debug log
               raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
           
           print(f"Response status: {resp.status}")  # Debug log
           print(f"Response headers: {dict(resp.headers)}")  # Debug log
           
           async for line in resp.content:
               decoded_line = line.decode('utf-8')
               if decoded_line.strip():
                   yield decoded_line

# fetch chat completion from the model API non-streaming
async def fetch_chat_completion(model_config: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
   url = f"{model_config['params']['api_base']}/chat/completions"
   headers = {
       "Content-Type": "application/json",
   }
   if model_config['params'].get('api_key') and model_config['params']['api_key'] != "no_token":
       headers["Authorization"] = f"Bearer {model_config['params']['api_key']}"

   async with aiohttp.ClientSession() as session:
       async with session.post(url, headers=headers, json=request_data) as resp:
           if resp.status != 200:
               text = await resp.text()
               raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
           return await resp.json()

# Add this new function after your existing fetch functions
async def fetch_embeddings(model_config: Dict[str, Any], request_data: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{model_config['params']['api_base']}/embeddings"  # Note: /embeddings not /chat/completions
    headers = {
        "Content-Type": "application/json",
    }
    if model_config['params'].get('api_key') and model_config['params']['api_key'] != "no_token":
        headers["Authorization"] = f"Bearer {model_config['params']['api_key']}"
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
            return await resp.json()

# /chat/completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, user_key = Depends(verify_token)):
   model_config = get_model_config(request.model, user_key)
   request_data = request.dict(by_alias=True)
   
   # FIX: Use the actual model name from config
   request_data["model"] = model_config['params']['model']  # Maps "devstral" to "devstral:24b"
   
   if model_config['params'].get('drop_params'):
       # keep only model, messages, and stream
       request_data = {
           "model": request_data["model"],
           "messages": request_data["messages"],
           "stream": request_data.get("stream", False)  # Keep the stream parameter!
       }
   if model_config['params'].get('max_input_tokens'):
       # truncate messages to fit max_input_tokens
       total_tokens = sum(len(msg['content'].split()) for msg in request_data['messages'])
       while total_tokens > model_config['params']['max_input_tokens'] and len(request_data['messages']) > 1:
           removed_msg = request_data['messages'].pop(0)
           total_tokens -= len(removed_msg['content'].split())
   
   # start time for CO2 calculation
   tracker = OfflineEmissionsTracker(country_iso_code="FRA")
   tracker.start()
   
   if request.stream:
       # streaming response
       async def event_generator():
           try:
               async for chunk in fetch_chat_completion_stream(model_config, request_data):
                   yield chunk
           except Exception as e:
               # Yield error in SSE format
               error_data = {
                   "error": {
                       "message": str(e),
                       "type": "stream_error"
                   }
               }
               yield f"data: {json.dumps(error_data)}\n\n"
           finally:
               # Send the final [DONE] message
               yield "data: [DONE]\n\n"
               
               # Log the request
               lib.db.create_request(
                   user_name=get_user_from_token(user_key['token']),
                   model_name=request.model,
                   prompt=json.dumps([msg.dict() for msg in request.messages]),
                   response="streamed_response",
                   co2=tracker.stop()
               )
       
       return StreamingResponse(
           event_generator(),
           media_type="text/event-stream",
           headers={
               "Cache-Control": "no-cache",
               "Connection": "keep-alive",
               "X-Accel-Buffering": "no"  # Disable nginx buffering if applicable
           }
       )
   else:
       # non-streaming response
       response_data = await fetch_chat_completion(model_config, request_data)
       # log the request in the database
       lib.db.create_request(
           user_name=get_user_from_token(user_key['token']),
           model_name=request.model,
           prompt=json.dumps([msg.dict() for msg in request.messages]),
           response=json.dumps(response_data),
           co2=tracker.stop()
       )
       return response_data

# /embeddings endpoint
@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingInput, user_key = Depends(verify_token)):
    model_config = get_model_config(request.model, user_key)
    request_data = request.dict()
    
    request_data["model"] = model_config['params']['model']  # Maps "devstral" to "devstral:24b"
    
    if model_config['params'].get('drop_params'):
        # keep only model and input
        request_data = {
            "model": request_data["model"],
            "input": request_data["input"]
        }
    if model_config['params'].get('max_input_tokens'):
        # truncate input to fit max_input_tokens
        total_tokens = sum(len(text.split()) for text in request_data['input'])
        while total_tokens > model_config['params']['max_input_tokens'] and len(request_data['input']) > 1:
            removed_text = request_data['input'].pop(0)
            total_tokens -= len(removed_text.split())
    
    # start time for CO2 calculation
    tracker = OfflineEmissionsTracker(country_iso_code="FRA")
    tracker.start()
    
    response_data = await fetch_embeddings(model_config, request_data)
    
    # log the request in the database
    lib.db.create_request(
        user_name=get_user_from_token(user_key['token']),
        model_name=request.model,
        prompt=json.dumps(request.input),
        response=json.dumps(response_data),
        co2=tracker.stop()
    )
    
    return response_data

# /audio/transcriptions endpoint
@app.post("/v1/audio/transcriptions")
async def create_transcription(
    file: UploadFile = File(...),
    model: str = Form(...),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: Optional[str] = Form("json"),
    temperature: Optional[float] = Form(0),
    user_key = Depends(verify_token)
):
    # Validate file type
    allowed_extensions = {'.mp3', '.mp4', '.mpeg', '.mpga', '.m4a', '.wav', '.webm'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (e.g., max 25MB)
    max_file_size = 25 * 1024 * 1024  # 25MB
    file_content = await file.read()
    if len(file_content) > max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="File size exceeds 25MB limit"
        )
    
    model_config = get_model_config(model, user_key)
    
    # Create request data
    request_data = {
        "model": model_config['params']['model'],  # Use actual model name from config
        "language": language,
        "prompt": prompt,
        "response_format": response_format,
        "temperature": temperature
    }
    
    if model_config['params'].get('drop_params'):
        # Keep only essential parameters
        request_data = {
            "model": request_data["model"],
            "response_format": response_format or "json"
        }
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    
    try:
        # Start CO2 tracking
        tracker = OfflineEmissionsTracker(country_iso_code="FRA")
        tracker.start()
        
        # Make the transcription request
        response_data = await fetch_transcription(model_config, temp_file_path, request_data)
        
        # Log the request in the database
        lib.db.create_request(
            user_name=get_user_from_token(user_key['token']),
            model_name=model,
            prompt=f"Audio transcription: {file.filename}",
            response=json.dumps(response_data),
            co2=tracker.stop()
        )
        
        return response_data
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)

# /audio/speech endpoint
@app.post("/v1/audio/speech")
async def create_speech(
    request: SpeechRequest, 
    user_key = Depends(verify_token)
):
    # Validate input length (e.g., max 4096 characters for most TTS models)
    max_input_length = 4096
    if len(request.input) > max_input_length:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Input text exceeds maximum length of {max_input_length} characters"
        )
    
    # Voice validation removed - support custom models with any voice option
    
    # Validate response format
    valid_formats = ["mp3", "opus", "aac", "flac", "wav", "pcm"]
    if request.response_format not in valid_formats:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid response format. Valid options: {', '.join(valid_formats)}"
        )
    
    model_config = get_model_config(request.model, user_key)
    request_data = request.dict()
    
    # Use the actual model name from config
    request_data["model"] = model_config['params']['model']
    
    if model_config['params'].get('drop_params'):
        # Keep only essential parameters
        request_data = {
            "model": request_data["model"],
            "input": request_data["input"],
            "voice": request_data.get("voice", "fr_FR-upmc-medium")  # Keep voice parameter
        }
    
    # Start CO2 tracking
    tracker = OfflineEmissionsTracker(country_iso_code="FRA")
    tracker.start()
    
    try:
        # Make the speech request
        audio_data = await fetch_speech(model_config, request_data)
        
        # Log the request in the database
        lib.db.create_request(
            user_name=get_user_from_token(user_key['token']),
            model_name=request.model,
            prompt=f"TTS ({request.voice}): {request.input[:100]}{'...' if len(request.input) > 100 else ''}",  # Include voice in log
            response=f"Audio generated ({len(audio_data)} bytes)",
            co2=tracker.stop()
        )
        
        # Set appropriate content type based on format
        content_types = {
            "mp3": "audio/mpeg",
            "opus": "audio/opus",
            "aac": "audio/aac", 
            "flac": "audio/flac",
            "wav": "audio/wav",
            "pcm": "audio/pcm"
        }
        
        content_type = content_types.get(request.response_format, "audio/mpeg")
        
        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(audio_data),
            media_type=content_type,
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "Content-Length": str(len(audio_data))
            }
        )
        
    except Exception as e:
        tracker.stop()  # Stop tracker in case of error
        raise e

# list models endpoint
@app.get("/v1/models")
async def list_models(user_key = Depends(verify_token)):
   models = []
   for model in CONFIG['model_list']:
       if model['model_name'] in user_key['models']:
           models.append({
               "id": model['model_name'],
               "object": "model",
               "created": int(time.time()),
               "owned_by": "organization",
               "permission": [],
           })
   return {
       "object": "list",
       "data": models
   }

if __name__ == "__main__":
   import uvicorn
   uvicorn.run(app, host="0.0.0.0", port=8000)