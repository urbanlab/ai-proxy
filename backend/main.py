from fastapi import FastAPI, HTTPException, Depends, Security, status, File, UploadFile, Form
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from codecarbon import OfflineEmissionsTracker
import tempfile
import io
import time
import json
import yaml
import os
import base64
from lib.types import ChatCompletionRequest, EmbeddingInput, SpeechRequest, Message, MessageContent
from lib.utils import estimate_tokens, extract_tokens_from_response
from typing import Optional, List, Dict, Any, AsyncGenerator, Union
import aiohttp
import lib.db

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

def validate_vision_request(model_config: Dict[str, Any], messages: List[Message]):
    """Validate that vision requests are only made to vision-enabled models"""
    has_images = False
    
    for message in messages:
        if isinstance(message.content, list):
            for content_item in message.content:
                if content_item.type == "image_url":
                    has_images = True
                    break
        if has_images:
            break
    
    if has_images and not model_config['params'].get('vision', False):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model {model_config['model_name']} does not support vision/image inputs"
        )
    
    return has_images

def validate_image_content(content_item: MessageContent):
    """Validate image content in messages"""
    if content_item.type == "image_url" and content_item.image_url:
        url = content_item.image_url.url
        
        # Check if it's a base64 image
        if url.startswith("data:image/"):
            try:
                # Extract base64 data
                header, data = url.split(",", 1)
                base64.b64decode(data)
                return True
            except Exception as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid base64 image data: {str(e)}"
                )
        
        # Check if it's a URL (optional - you might want to disable this for security)
        elif url.startswith(("http://", "https://")):
            return True
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image URL must be either a base64 data URL or HTTP(S) URL"
            )
    
    return False

def estimate_tokens_with_vision(messages: List[Message]) -> int:
    """Estimate tokens for messages that may contain images"""
    total_tokens = 0
    
    for message in messages:
        if isinstance(message.content, str):
            # Simple text message
            total_tokens += estimate_tokens(message.content)
        elif isinstance(message.content, list):
            # Multimodal message
            for content_item in message.content:
                if content_item.type == "text" and content_item.text:
                    total_tokens += estimate_tokens(content_item.text)
                elif content_item.type == "image_url":
                    # Images typically cost more tokens - this is a rough estimate
                    # Different models have different token costs for images
                    total_tokens += 1000  # Approximate - adjust based on your models
    
    return total_tokens

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
    
    # Validate vision support
    has_images = validate_vision_request(model_config, request.messages)
    
    # Validate image content if present
    if has_images:
        for message in request.messages:
            if isinstance(message.content, list):
                for content_item in message.content:
                    if content_item.type == "image_url":
                        validate_image_content(content_item)
    
    request_data = request.dict(by_alias=True)
    
    # FIX: Use the actual model name from config
    request_data["model"] = model_config['params']['model']  # Maps "devstral" to "devstral:24b"
    
    if model_config['params'].get('drop_params'):
        # For vision models, keep more parameters
        if has_images:
            request_data = {
                "model": request_data["model"],
                "messages": request_data["messages"],
                "stream": request_data.get("stream", False),
                "max_tokens": request_data.get("max_tokens"),
                "temperature": request_data.get("temperature")
            }
        else:
            # Regular text-only request
            request_data = {
                "model": request_data["model"],
                "messages": request_data["messages"],
                "stream": request_data.get("stream", False)
            }
    
    # Handle max_input_tokens for vision models differently
    if model_config['params'].get('max_input_tokens') and not has_images:
        # Only truncate text-only messages
        total_tokens = 0
        for msg in request_data['messages']:
            if isinstance(msg.get('content'), str):
                total_tokens += len(msg['content'].split())
        
        while total_tokens > model_config['params']['max_input_tokens'] and len(request_data['messages']) > 1:
            removed_msg = request_data['messages'].pop(0)
            if isinstance(removed_msg.get('content'), str):
                total_tokens -= len(removed_msg['content'].split())
    
    # Start timing and CO2 tracking
    start_time = time.time()
    tracker = OfflineEmissionsTracker(country_iso_code="FRA")
    tracker.start()
    
    # Estimate input tokens with vision support
    estimated_input_tokens = estimate_tokens_with_vision(request.messages)
    
    if request.stream:
        # streaming response
        collected_response = ""
        collected_chunks = []  # Store all chunks for logging
        
        async def event_generator():
            nonlocal collected_response, collected_chunks
            try:
                async for chunk in fetch_chat_completion_stream(model_config, request_data):
                    # Store the raw chunk
                    collected_chunks.append(chunk)
                    
                    # Try to extract content from streaming chunks
                    if chunk.startswith("data: ") and not chunk.strip().endswith("[DONE]"):
                        try:
                            chunk_data = json.loads(chunk[6:])  # Remove "data: " prefix
                            if 'choices' in chunk_data and len(chunk_data['choices']) > 0:
                                delta = chunk_data['choices'][0].get('delta', {})
                                if 'content' in delta:
                                    collected_response += delta['content']
                        except json.JSONDecodeError:
                            pass
                        except Exception:
                            pass
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
                
                # Calculate metrics
                response_time = time.time() - start_time
                output_tokens = estimate_tokens(collected_response) if collected_response else 0
                total_tokens = estimated_input_tokens + output_tokens
                
                # Create a structured response for logging
                if collected_response:
                    # Store the actual collected content
                    structured_response = {
                        "content": collected_response,
                        "streaming": True,
                        "chunks_count": len(collected_chunks),
                        "response_time": response_time,
                        "has_vision": has_images
                    }
                    response_for_db = json.dumps(structured_response)
                else:
                    # Fallback: store raw chunks if no content was extracted
                    response_for_db = json.dumps({
                        "raw_chunks": collected_chunks[:10],  # Limit to first 10 chunks to avoid huge logs
                        "streaming": True,
                        "chunks_count": len(collected_chunks),
                        "response_time": response_time,
                        "has_vision": has_images,
                        "note": "Content extraction failed, storing raw chunks"
                    })
                
                # Prepare messages for logging
                messages_for_log = []
                for msg in request.messages:
                    if isinstance(msg.content, str):
                        messages_for_log.append({"role": msg.role, "content": msg.content})
                    else:
                        # For vision messages, create a summary
                        content_summary = []
                        for item in msg.content:
                            if item.type == "text":
                                content_summary.append({"type": "text", "text": item.text})
                            elif item.type == "image_url":
                                content_summary.append({"type": "image_url", "summary": "Image provided"})
                        messages_for_log.append({"role": msg.role, "content": content_summary})
                
                # Log the request
                lib.db.create_request(
                    user_name=get_user_from_token(user_key['token']),
                    model_name=request.model,
                    prompt=json.dumps(messages_for_log),
                    response=response_for_db,
                    co2=tracker.stop(),
                    tokens_used=total_tokens,
                    response_latency=response_time
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
        response_time = time.time() - start_time
        
        # Extract tokens from response or estimate
        total_tokens = extract_tokens_from_response(response_data)
        if total_tokens == 0:
            # Estimate if not provided by API
            response_text = ""
            if 'choices' in response_data:
                response_text = " ".join([choice.get('message', {}).get('content', '') for choice in response_data['choices']])
            output_tokens = estimate_tokens(response_text)
            total_tokens = estimated_input_tokens + output_tokens
        
        # Prepare messages for logging
        messages_for_log = []
        for msg in request.messages:
            if isinstance(msg.content, str):
                messages_for_log.append({"role": msg.role, "content": msg.content})
            else:
                # For vision messages, create a summary
                content_summary = []
                for item in msg.content:
                    if item.type == "text":
                        content_summary.append({"type": "text", "text": item.text})
                    elif item.type == "image_url":
                        content_summary.append({"type": "image_url", "summary": "Image provided"})
                messages_for_log.append({"role": msg.role, "content": content_summary})
        
        # log the request in the database
        lib.db.create_request(
            user_name=get_user_from_token(user_key['token']),
            model_name=request.model,
            prompt=json.dumps(messages_for_log),
            response=json.dumps(response_data),
            co2=tracker.stop(),
            tokens_used=total_tokens,
            response_latency=response_time
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
    
    # Start timing and CO2 tracking
    start_time = time.time()
    tracker = OfflineEmissionsTracker(country_iso_code="FRA")
    tracker.start()
    
    # Estimate input tokens
    input_text = " ".join(request.input if isinstance(request.input, list) else [request.input])
    estimated_tokens = estimate_tokens(input_text)
    
    response_data = await fetch_embeddings(model_config, request_data)
    
    # Calculate response time
    response_time = time.time() - start_time
    
    # Extract tokens from response if available, otherwise use estimate
    total_tokens = extract_tokens_from_response(response_data)
    if total_tokens == 0:
        total_tokens = estimated_tokens
    
    # log the request in the database
    lib.db.create_request(
        user_name=get_user_from_token(user_key['token']),
        model_name=request.model,
        prompt=json.dumps(request.input),
        response=json.dumps(response_data),
        co2=tracker.stop(),
        tokens_used=total_tokens,
        response_latency=response_time
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
        # Start timing and CO2 tracking
        start_time = time.time()
        tracker = OfflineEmissionsTracker(country_iso_code="FRA")
        tracker.start()
        
        # Make the transcription request
        response_data = await fetch_transcription(model_config, temp_file_path, request_data)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Estimate tokens based on transcription output
        transcription_text = ""
        if isinstance(response_data, dict):
            transcription_text = response_data.get('text', '')
        elif isinstance(response_data, str):
            transcription_text = response_data
        
        estimated_tokens = estimate_tokens(transcription_text) if transcription_text else 0
        
        # Log the request in the database
        lib.db.create_request(
            user_name=get_user_from_token(user_key['token']),
            model_name=model,
            prompt=f"Audio transcription: {file.filename}",
            response=json.dumps(response_data),
            co2=tracker.stop(),
            tokens_used=estimated_tokens,
            response_latency=response_time
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
    
    # Start timing and CO2 tracking
    start_time = time.time()
    tracker = OfflineEmissionsTracker(country_iso_code="FRA")
    tracker.start()
    
    # Estimate input tokens
    estimated_tokens = estimate_tokens(request.input)
    
    try:
        # Make the speech request
        audio_data = await fetch_speech(model_config, request_data)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log the request in the database
        lib.db.create_request(
            user_name=get_user_from_token(user_key['token']),
            model_name=request.model,
            prompt=f"TTS ({request.voice}): {request.input[:100]}{'...' if len(request.input) > 100 else ''}",  # Include voice in log
            response=f"Audio generated ({len(audio_data)} bytes)",
            co2=tracker.stop(),
            tokens_used=estimated_tokens,
            response_latency=response_time
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
            model_info = {
                "id": model['model_name'],
                "object": "model",
                "created": int(time.time()),
                "owned_by": "organization",
                "permission": [],
            }
            
            # Add vision capability information
            if model['params'].get('vision', False):
                model_info["capabilities"] = ["text", "vision"]
            else:
                model_info["capabilities"] = ["text"]
                
            models.append(model_info)
    
    return {
        "object": "list",
        "data": models
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)