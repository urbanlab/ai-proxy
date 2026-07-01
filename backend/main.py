from fastapi import FastAPI, HTTPException, Depends, Request, status, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from prometheus_client import make_asgi_app
import tempfile
import io
import time
import json
import uuid
import yaml
import os
import base64
from typing import Optional, List, Dict, Any
import aiohttp
from lib.data_types import ChatCompletionRequest, EmbeddingInput, SpeechRequest, Message, MessageContent, AnthropicMessageRequest
from lib.openai import fetch_chat_completion, fetch_chat_completion_stream, fetch_embeddings, fetch_transcription, fetch_speech, fetch_chat_completion_failover, fetch_chat_completion_stream_failover
from lib.utils import estimate_tokens, extract_tokens_from_response, fetch_image_as_base64, message_to_string
from lib.auth import metrics_auth_middleware, verify_token, get_username_from_token, verify_auth
from lib.metric import log_metrics, log_error
from lib.cost import calculate_token_cost
from lib.anthropic import fetch_anthropic_messages, fetch_anthropic_messages_stream, extract_text_from_anthropic_messages
from lib.loadbalancer import LoadBalancer
from lib.providers import apply_provider_mapping
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
app.middleware("http")(metrics_auth_middleware)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Load configuration
with open("/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Load balancer: when a model_name is declared multiple times in model_list,
# requests are spread across the healthy endpoints in round-robin order.
load_balancer = LoadBalancer(
    CONFIG,
    interval=CONFIG.get("health_check_interval", 30),
    timeout=CONFIG.get("health_check_timeout", 5),
)


@app.on_event("startup")
async def _start_load_balancer():
    # Run an initial probe so health state is accurate before traffic arrives,
    # then keep checking in the background.
    await load_balancer.check_all()
    load_balancer.start()


def get_model_config(model_name: str, user_key: Dict[str, Any]) -> Dict[str, Any]:
    if model_name not in user_key['models']:
        log_error(user_key["name"],403)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to the model is forbidden for this user",
        )
    # Picks a healthy endpoint, load-balancing when several share this model_name.
    model = load_balancer.select(model_name)
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    return model  # Complete model config (same shape as a model_list entry)


def get_model_candidates(model_name: str, user_key: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Access-checked, failover-ordered endpoint list for a model_name.

    Same authorization as get_model_config, but returns every replica (healthy
    first, round-robined) so the caller can fail over on connection errors.
    """
    if model_name not in user_key['models']:
        log_error(user_key["name"], 403)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access to the model is forbidden for this user",
        )
    candidates = load_balancer.candidates(model_name)
    if not candidates:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Model not found",
        )
    return candidates



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
        
        # Allow HTTP(S) URLs without additional validation
        elif url.startswith(("http://", "https://")):
            return True
        
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image URL must be either a base64 data URL or HTTP(S) URL"
            )
    
    return False

# /chat/completions endpoint
@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, user_key = Depends(verify_auth)):
    # Failover-ordered replicas for this model. model_config holds the shared
    # metadata (cost, limits — identical across replicas); the actual upstream
    # call walks `candidates` and fails over on connection errors.
    candidates = get_model_candidates(request.model, user_key)
    model_config = candidates[0]
    input_tokens_nb = estimate_tokens(message_to_string(request.messages))
    cost_per_input_token = model_config["params"].get("cost_per_input_token", 0)
    cost_per_output_token = model_config["params"].get("cost_per_output_token", 0)
    cost_per_input = calculate_token_cost(cost_per_input_token, input_tokens_nb)

    
    # Validate vision support
    has_images = validate_vision_request(model_config, request.messages)
    
    # Validate and convert image content if present
    if has_images:
        for message in request.messages:
            if isinstance(message.content, list):
                for content_item in message.content:
                    if content_item.type == "image_url":
                        validate_image_content(content_item)
                        
                        # Convert HTTP(S) URLs to base64 for OpenAI
                        if content_item.image_url.url.startswith(("http://", "https://")):
                            content_item.image_url.url = await fetch_image_as_base64(content_item.image_url.url)

    request_data = request.model_dump(by_alias=True)

    # Convert to OpenAI format for vision messages
    if has_images:
        openai_messages = []
        for message in request.messages:
            if isinstance(message.content, list):
                # Convert to proper OpenAI format
                openai_content = []
                for item in message.content:
                    if item.type == "text":
                        openai_content.append({"type": "text", "text": item.text})
                    elif item.type == "image_url":
                        openai_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": item.image_url.url,
                                "detail": getattr(item.image_url, 'detail', 'auto')
                            }
                        })
                openai_messages.append({"role": message.role, "content": openai_content})
            else:
                openai_messages.append({"role": message.role, "content": message.content or ""})
        request_data["messages"] = openai_messages

    # Use the actual model name from config
    request_data["model"] = model_config['params']['model']

    # Translate OpenAI-standard params into each backend's native equivalents,
    # keyed on the model's `provider`. Must run before the drop_params whitelist
    # below so the native keys it emits (grammar/chat_template_kwargs/think) are
    # in the allowed list and survive.
    request_data = apply_provider_mapping(request_data, model_config)

    if model_config['params'].get('drop_params'):
        # Keep OpenAI-compatible parameters only, plus the provider-native keys
        # emitted by apply_provider_mapping (each is only ever set for the backend
        # that supports it, so allowing them globally is harmless for the others).
        allowed_params = ["model", "messages", "stream", "stream_options", "max_tokens", "temperature", "top_p", "n", "stop", "presence_penalty", "frequency_penalty", "user", "response_format", "tools", "tool_choice", "grammar", "chat_template_kwargs", "think"]
        request_data = {k: v for k, v in request_data.items() if k in allowed_params and v is not None}
        
        # Some models don't allow both temperature and top_p
        if 'temperature' in request_data and 'top_p' in request_data:
            # Remove top_p, keep temperature (or vice versa based on your preference)
            request_data.pop('top_p')
    else:
        # Even without the whitelist, never forward null-valued top-level params.
        # Clients (e.g. Cline) may omit optional fields like max_tokens, which
        # model_dump serializes as null — and some backends (llama.cpp) reject
        # `max_tokens: null` with "type must be number, but is null" instead of
        # treating it as absent.
        request_data = {k: v for k, v in request_data.items() if v is not None}

    # Enforce a configured input-token limit, if one is set for this model. When
    # `max_input_tokens` is present we reject over-limit requests with an
    # OpenAI-compatible error instead of silently truncating. When it is not set,
    # we do nothing and let the upstream enforce its own context window.
    max_input_tokens = model_config['params'].get('max_input_tokens')
    if max_input_tokens and input_tokens_nb > max_input_tokens:
        return JSONResponse(
            status_code=413,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Prompt is too long",
                },
            },
        )
    
    # Clean up messages to remove None values for Scaleway compatibility
    if "messages" in request_data:
        cleaned_messages = []
        for msg in request_data["messages"]:
            # Create clean message dict without None values
            clean_msg = {}
            for key, value in msg.items():
                if value is not None:
                    clean_msg[key] = value
            # Assistant tool_call messages may have content: null — keep it as empty string
            if clean_msg.get("role") == "assistant" and "content" not in clean_msg:
                clean_msg["content"] = ""
            cleaned_messages.append(clean_msg)
        request_data["messages"] = cleaned_messages

    # Start timing
    start_time = time.time()
    
    # Estimate input tokens with vision support
    estimated_input_tokens = estimate_tokens(request.messages[0].content if isinstance(request.messages[0].content, str) else "")
    if request.stream:
        # streaming response
        collected_response = ""
        collected_chunks = []  # Store all chunks for logging
        # OpenAI-compatible streaming usage: only emit our own usage chunk when the
        # client asked for it (stream_options.include_usage) AND the upstream did not
        # already stream one — otherwise we'd duplicate it.
        include_usage = bool(request.stream_options and request.stream_options.include_usage)
        upstream_usage = None  # real usage object from upstream, if it streams one
        stream_id = None
        stream_created = None
        stream_model = None

        async def event_generator():
            nonlocal collected_response, collected_chunks
            nonlocal upstream_usage, stream_id, stream_created, stream_model
            try:
                async for chunk in fetch_chat_completion_stream_failover(candidates, request_data, load_balancer.mark_unhealthy):
                    # Suppress the upstream terminator; we emit a single, well-ordered
                    # terminal sequence (optional usage chunk + [DONE]) ourselves below.
                    if chunk.strip().endswith("[DONE]"):
                        continue

                    # Store the raw chunk
                    collected_chunks.append(chunk)

                    # Inspect data chunks: capture stream identifiers, any upstream
                    # usage object, and accumulate streamed content for accounting.
                    if chunk.startswith("data: "):
                        try:
                            chunk_data = json.loads(chunk[6:])  # Remove "data: " prefix
                            if stream_id is None and chunk_data.get('id'):
                                stream_id = chunk_data['id']
                                stream_created = chunk_data.get('created')
                                stream_model = chunk_data.get('model')
                            if chunk_data.get('usage'):
                                upstream_usage = chunk_data['usage']
                            choices = chunk_data.get('choices') or []
                            if len(choices) > 0:
                                delta = choices[0].get('delta', {})
                                if delta.get('content'):
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
                # If the client requested usage and the upstream did not already
                # stream a usage object, emit an OpenAI-compatible usage chunk
                # (empty choices + usage) right before the terminator.
                if include_usage and upstream_usage is None:
                    completion_tokens = estimate_tokens(collected_response) if collected_response else 0
                    usage_chunk = {
                        "id": stream_id or f"chatcmpl-{int(start_time)}",
                        "object": "chat.completion.chunk",
                        "created": stream_created or int(start_time),
                        "model": stream_model or request.model,
                        "choices": [],
                        "usage": {
                            "prompt_tokens": input_tokens_nb,
                            "completion_tokens": completion_tokens,
                            "total_tokens": input_tokens_nb + completion_tokens,
                        },
                    }
                    yield f"data: {json.dumps(usage_chunk)}\n\n"

                # Send the final [DONE] message
                yield "data: [DONE]\n\n"

                # Calculate metrics — prefer the upstream's real token counts when it
                # streamed them, otherwise fall back to local estimates.
                response_time = time.time() - start_time
                if upstream_usage:
                    output_tokens_nb = upstream_usage.get('completion_tokens') or 0
                    total_tokens = upstream_usage.get('total_tokens') or (input_tokens_nb + output_tokens_nb)
                else:
                    output_tokens_nb = estimate_tokens(collected_response) if collected_response else 0
                    total_tokens = input_tokens_nb + output_tokens_nb
                cost_per_output = calculate_token_cost(cost_per_output_token, output_tokens_nb)
                # Log metrics
                log_metrics(
                    request.model,
                    get_username_from_token(user_key['token']),
                    total_tokens,
                    response_time,
                    cost_per_input,
                    cost_per_output
                )
            
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
                    if msg.content is None:
                        messages_for_log.append({"role": msg.role, "content": ""})
                    elif isinstance(msg.content, str):
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
                    user_name=get_username_from_token(user_key['token']),
                    model_name=request.model,
                    prompt=json.dumps(messages_for_log),
                    response=response_for_db,
                    co2=0,  # No CO2 tracking
                    tokens_used=total_tokens,
                    response_latency=response_time,
                    input_cost = cost_per_input,
                    output_cost = cost_per_output

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
        response_data, model_config = await fetch_chat_completion_failover(candidates, request_data, load_balancer.mark_unhealthy)
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
            if msg.content is None:
                messages_for_log.append({"role": msg.role, "content": ""})
            elif isinstance(msg.content, str):
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
        
        
         # Log metrics
         #
        output_tokens_nb = estimate_tokens(message_to_string(request.messages))  
        cost_per_output = calculate_token_cost(cost_per_output_token, output_tokens_nb)
        total_tokens = estimated_input_tokens + output_tokens_nb

        log_metrics(
            request.model,
            get_username_from_token(user_key['token']),
            total_tokens,
            response_time,
            cost_per_input,
            cost_per_output
        )
        # log the request in the database
        lib.db.create_request(
            user_name=get_username_from_token(user_key['token']),
            model_name=request.model,
            prompt=json.dumps(messages_for_log),
            response=json.dumps(response_data),
            co2=0,  # No CO2 tracking
            tokens_used=total_tokens,
            response_latency=response_time,
            input_cost = cost_per_input,
            output_cost = cost_per_output
        )
        return response_data

# /embeddings endpoint
@app.post("/v1/embeddings")
async def create_embedding(request: EmbeddingInput, user_key = Depends(verify_auth)):
    model_config = get_model_config(request.model, user_key)
    request_data = request.model_dump()
    
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
    
    # Start timing
    start_time = time.time()
    
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
    # Log metrics
    log_metrics(request.model, get_username_from_token(user_key['token']), total_tokens, response_time)
    # log the request in the database
    lib.db.create_request(
        user_name=get_username_from_token(user_key['token']),
        model_name=request.model,
        prompt=json.dumps(request.input),
        response=json.dumps(response_data),
        co2=0,  # No CO2 tracking
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
    user_key = Depends(verify_auth)
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
        # Start timing
        start_time = time.time()
        
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
        # Log metrics
        log_metrics(model, get_username_from_token(user_key['token']), estimated_tokens, response_time)
        # Log the request in the database
        lib.db.create_request(
            user_name=get_username_from_token(user_key['token']),
            model_name=model,
            prompt=f"Audio transcription: {file.filename}",
            response=json.dumps(response_data),
            co2=0,  # No CO2 tracking
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
    user_key = Depends(verify_auth)
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
    
    # Start timing
    start_time = time.time()
    
    # Estimate input tokens
    estimated_tokens = estimate_tokens(request.input)
    
    try:
        # Make the speech request
        audio_data = await fetch_speech(model_config, request_data)
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Log metrics
        log_metrics(request.model, get_username_from_token(user_key['token']), estimated_tokens, response_time)
        # Log the request in the database
        lib.db.create_request(
            user_name=get_username_from_token(user_key['token']),
            model_name=request.model,
            prompt=f"TTS ({request.voice}): {request.input[:100]}{'...' if len(request.input) > 100 else ''}",  # Include voice in log
            response=f"Audio generated ({len(audio_data)} bytes)",
            co2=0,  # No CO2 tracking
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
        raise e

# Anthropic-compatible /v1/messages endpoint — native passthrough
@app.post("/v1/messages")
async def anthropic_messages(request: AnthropicMessageRequest, raw_request: Request, user_key = Depends(verify_auth)):
    request_dict = request.model_dump(exclude_none=True)

    # Ensure all messages have a content field (assistant tool_use messages may have content: null)
    for msg in request_dict.get("messages", []):
        if "content" not in msg:
            msg["content"] = []

    model_name = request.model
    stream = request.stream

    model_config = get_model_config(model_name, user_key)

    cost_per_input_token = model_config["params"].get("cost_per_input_token", 0)
    cost_per_output_token = model_config["params"].get("cost_per_output_token", 0)

    # Token / cost estimation for input
    input_text = extract_text_from_anthropic_messages(
        request_dict.get("messages", []), request_dict.get("system", None)
    )
    input_tokens_nb = estimate_tokens(input_text)
    cost_per_input = calculate_token_cost(cost_per_input_token, input_tokens_nb)

    # Enforce a configured input-token limit, if one is set for this model. When
    # `max_input_tokens` is present we reject over-limit requests; otherwise we let
    # the upstream enforce its own context window.
    max_input_tokens = model_config["params"].get("max_input_tokens")
    if max_input_tokens and input_tokens_nb > max_input_tokens:
        return JSONResponse(
            status_code=413,
            content={
                "type": "error",
                "error": {
                    "type": "invalid_request_error",
                    "message": "Prompt is too long",
                },
            },
        )

    start_time = time.time()
    username = get_username_from_token(user_key["token"])

    anthropic_headers = {
        "anthropic-beta": raw_request.headers.get("anthropic-beta", ""),
        "anthropic-version": raw_request.headers.get("anthropic-version", ""),
    }

    # Replace model name with the real backend model
    request_dict["model"] = model_config["params"]["model"]

    if stream:
        collected_text = ""

        async def event_generator():
            nonlocal collected_text
            try:
                async for chunk in fetch_anthropic_messages_stream(
                    model_config, request_dict, anthropic_headers
                ):
                    # Collect text for logging
                    if "text_delta" in chunk:
                        try:
                            data_str = chunk.split("data: ", 1)[1]
                            evt = json.loads(data_str)
                            if evt.get("delta", {}).get("type") == "text_delta":
                                collected_text += evt["delta"].get("text", "")
                        except Exception:
                            pass
                    yield chunk
            except HTTPException:
                raise
            except Exception as e:
                error_event = {"type": "error", "error": {"type": "api_error", "message": f"Model is not compatible with the Anthropic Messages API: {e}"}}
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
            finally:
                response_time = time.time() - start_time
                output_tokens_nb = estimate_tokens(collected_text)
                cost_per_output = calculate_token_cost(cost_per_output_token, output_tokens_nb)
                total_tokens = input_tokens_nb + output_tokens_nb
                log_metrics(model_name, username, total_tokens, response_time, cost_per_input, cost_per_output)
                lib.db.create_request(
                    user_name=username,
                    model_name=model_name,
                    prompt=json.dumps(request_dict.get("messages", [])),
                    response=collected_text,
                    co2=0,
                    tokens_used=total_tokens,
                    response_latency=response_time,
                    input_cost=cost_per_input,
                    output_cost=cost_per_output,
                )

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
    else:
        try:
            anthropic_response = await fetch_anthropic_messages(
                model_config, request_dict, anthropic_headers
            )
        except HTTPException as e:
            raise HTTPException(
                status_code=e.status_code,
                detail=f"Model is not compatible with the Anthropic Messages API: {e.detail}",
            )

        response_time = time.time() - start_time

        output_tokens_nb = anthropic_response.get("usage", {}).get("output_tokens", 0)
        if output_tokens_nb == 0:
            content = anthropic_response.get("content", [])
            text = content[0]["text"] if content else ""
            output_tokens_nb = estimate_tokens(text)

        cost_per_output = calculate_token_cost(cost_per_output_token, output_tokens_nb)
        total_tokens = input_tokens_nb + output_tokens_nb

        log_metrics(model_name, username, total_tokens, response_time, cost_per_input, cost_per_output)
        lib.db.create_request(
            user_name=username,
            model_name=model_name,
            prompt=json.dumps(request_dict.get("messages", [])),
            response=json.dumps(anthropic_response),
            co2=0,
            tokens_used=total_tokens,
            response_latency=response_time,
            input_cost=cost_per_input,
            output_cost=cost_per_output,
        )

        return anthropic_response


@app.post("/v1/messages/count_tokens")
async def count_tokens(user_key = Depends(verify_auth)):
    raise HTTPException(status_code=501, detail="Token counting not supported")


# list models endpoint
@app.get("/v1/models")
async def list_models(user_key = Depends(verify_token)):
    models = []
    seen = set()
    for model in CONFIG['model_list']:
        name = model['model_name']
        # A model_name may be declared multiple times for load balancing; expose
        # it to clients only once, otherwise tools like Cline see duplicates.
        if name in seen or name not in user_key['models']:
            continue
        seen.add(name)
        model_info = {
            "id": name,
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

# Load-balancer endpoint health (which replicas are up for each multi-endpoint model)
@app.get("/health/endpoints")
async def endpoints_health():
    return load_balancer.health_snapshot()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
