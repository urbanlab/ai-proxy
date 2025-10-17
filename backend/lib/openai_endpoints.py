import aiohttp
import base64
import os
from fastapi import HTTPException, status
from typing import Any, AsyncGenerator, Dict, List, Union
from lib.types import Message, MessageContent
from lib.utils import estimate_tokens

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
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"Error response: {text}")  # Debug log
                raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
            
            print(f"Response status: {resp.status}")  # Debug log
            print(f"Response headers: {dict(resp.headers)}")  # Debug log
            
            # Process line by line, not chunk by chunk
            async for line in resp.content:
                line_str = line.decode('utf-8').strip()
                if line_str:
                    # If the line doesn't start with "data:", add it
                    if line_str.startswith('{'):
                        yield f"data: {line_str}\n\n"
                    elif line_str == "[DONE]":
                        yield f"data: [DONE]\n\n"
                    else:
                        # Line already properly formatted
                        yield f"{line_str}\n\n"
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