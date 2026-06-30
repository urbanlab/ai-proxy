import aiohttp
import asyncio
import logging
from fastapi import HTTPException
from typing import Dict, Any, AsyncGenerator, Callable, List, Optional, Tuple
import os

logger = logging.getLogger("openai_proxy")

# Errors that mean "this endpoint is unreachable" — safe to fail over to the next
# replica.
_CONNECTION_ERRORS = (
    aiohttp.ClientConnectorError,
    aiohttp.ServerDisconnectedError,
    aiohttp.ClientOSError,
    asyncio.TimeoutError,
)

# HTTP statuses that mean "this endpoint isn't serving the request" rather than
# "the request itself is bad". A downed model server fronted by a gateway often
# answers with 405/502/503/504 instead of dropping the connection, so these must
# trigger failover too. Genuine request errors (400/401/403/404/422...) are NOT
# in this set: they would be identical on every interchangeable replica, so we
# return them to the client unchanged instead of masking them.
_FAILOVER_STATUS_CODES = frozenset({405, 408, 429})


def _should_failover(status: int) -> bool:
    return status >= 500 or status in _FAILOVER_STATUS_CODES


def _endpoint_request(ep: Dict[str, Any], request_data: Dict[str, Any]) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """Build (url, headers, body) for a chat completion against one endpoint."""
    url = f"{ep['params']['api_base']}/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = ep['params'].get('api_key')
    if api_key and api_key != "no_token":
        headers["Authorization"] = f"Bearer {api_key}"
    body = dict(request_data)
    body["model"] = ep['params']['model']  # use this endpoint's upstream model name
    return url, headers, body


async def fetch_chat_completion_failover(
    candidates: List[Dict[str, Any]],
    request_data: Dict[str, Any],
    on_dead: Callable[[Dict[str, Any]], None],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Non-streaming request with failover. Tries each candidate in order; on a
    connection failure or an endpoint-unavailable HTTP status (see
    _should_failover) marks it dead via on_dead(ep) and moves on. Returns
    (response_json, endpoint_that_served). Genuine request errors are raised
    immediately; if every endpoint fails the last error is re-raised."""
    last_exc: Optional[HTTPException] = None
    for ep in candidates:
        url, headers, body = _endpoint_request(ep, request_data)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, headers=headers, json=body) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        exc = HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
                        if _should_failover(resp.status):
                            logger.warning("Endpoint %s returned %s; failing over to next endpoint", url, resp.status)
                            on_dead(ep)
                            last_exc = exc
                            continue
                        raise exc
                    return await resp.json(), ep
        except _CONNECTION_ERRORS as e:
            logger.warning("Connect failed for %s: %s; failing over to next endpoint", url, e)
            on_dead(ep)
            last_exc = HTTPException(status_code=502, detail=f"Upstream connection failed: {e}")
            continue
    if last_exc is not None:
        raise last_exc
    raise HTTPException(status_code=502, detail="No upstream endpoints available")


async def fetch_chat_completion_stream_failover(
    candidates: List[Dict[str, Any]],
    request_data: Dict[str, Any],
    on_dead: Callable[[Dict[str, Any]], None],
) -> AsyncGenerator[str, None]:
    """Streaming request with failover. The connection to each candidate is
    established before any bytes are yielded, so failing over to a live replica
    is fully transparent to the client. Only if every endpoint is unreachable
    does this raise (surfaced to the client as a stream error)."""
    last_exc: Optional[HTTPException] = None
    for ep in candidates:
        url, headers, body = _endpoint_request(ep, request_data)
        session = aiohttp.ClientSession()
        try:
            resp = await session.post(url, headers=headers, json=body)
        except _CONNECTION_ERRORS as e:
            await session.close()
            logger.warning("Stream connect failed for %s: %s; failing over to next endpoint", url, e)
            on_dead(ep)
            last_exc = HTTPException(status_code=502, detail=f"Upstream connection failed: {e}")
            continue

        if resp.status != 200:
            text = await resp.text()
            await session.close()
            exc = HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
            if _should_failover(resp.status):
                logger.warning("Stream endpoint %s returned %s; failing over to next endpoint", url, resp.status)
                on_dead(ep)
                last_exc = exc
                continue
            raise exc

        # Connected — stream this endpoint to completion, then we're done.
        try:
            async for line in resp.content:
                line_str = line.decode('utf-8').strip()
                if line_str:
                    if line_str.startswith('{'):
                        yield f"data: {line_str}\n\n"
                    elif line_str == "[DONE]":
                        yield f"data: [DONE]\n\n"
                    else:
                        yield f"{line_str}\n\n"
            return
        finally:
            await session.close()

    if last_exc is not None:
        raise last_exc
    raise HTTPException(status_code=502, detail="No upstream endpoints available")

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
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")

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
