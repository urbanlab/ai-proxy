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


# Request timeouts (seconds). Overridable from config at startup via
# configure_timeouts(). We deliberately do NOT bound the total time of a request
# by the model's generation time — a slow-but-alive model is fine. Instead:
#   * connect: cap time to establish the TCP/TLS connection.
#   * request (non-streaming): total wall-clock cap; a non-streaming upstream is
#     silent until the whole answer is ready, so a per-read idle timeout can't be
#     used there.
#   * stream_idle (streaming): max gap between streamed chunks. Catches a GPU that
#     hangs mid-generation (which otherwise surfaces to the client as an empty
#     message) without killing a long but actively-streaming response.
_CONNECT_TIMEOUT = 10
_REQUEST_TIMEOUT = 600
_STREAM_IDLE_TIMEOUT = 120


def configure_timeouts(connect=None, request=None, stream_idle=None) -> None:
    """Override upstream request timeouts (called once at startup from config)."""
    global _CONNECT_TIMEOUT, _REQUEST_TIMEOUT, _STREAM_IDLE_TIMEOUT
    if connect is not None:
        _CONNECT_TIMEOUT = connect
    if request is not None:
        _REQUEST_TIMEOUT = request
    if stream_idle is not None:
        _STREAM_IDLE_TIMEOUT = stream_idle


def _request_client_timeout() -> aiohttp.ClientTimeout:
    return aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT, sock_connect=_CONNECT_TIMEOUT)


def _stream_client_timeout() -> aiohttp.ClientTimeout:
    # total=None: don't cap a long streaming generation; sock_read bounds the idle
    # gap between chunks so a stalled upstream errors out instead of hanging.
    return aiohttp.ClientTimeout(
        total=None, sock_connect=_CONNECT_TIMEOUT, sock_read=_STREAM_IDLE_TIMEOUT
    )


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
    load_balancer: Any,
    model_name: str,
    request_data: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Non-streaming request with least-connections dispatch and failover.

    Repeatedly asks the load balancer for the freest untried replica
    (`acquire`), holding that endpoint's concurrency slot for the whole request
    and releasing it when done. If a replica is unreachable or returns an
    endpoint-unavailable status (see _should_failover) it is marked dead and the
    next replica is tried. Returns (response_json, endpoint_that_served). Genuine
    request errors are raised immediately; if every endpoint fails the last error
    is re-raised."""
    last_exc: Optional[HTTPException] = None
    tried: set = set()
    while True:
        ep = await load_balancer.acquire(model_name, tried)
        if ep is None:
            break
        url, headers, body = _endpoint_request(ep, request_data)
        try:
            async with aiohttp.ClientSession(timeout=_request_client_timeout()) as session:
                async with session.post(url, headers=headers, json=body) as resp:
                    if resp.status != 200:
                        text = await resp.text()
                        exc = HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
                        if _should_failover(resp.status):
                            logger.warning("Endpoint %s returned %s; failing over to next endpoint", url, resp.status)
                            load_balancer.mark_unhealthy(ep)
                            last_exc = exc
                            continue
                        raise exc
                    return await resp.json(), ep
        except _CONNECTION_ERRORS as e:
            logger.warning("Connect failed for %s: %s; failing over to next endpoint", url, e)
            load_balancer.mark_unhealthy(ep)
            last_exc = HTTPException(status_code=502, detail=f"Upstream connection failed: {e}")
            continue
        finally:
            load_balancer.release(ep)   # sync, cancellation-proof
    if last_exc is not None:
        raise last_exc
    raise HTTPException(status_code=502, detail="No upstream endpoints available")


async def fetch_chat_completion_stream_failover(
    load_balancer: Any,
    model_name: str,
    request_data: Dict[str, Any],
) -> AsyncGenerator[str, None]:
    """Streaming request with least-connections dispatch and failover. The
    endpoint's concurrency slot is held for the whole stream and released when it
    finishes or errors. The connection to each candidate is established before any
    bytes are yielded, so failing over to a live replica is fully transparent to
    the client. Only if every endpoint is unreachable does this raise (surfaced to
    the client as a stream error)."""
    last_exc: Optional[HTTPException] = None
    tried: set = set()
    while True:
        ep = await load_balancer.acquire(model_name, tried)
        if ep is None:
            break
        url, headers, body = _endpoint_request(ep, request_data)
        session = aiohttp.ClientSession(timeout=_stream_client_timeout())
        # Everything from here is wrapped so the slot is ALWAYS released for this
        # `ep`, even if the client disconnects during connect (CancelledError) —
        # otherwise the in-flight count leaks and the endpoint looks permanently
        # full. release() is synchronous, so it runs first and can't be cancelled.
        try:
            try:
                resp = await session.post(url, headers=headers, json=body)
            except _CONNECTION_ERRORS as e:
                logger.warning("Stream connect failed for %s: %s; failing over to next endpoint", url, e)
                load_balancer.mark_unhealthy(ep)
                last_exc = HTTPException(status_code=502, detail=f"Upstream connection failed: {e}")
                continue

            if resp.status != 200:
                text = await resp.text()
                exc = HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
                if _should_failover(resp.status):
                    logger.warning("Stream endpoint %s returned %s; failing over to next endpoint", url, resp.status)
                    load_balancer.mark_unhealthy(ep)
                    last_exc = exc
                    continue
                raise exc

            # Connected — stream this endpoint to completion, then we're done.
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
            load_balancer.release(ep)   # sync, cancellation-proof — must run first
            try:
                await session.close()
            except Exception:
                pass

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
