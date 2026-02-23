import aiohttp
from fastapi import HTTPException
from typing import Any, AsyncGenerator, Dict, List, Optional, Union


def extract_text_from_anthropic_messages(
    messages: List[Dict[str, Any]],
    system: Optional[Union[str, List[Dict[str, Any]]]] = None,
) -> str:
    """Extract plain text from Anthropic-format messages for token estimation."""
    parts = []

    if isinstance(system, str):
        parts.append(system)
    elif isinstance(system, list):
        for block in system:
            if block.get("type") == "text":
                parts.append(block.get("text", ""))

    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))

    return " ".join(parts)


async def fetch_anthropic_messages(
    model_config: Dict[str, Any],
    request_data: Dict[str, Any],
    anthropic_headers: Dict[str, str],
) -> Dict[str, Any]:
    """Proxy a non-streaming request to a native Anthropic Messages API endpoint."""
    api_base = model_config["params"]["api_base"]
    # Strip trailing /v1 if present so we can append /v1/messages
    url = api_base.rstrip("/").removesuffix("/v1") + "/v1/messages"

    headers: Dict[str, str] = {"Content-Type": "application/json"}

    api_key = model_config["params"].get("api_key")
    if api_key and api_key != "no_token":
        headers["x-api-key"] = api_key

    # Forward Anthropic-specific headers from the client
    for h in ("anthropic-beta", "anthropic-version"):
        if anthropic_headers.get(h):
            headers[h] = anthropic_headers[h]

    # Ensure a default anthropic-version if the client didn't send one
    if "anthropic-version" not in headers:
        headers["anthropic-version"] = "2023-06-01"

    print(f"[anthropic-native] POST {url}")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"[anthropic-native] Error {resp.status}: {text}")
                raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")
            return await resp.json()


async def fetch_anthropic_messages_stream(
    model_config: Dict[str, Any],
    request_data: Dict[str, Any],
    anthropic_headers: Dict[str, str],
) -> AsyncGenerator[str, None]:
    """Proxy a streaming request to a native Anthropic Messages API endpoint.

    Yields raw SSE chunks exactly as received from the backend.
    """
    api_base = model_config["params"]["api_base"]
    url = api_base.rstrip("/").removesuffix("/v1") + "/v1/messages"

    headers: Dict[str, str] = {"Content-Type": "application/json"}

    api_key = model_config["params"].get("api_key")
    if api_key and api_key != "no_token":
        headers["x-api-key"] = api_key

    for h in ("anthropic-beta", "anthropic-version"):
        if anthropic_headers.get(h):
            headers[h] = anthropic_headers[h]

    if "anthropic-version" not in headers:
        headers["anthropic-version"] = "2023-06-01"

    print(f"[anthropic-native-stream] POST {url}")

    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=request_data) as resp:
            if resp.status != 200:
                text = await resp.text()
                print(f"[anthropic-native-stream] Error {resp.status}: {text}")
                raise HTTPException(status_code=resp.status, detail=f"Model API error: {text}")

            async for line in resp.content:
                line_str = line.decode("utf-8")
                if line_str.strip():
                    yield line_str
