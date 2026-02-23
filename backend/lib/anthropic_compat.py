import json
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


def anthropic_to_openai_request(anthropic_req: Dict[str, Any]) -> Dict[str, Any]:
    """Convert an Anthropic Messages API request dict to OpenAI Chat Completions format."""
    messages = []

    # Promote top-level system field to a system message
    system = anthropic_req.get("system")
    if system:
        if isinstance(system, str):
            messages.append({"role": "system", "content": system})
        elif isinstance(system, list):
            system_text = " ".join(
                block.get("text", "")
                for block in system
                if block.get("type") == "text"
            )
            if system_text:
                messages.append({"role": "system", "content": system_text})

    for msg in anthropic_req.get("messages", []):
        role = msg["role"]
        content = msg["content"]

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        # Convert Anthropic content blocks → OpenAI content parts
        openai_parts = []
        for block in content:
            block_type = block.get("type")
            if block_type == "text":
                openai_parts.append({"type": "text", "text": block.get("text", "")})
            elif block_type == "image":
                source = block.get("source", {})
                src_type = source.get("type")
                if src_type == "base64":
                    data_url = f"data:{source['media_type']};base64,{source['data']}"
                    openai_parts.append({"type": "image_url", "image_url": {"url": data_url}})
                elif src_type == "url":
                    openai_parts.append({"type": "image_url", "image_url": {"url": source["url"]}})

        # Simplify to a plain string when there is only a single text part
        if len(openai_parts) == 1 and openai_parts[0]["type"] == "text":
            messages.append({"role": role, "content": openai_parts[0]["text"]})
        elif openai_parts:
            messages.append({"role": role, "content": openai_parts})

    openai_req: Dict[str, Any] = {
        "messages": messages,
        "stream": anthropic_req.get("stream", False),
    }

    if anthropic_req.get("max_tokens") is not None:
        openai_req["max_tokens"] = anthropic_req["max_tokens"]
    if anthropic_req.get("temperature") is not None:
        openai_req["temperature"] = anthropic_req["temperature"]
    if anthropic_req.get("top_p") is not None:
        openai_req["top_p"] = anthropic_req["top_p"]
    if anthropic_req.get("stop_sequences"):
        openai_req["stop"] = anthropic_req["stop_sequences"]

    return openai_req


def openai_to_anthropic_response(
    openai_resp: Dict[str, Any],
    model_name: str,
    message_id: str,
) -> Dict[str, Any]:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format."""
    choices = openai_resp.get("choices", [])
    choice = choices[0] if choices else {}
    message = choice.get("message", {})
    content_text = message.get("content", "") or ""

    finish_reason = choice.get("finish_reason", "stop")
    stop_reason_map = {
        "stop": "end_turn",
        "length": "max_tokens",
        "content_filter": "stop_sequence",
    }
    stop_reason = stop_reason_map.get(finish_reason, "end_turn")

    usage = openai_resp.get("usage", {})

    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": content_text}],
        "model": model_name,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


async def openai_stream_to_anthropic_stream(
    openai_gen: AsyncGenerator[str, None],
    model_name: str,
    message_id: str,
    input_tokens: int = 0,
) -> AsyncGenerator[str, None]:
    """Translate an OpenAI SSE stream into an Anthropic SSE event stream."""

    yield (
        "event: message_start\n"
        f"data: {json.dumps({'type': 'message_start', 'message': {'id': message_id, 'type': 'message', 'role': 'assistant', 'content': [], 'model': model_name, 'stop_reason': None, 'stop_sequence': None, 'usage': {'input_tokens': input_tokens, 'output_tokens': 1}}})}\n\n"
    )

    yield (
        "event: content_block_start\n"
        f"data: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"
    )

    yield f"event: ping\ndata: {json.dumps({'type': 'ping'})}\n\n"

    output_tokens = 0
    stop_reason = "end_turn"

    async for chunk in openai_gen:
        if not chunk.startswith("data: "):
            continue
        raw = chunk[6:].strip()
        if raw == "[DONE]":
            break
        try:
            data = json.loads(raw)
            choices = data.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            finish_reason = choices[0].get("finish_reason")

            if finish_reason:
                fr_map = {
                    "stop": "end_turn",
                    "length": "max_tokens",
                    "content_filter": "stop_sequence",
                }
                stop_reason = fr_map.get(finish_reason, "end_turn")

            content = delta.get("content")
            if content:
                output_tokens += 1
                yield (
                    "event: content_block_delta\n"
                    f"data: {json.dumps({'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': content}})}\n\n"
                )
        except json.JSONDecodeError:
            continue

    yield (
        "event: content_block_stop\n"
        f"data: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"
    )

    yield (
        "event: message_delta\n"
        f"data: {json.dumps({'type': 'message_delta', 'delta': {'stop_reason': stop_reason, 'stop_sequence': None}, 'usage': {'output_tokens': output_tokens}})}\n\n"
    )

    yield f"event: message_stop\ndata: {json.dumps({'type': 'message_stop'})}\n\n"
