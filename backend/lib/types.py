from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Union

# Add these models after your existing Pydantic models
class TranscriptionRequest(BaseModel):
    model: str
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"  # json, text, srt, verbose_json, vtt
    temperature: Optional[float] = 0

class TranscriptionResponse(BaseModel):
    text: str

# Vision support models
class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"  # auto, low, high

class MessageContent(BaseModel):
    type: str  # "text" or "image_url"
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None

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
    content: Union[str, List[MessageContent]]  # Can be string or list for vision
    name: Optional[str] = None

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

# embeddings request and response models
class EmbeddingInput(BaseModel):
    model: str
    input: List[str]
    user: Optional[str] = None

class EmbeddingData(BaseModel):
    object: str
    embedding: List[float]
    index: int

class EmbeddingResponse(BaseModel):
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Optional[Usage] = None

class SpeechRequest(BaseModel):
    model: str
    input: str
    voice: str = "alloy"  # alloy, echo, fable, onyx, nova, shimmer
    response_format: Optional[str] = "mp3"  # mp3, opus, aac, flac, wav, pcm
    speed: Optional[float] = Field(1.0, ge=0.25, le=4.0)  # Speed between 0.25 and 4.0