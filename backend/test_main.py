from fastapi.testclient import TestClient
from main import app
from lib.types import ChatCompletionRequest, EmbeddingInput, SpeechRequest, Message, MessageContent
import os


token = os.environ["TOKEN"]
print("TOKEN", token )
client = TestClient(app)


data = {
    "model": "gemma3",  # Replace with actual model name (e.g., "gpt-4")
    "messages": [
        {
            "role": "user",  # or "system", "assistant"
            "content": "Hello, how are you?",  # Replace with your actual message
            # "name": "optional_name"  # Only needed for named users in some APIs
        }
    ],
    "max_tokens": 100,  # Replace with your desired max tokens
    "temperature": 0.7,  # Replace with your desired temperature (0-2)
    "top_p": 1,
    "n": 1,
    "stream": False,
    "stop": None,  # or ["string"] if you want to stop at certain phrases
    "presence_penalty": 0,
    "frequency_penalty": 0
    # Note: 'usage' is typically part of the response, not the request
}

def test_read_main():
    response = client.get("/")
    assert response.status_code == 404
    assert response.json() == {"detail":"Not Found"}
def test_list_models():
    print("HEYHO", token)
    response = client.get("/v1/models", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["data"], list)
def test_completion_chat():
    response = client.get("/v1/models", headers={"Authorization": "Bearer " + token})
    print(response.json())
    data = response.json()
    # Check response using types and pydantic ?
    assert response.json
