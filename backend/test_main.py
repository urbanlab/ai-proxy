from fastapi.testclient import TestClient
from main import app
from lib.data_types import ChatCompletionResponse, EmbeddingResponse
import os


token = os.environ["TOKEN"]
print("TOKEN", token )
client = TestClient(app)

b64_image = "data:image/gif;base64,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"
url_image = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
chatRequest = {
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

embeddingRequest = {
  "model": "text-embedding-3-small",
  "input": [
    "Message a embeder"
  ],
  "user": "string"
}


def test_read_main():
    response = client.get("/")
    assert response.status_code == 404
    assert response.json() == {"detail":"Not Found"}
def test_list_models():
    response = client.get("/v1/models", headers={"Authorization": "Bearer " + token})
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data["data"], list)
def test_completion_chat():
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer " + token},
        json=chatRequest 
    )
    data = response.json()
    # check type using pydantic class
    assert ChatCompletionResponse(**data)

def test_embedding_array():
    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer " + token},
        json=embeddingRequest        
    ,)
    data = response.json()
    assert EmbeddingResponse(**data)

def test_embedding_string():
    embeddingRequest["input"] = "Message to embed" 
    response = client.post(
        "/v1/embeddings",
        headers={"Authorization": "Bearer " + token},
        json=embeddingRequest        
    ,)
    data = response.json()
    assert EmbeddingResponse(**data) 

# def test_completion_chat_b64Image():
#     chatRequest["messages"][0]["content"] = [
#         {
#             "type": "text",
#             "text": "What is image ?"
#         },
#         {
#             "type": "image_url",
#             "image_url": {"url": b64_image}
#         }
#     ]
#     print(chatRequest) 
#     response = client.post(
#         "/v1/chat/completions",
#         headers={"Authorization": "Bearer " + token},
#         json=chatRequest 
#     )
#     data = response.json()
#     print(data)
#     # check type using pydantic class
#     assert ChatCompletionResponse(**data) 

# def test_completion_chat_urlImage():
#     chatRequest["messages"][0]["content"] = [
#         {
#             "type": "text",
#             "text": "What is image ?"
#         },
#         {
#             "type": "image_url",
#             "image_url": {"url": url_image}
#         }
#     ]
#     print(chatRequest) 
#     response = client.post(
#         "/v1/chat/completions",
#         headers={"Authorization": "Bearer " + token},
#         json=chatRequest 
#     )
#     data = response.json()
#     print(data)
#     # check type using pydantic class
#     assert ChatCompletionResponse(**data) 
