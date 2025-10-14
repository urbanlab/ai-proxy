import requests
import os
import time
import base64
from io import BytesIO

class TestAPI:
    api_url = os.getenv("API_URL", "http://localhost:8000")
    api_token = os.getenv("API_TOKEN", "")
    headers = {"Authorization": f"Bearer {api_token}"} if api_token else {}
    system_prompt = "You are a helpful assistant."
    user_message = "Hello, how can you assist me today?"
    model = "gemma3"
    image_url = "https://erasme.org/IMG/png/dataviz1.png"

    def health_check(self):
        try:
            response = requests.get(f"{self.api_url}/docs")
            return response.status_code == 200
        except requests.ConnectionError:
            return False
    
    def test_chat_completion(self):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.user_message}
            ]
        }
        response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload, headers=self.headers)
        data = response.json()
        assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"

        assert "choices" in data, "Response JSON does not contain 'choices'"
        assert len(data["choices"]) > 0, "No choices returned in response"
        print("Chat completion test passed.")

    def test_image_upload_b64(self):
    # download image and convert to base64
    image_data = requests.get(self.image_url).content
    image_b64 = base64.b64encode(image_data).decode('utf-8')
    
    payload = {
        "model": self.model,
        "messages": [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user", 
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the image."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ]
    }
    
    response = requests.post(f"{self.api_url}/v1/chat/completions", json=payload, headers=self.headers)
    data = response.json()
    
    assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
    assert "choices" in data, "Response JSON does not contain 'choices'"
    assert len(data["choices"]) > 0, "No choices returned in response"
    print("Image upload (base64) test passed. with response:", data["choices"][0]["message"]["content"])
if __name__ == "__main__":
    api_tester = TestAPI()
    while api_tester.health_check() is False:
        print("Waiting for the API to be ready...")
        time.sleep(1)
    print("API is ready")
    api_tester.test_chat_completion()
    api_tester.test_image_upload_b64()