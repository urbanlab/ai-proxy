import requests

def completion(model: str, prompt: str, token: str = None, url: str = "http://localhost:8000/v1"):

    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}"
    }

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 1,
        "n": 1,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0
    }

    url = f"{url}/chat/completions"

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors
        return response.json()
    except requests.exceptions.RequestException as e:
        print("Request failed:", e)
        return None