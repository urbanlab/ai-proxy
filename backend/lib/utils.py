import tiktoken 


# Helper function to estimate tokens (add this after your other helper functions)
def estimate_tokens(text: str) -> int:
    """Estimate token count for text"""
    try:
        # Using tiktoken for OpenAI-compatible token counting
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except:
        # Fallback: rough estimation (1 token ≈ 0.75 words)
        return int(len(text.split()) * 1.33)

def extract_tokens_from_response(response_data: dict) -> int:
    """Extract token count from API response"""
    if isinstance(response_data, dict) and 'usage' in response_data:
        return response_data['usage'].get('total_tokens', 0)
    return 0