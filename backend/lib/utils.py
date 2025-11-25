from fastapi import status, HTTPException
import tiktoken 
import aiohttp
import base64
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


async def fetch_image_as_base64(url: str) -> str:
    """Fetch an image from a URL and convert it to base64 data URL"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Referer': 'https://www.google.com/'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                if resp.status != 200:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"Failed to fetch image from URL: HTTP {resp.status}"
                    )
                
                # Get content type
                content_type = resp.headers.get('content-type', 'image/jpeg')
                if not content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail=f"URL does not point to an image (content-type: {content_type})"
                    )
                
                # Read image data
                image_data = await resp.read()
                
                # Validate image data is not empty
                if len(image_data) == 0:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Fetched image data is empty"
                    )
                
                # Convert to base64
                base64_data = base64.b64encode(image_data).decode('utf-8')
                
                # Return as data URL
                return f"data:{content_type};base64,{base64_data}"
    
    except HTTPException:
        raise
    except aiohttp.ClientError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to fetch image from URL: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Error processing image URL: {str(e)}"
        )
