import secrets
import base64
import os
import yaml
from typing import Optional
from fastapi import Request, Response, Security, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from lib.rate_limit import rateLimit

# In your config.yaml loading section
with open("/config.yaml", "r") as f:
    CONFIG = yaml.safe_load(f)

# Get metrics auth from config or environment
METRICS_AUTH = CONFIG.get('metrics_auth', {})
METRICS_USERNAME = METRICS_AUTH.get('username', 'admin')
METRICS_PASSWORD = METRICS_AUTH.get('password', 'change-me')

# Security
security = HTTPBearer()



    

# verify user token and model access
def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    for key in CONFIG['keys']:
        if key['token'] == token:
            return key
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing token or insufficient permissions",
        headers={"WWW-Authenticate": "Bearer"},
    )

def get_username_from_token(token: str) -> Optional[str]:
    for key in CONFIG['keys']:
        if key['token'] == token:
            return key['name']
    return None

def get_user_from_token(token: str) -> Optional[str]:
    for key in CONFIG['keys']:
        if key['token'] == token:
            return key
    return None



def verify_metrics_auth(credentials: str) -> bool:
    """Verify HTTP Basic Auth credentials for metrics endpoint"""
    try:
        decoded = base64.b64decode(credentials).decode("utf-8")
        username, password = decoded.split(":", 1)
        # Use secrets.compare_digest to prevent timing attacks
        username_correct = secrets.compare_digest(username, METRICS_USERNAME)
        password_correct = secrets.compare_digest(password, METRICS_PASSWORD)
        return username_correct and password_correct
    except Exception:
        return False
    
async def metrics_auth_middleware(request: Request, call_next):
    """Middleware to protect /metrics endpoint with Basic Auth"""
    if request.url.path.startswith("/metrics"):
        auth_header = request.headers.get("Authorization")
        
        if not auth_header or not auth_header.startswith("Basic "):
            return Response(
                content="Unauthorized",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Metrics"'}
            )
        
        credentials = auth_header[6:]  # Remove "Basic " prefix
        if not verify_metrics_auth(credentials):
            return Response(
                content="Invalid credentials",
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Metrics"'}
            )
    
    response = await call_next(request)
    return response

user_rate_limits = {}


def check_rate_limit(user_key, rpm_limit):
    current_rate_limit = user_rate_limits.get(user_key, None)
    if current_rate_limit:
        print(current_rate_limit.is_allowed())
    
    else:
       rate_limit = rateLimit()
       rate_limit.request_limit  = rpm_limit
       user_rate_limits.update({
           user_key: rate_limit
       })

    if user_rate_limits[user_key].is_allowed() == False:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests please wait before sending a new one",
        )
    



def verify_auth(credentials: HTTPAuthorizationCredentials = Security(security)):
    user_key = verify_token(credentials)
    token = credentials.credentials
    user = get_user_from_token(token)
    print("USER",user)
    check_rate_limit(user["name"],user["rpm_limit"])
    return user_key
