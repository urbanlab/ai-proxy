from prometheus_client import Counter, Gauge

request_by_model_count = Counter(
    'llm_requests_total',
    'Total number of requests by model and user',
    ['model']
)
request_by_user_count = Counter(
    'llm_requests_total_user',
    'Total number of requests by user',
    ['user', 'model']
)
token_by_request_count = Counter(
    'llm_tokens_total',
    'Total number of tokens used by model and user',
    ['model']
)
token_by_user_count = Counter(
    'llm_tokens_total_user',
    'Total number of tokens used by user and model',
    ['user', 'model']
)
latency_by_model = Gauge(
    'llm_request_latency_seconds',
    'Request latency in seconds by model',
    ['model']
)
latency_by_user = Gauge(
    'llm_request_latency_seconds_user',
    'Request latency in seconds by user',
    ['user']
)


# prometheus log functions
def log_metrics(model: str, user: str, tokens: int, latency: float):
    request_by_model_count.labels(model=model).inc()
    request_by_user_count.labels(user=user, model=model).inc()
    token_by_request_count.labels(model=model).inc(tokens)
    token_by_user_count.labels(user=user, model=model).inc(tokens)
    latency_by_model.labels(model=model).set(latency)
    latency_by_user.labels(user=user).set(latency)
