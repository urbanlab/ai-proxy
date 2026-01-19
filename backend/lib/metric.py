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
    ['user','model']
)

cost_per_user_input = Counter(
    "llm_request_input_cost_user",
    "Cost per user input tokens",
    ['user', 'model']
)

cost_per_user_output = Counter(
    "llm_request_output_cost_user",
    "Cost per user output tokens",
    ['user', 'model']
)


total_cost_per_user = Counter(
    "llm_request_total_cost_user",
    "Total Cost per user tokens",
    ['user', 'model']
)


cost_per_model_input = Counter(
    "llm_request_input_cost",
    "Cost per model input tokens",
    ['model']
)

cost_per_model_output = Counter(
    "llm_request_output_cost",
    "Cost per model output tokens",
    ['model']
)

total_cost_per_model = Counter(
    "llm_request_total_cost",
    "Total token cost per model",
    ['model']
)
# prometheus log functions
def log_metrics(
        model: str,
        user: str,
        tokens: int,
        latency: float,
        input_cost: float = 0,
        output_cost: float =  0
):
    request_by_model_count.labels(model=model).inc()
    request_by_user_count.labels(user=user, model=model).inc()
    token_by_request_count.labels(model=model).inc(tokens)
    token_by_user_count.labels(user=user, model=model).inc(tokens)
    latency_by_model.labels(model=model).set(latency)
    latency_by_user.labels(user=user, model=model).set(latency)
    cost_per_model_input.labels(model=model).inc(input_cost)
    cost_per_model_output.labels(model=model).inc(output_cost)
    total_cost_per_model.labels(model=model).inc(output_cost+input_cost)
    cost_per_user_input.labels(user=user, model=model).inc(input_cost)
    cost_per_user_output.labels(user=user, model=model).inc(output_cost)
    total_cost_per_user.labels(user=user, model=model).inc(output_cost+input_cost)
