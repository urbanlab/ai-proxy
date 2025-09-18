# AI PROXY

AI Proxy is an open-source project that provides a simple way to create a proxy server for LLM models.

Many existing solutions are pseudo-open-source with hidden features behind paywalls. AI Proxy aims to be fully open-source and free to use.


## Features
- CO2 emission tracking (CodeCarbon API)
- Monitor requests and responses
- api key model permission management
- Rate limiting
- Support for openai api endpoint


## 🚀 Quickstart

**Requirements:**
- Docker

**Copy the example configuration file and edit it to your needs:**

```bash
cp config.example.yaml config.yaml
```
Edit `config.yaml` to set your OpenAI API key and other configurations.
```
global:
model_list:
  - model_name: devstral
    params:
      model: devstral:latest
      api_base: http://ollama-service.ollama.svc.cluster.local:11434/v1
      drop_params: true
      api_key: "no_token"
      max_input_tokens: 25000

keys:
  - name: "user"
    token: "token"
    models:
      - "devstral"
```

**Run the server:**

```bash
docker-compose up -d
```

The server will be available at `http://localhost:8000`.
And the docs at `http://localhost:8000/docs`.

## Monitoring
An prometheus endpoint is available at `http://localhost:8001/metrics`.

exposed metrics:
- request_count
- request_latency
- request_tokens
- response_tokens



## TODO
- [x] embedding routes
- [x] transcript routes
- [x] tts routes
- [ ] streaming tts route
- [ ] streaming transcript route
- basic stats (hours/days/weeks/months)
    - [x] models
        - [x] requests per model (line chart)
        - [x] tokens per model (line chart)
        - [x] model repartition (pie chart)
        - [x] average response latency per model (line chart)
        - [ ] nb or requests per min per model (bar chart)
    - [ ] user
        - [x] requests per user (bar chart)
        - [x] tokens per user (line chart)
        - [x] user repartition per token (pie chart)
        - user repartition per request (pie chart)
        - [x] average response latency per user (line chart)
        - nb or requets per min per user (bar chart)
        - max token per request (line chart)
        - min token per request (line chart)
    - [x] totals
        - [x] requests (bar chart)
        - [x] tokens (line chart)
        - [x] user repartition (pie chart)

- FIX:
    - [x] support image upload
    - [ ] Add cline support
- [ ] gpu api agent

