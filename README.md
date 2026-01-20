# AI PROXY

![logo](medias/logo_invert.png)

AI Proxy is an open-source project that provides a simple way to create a proxy server for LLM models.

Many existing solutions are pseudo-open-source with hidden features behind paywalls. AI Proxy aims to be fully open-source and free to use.


## 🍱 Features
- Monitor requests and responses
- api key model permission management
- Partial Support of openai api endpoint


## 📅 Planned Features
- Rate limiting
- CO2 emission tracking (CodeCarbon API)
- Same model load balancing

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
      cost_per_token: 0.20 # 0.20€ for ex
      max_budget: 10 # 10€ for ex 
      budget_duration: 30d # frequency of reset (every x day counting from last update)
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

## 📈 Monitoring
The api exposes prometheus metrics for monitoring.
The prometheus endpoint is available at `http://localhost:8001/metrics`.

exposed metrics:
- request_count
- request_latency
- request_tokens
- response_tokens


## ❤️ Humans.txt
- aangelot
- pipazoul
