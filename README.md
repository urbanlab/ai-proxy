# AI PROXY

AI Proxy is an open-source project that provides a simple way to create a proxy server for LLM models.

Many existing solutions are pseudo-open-source with hidden features behind paywalls. AI Proxy aims to be fully open-source and free to use.


## Features
- CO2 emission tracking (CodeCarbon API)
- Monitor requests and responses
- api key model permission management
- Rate limiting
- Support for openai api endpoint


## TODO
- [x] embedding routes
- [x] transcript routes
- [x] tts routes
- [ ] streaming tts route
- [ ] streaming transcript route
- basic stats (hours/days/weeks/months)
    - [x] models
        OK - requests per model (line chart)
        OK - tokens per model (line chart)
        OK - model repartition (pie chart)
        OK - average response latency per model (line chart)
        - nb or requets per min per model (bar chart)
    - [ ] user
       OK - requests per user (bar chart)
       OK - tokens per user (line chart)
       OK - user repartition per token (pie chart)
        - user repartition per request (pie chart)
       OK - average response latency per user (line chart)
        - nb or requets per min per user (bar chart)
        OK - max token per request (line chart)
        OK - min token per request (line chart)
    - [ ] totals
        OK - requests (bar chart)
        - tokens (line chart)
        - user repartition (pie chart)

- FIX:
    - [ ] support image upload
    - [ ] Add cline support
- gpu api agent


## Quickstart

**Copy the example configuration file and edit it to your needs:**

```bash
cp config.example.yaml config.yaml
```

**Run the server:**

```bash
docker-compose up -d
```

The server will be available at `http://localhost:8000`.
