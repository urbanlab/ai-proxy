"""Per-provider OpenAI-compatible parameter mapping.

Different backends honor the same OpenAI request params through different native
mechanisms. Rather than one inline if/else gated on a `drop_params` boolean, this
is a small registry of per-provider transform functions — a scaled-down version
of LiteLLM's per-provider Config pattern.

Each function receives the outgoing request dict (already `model_dump`ed and
model-name-rewritten) and returns it with OpenAI-standard params translated into
what that backend actually enforces. A model opts in via `provider:` in its
config.yaml entry; models without a known `provider` pass through untouched
(correct for OpenAI / Anthropic, which handle the OpenAI params natively).

The transforms run BEFORE the `drop_params` whitelist in main.py, so the native
output keys they emit (`grammar`, `chat_template_kwargs`, `think`) must also be
present in that whitelist to survive for `drop_params: true` backends. Each of
those keys is only ever populated by the provider that supports it, so allowing
them globally is harmless for the others.
"""

# Generic GBNF grammar constraining output to any valid JSON value. Makes
# `response_format: {"type": "json_object"}` actually enforced on llama.cpp,
# which otherwise treats it as a hint and can still emit ```json fences.
JSON_OBJECT_GRAMMAR = r"""
root   ::= object
value  ::= object | array | string | number | ("true" | "false" | "null") ws
object ::=
  "{" ws (
            string ":" ws value
    ("," ws string ":" ws value)*
  )? "}" ws
array  ::=
  "[" ws (
            value
    ("," ws value)*
  )? "]" ws
string ::=
  "\"" (
    [^"\\\x7F\x00-\x1F] |
    "\\" (["\\bfnrt/] | "u" [0-9a-fA-F]{4})
  )* "\"" ws
number ::= ("-"? ([0-9] | [1-9] [0-9]{0,15})) ("." [0-9]+)? ([eE] [-+]? [0-9]{1,4})? ws
ws ::= | " " | "\n" [ \t]{0,20}
"""

# Effort tiers that mean "don't think". Everything else enables thinking.
_THINKING_OFF = ("none", "minimal")


def _thinking_enabled(effort):
    return str(effort).lower() not in _THINKING_OFF


def _map_llamacpp(params):
    """llama.cpp: json_object needs a grammar to be enforced; reasoning_effort is
    ignored — the working lever is the chat template's enable_thinking kwarg."""
    rf = params.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_object" and "grammar" not in params:
        params["grammar"] = JSON_OBJECT_GRAMMAR

    effort = params.pop("reasoning_effort", None)
    if effort is not None:
        kwargs = params.get("chat_template_kwargs")
        if not isinstance(kwargs, dict):
            kwargs = {}
        kwargs.setdefault("enable_thinking", _thinking_enabled(effort))
        params["chat_template_kwargs"] = kwargs
    return params


def _map_ollama(params):
    """ollama: response_format json_object/json_schema is enforced natively; the
    thinking lever is the native boolean `think` field."""
    effort = params.pop("reasoning_effort", None)
    if effort is not None:
        params.setdefault("think", _thinking_enabled(effort))
    return params


def _map_vllm(params):
    """vLLM: response_format (json_object + json_schema) and reasoning_effort are
    native, BUT its reasoning_effort enum only accepts low/medium/high, so map the
    off tiers to chat_template_kwargs.enable_thinking=false to avoid a 400."""
    effort = params.get("reasoning_effort")
    if effort is not None and str(effort).lower() in _THINKING_OFF:
        params.pop("reasoning_effort")
        kwargs = params.get("chat_template_kwargs")
        if not isinstance(kwargs, dict):
            kwargs = {}
        kwargs.setdefault("enable_thinking", False)
        params["chat_template_kwargs"] = kwargs
    return params


# provider string (from config.yaml `params.provider`) -> transform function.
_PROVIDERS = {
    "llamacpp": _map_llamacpp,
    "ollama": _map_ollama,
    "vllm": _map_vllm,
}


def apply_provider_mapping(request_data, model_config):
    """Apply the transform for this model's configured provider, if any.

    Unknown or missing `provider` -> passthrough (OpenAI / Anthropic handle the
    OpenAI-standard params natively).
    """
    provider = model_config.get("params", {}).get("provider")
    fn = _PROVIDERS.get(provider)
    return fn(request_data) if fn else request_data
