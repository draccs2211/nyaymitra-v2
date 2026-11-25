"""
Shared OpenAI access for all agents.

- Every call has a timeout (LLM_TIMEOUT_SECONDS) and SDK retries
  (LLM_MAX_RETRIES) — the SDK default timeout is 600 s, which would pin a
  worker thread for 10 minutes on a hung request.
- One client instance per client class is cached, so production shares a
  single pooled OpenAI client. Agents pass their module-level `OpenAI` class,
  which keeps the per-module test mocks working.
- Returns token usage so the pipeline can account cost per query.
"""
import json
import threading
from dataclasses import dataclass
from typing import Any, Dict, Tuple

from backend.config import LLM_MAX_RETRIES, LLM_MODEL, LLM_TIMEOUT_SECONDS, OPENAI_API_KEY

_clients: Dict[Any, Any] = {}
_lock = threading.Lock()


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.prompt_tokens + other.prompt_tokens,
                     self.completion_tokens + other.completion_tokens)


def get_client(client_cls):
    """Cached client per class (thread-safe). Construction errors are not cached."""
    with _lock:
        client = _clients.get(client_cls)
        if client is None:
            client = client_cls(api_key=OPENAI_API_KEY, timeout=LLM_TIMEOUT_SECONDS,
                                max_retries=LLM_MAX_RETRIES)
            _clients[client_cls] = client
        return client


def reset_clients() -> None:
    with _lock:
        _clients.clear()


def _usage_of(response) -> Usage:
    u = getattr(response, "usage", None)
    if u is None:
        return Usage()
    return Usage(int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0))


def chat_json(client_cls, system: str, user: str, *, max_tokens: int,
              temperature: float = 0.2) -> Tuple[dict, Usage]:
    """JSON-mode chat completion. Raises on API errors or invalid JSON (callers fall back)."""
    response = get_client(client_cls).chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    usage = _usage_of(response)
    return json.loads(response.choices[0].message.content), usage
