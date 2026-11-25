"""
Test-wide setup. Runs before any backend import.

- Disable LangSmith tracing: tests must not upload (mocked) runs to LangSmith,
  and the uploads stall on network timeouts. load_dotenv() won't override these.
- Block paid APIs in tests not marked `paid`: any OpenAI client a test didn't
  mock explicitly raises instead of spending money; Cohere rerank is disabled.
"""
import importlib
import os
import sys

import pytest

os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
# Test-only JWT key (forced, so tests never sign with the real one from .env)
os.environ["SECRET_KEY"] = "test-only-secret-key-" + "x" * 40
os.environ["APP_ENV"] = "test"

_LLM_MODULES = (
    "backend.agents.guardian",
    "backend.agents.orchestrator",
    "backend.agents.analysis",
    "backend.agents.drafting",
)


class _BlockedOpenAI:
    def __init__(self, *args, **kwargs):
        raise RuntimeError("Unmocked OpenAI call in a non-paid test — mock it or mark the test @pytest.mark.paid")


@pytest.fixture(autouse=True)
def _reset_rate_limiter():
    from backend.utils.rate_limit import limiter
    limiter.reset()
    yield
    limiter.reset()


@pytest.fixture(autouse=True)
def _block_paid_apis(request, monkeypatch):
    if request.node.get_closest_marker("paid"):
        return
    for name in _LLM_MODULES:
        monkeypatch.setattr(importlib.import_module(name), "OpenAI", _BlockedOpenAI)
    rag = sys.modules.get("backend.rag.hybrid_search")   # only if already loaded (slow import)
    if rag is not None:
        monkeypatch.setattr(rag.hybrid_rag, "cohere_client", None)
    # Indian Kanoon is a paid API: behave as "not configured" unless a test opts in
    monkeypatch.setattr(
        importlib.import_module("backend.mcp_servers.indiankanoon_mcp"), "INDIAN_KANOON_API_KEY", ""
    )
