"""
Agent output schema tests — no paid APIs (OpenAI / Cohere are mocked).
Run: pytest tests/test_agent_outputs.py -v
"""
import json
from types import SimpleNamespace

import pytest

from backend.agents import schemas
from tests.test_pipeline import make_state


def fake_openai(content: str):
    """Return an OpenAI-like class whose chat completion yields `content`."""
    class _Client:
        def __init__(self, *a, **kw):
            msg = SimpleNamespace(content=content)
            resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kw: resp)
            )
    return _Client


# ─── Guardian ─────────────────────────────────────────────────────────────────

class TestGuardianOutput:
    def test_normal_validates(self):
        from backend.agents.guardian import guardian_agent
        out = guardian_agent(make_state(user_query="landlord not returning deposit"))
        assert out == {"is_emergency": False, "emergency_type": ""}
        schemas.GuardianOutput(**out)

    def test_emergency_validates(self):
        from backend.agents.guardian import guardian_agent
        out = guardian_agent(make_state(user_query="client facing domestic violence"))
        m = schemas.GuardianOutput(**out)
        assert m.is_emergency and m.case_strength == "urgent"


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class TestOrchestratorOutput:
    def _run(self, monkeypatch, payload):
        from backend.agents import orchestrator
        content = payload if isinstance(payload, str) else json.dumps(payload)
        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(content))
        return orchestrator.orchestrator_agent(make_state(user_query="q"))

    def test_valid_llm_json(self, monkeypatch):
        out = self._run(monkeypatch, {
            "legal_domain": "property", "needs_document": True,
            "needs_referral": False, "language": "hinglish", "urgency": "medium",
            "query_type": "drafting", "key_issues": ["deposit"],
            "suggested_acts": ["UP Rent Act"], "reasoning": "tenant issue",
        })
        assert out["legal_domain"] == "property"
        assert out["needs_document"] is True
        assert out["language"] == "hinglish"
        assert out["case_analysis"]["key_issues"] == ["deposit"]
        schemas.OrchestratorOutput(**out)

    def test_invalid_enum_values_coerced(self, monkeypatch):
        out = self._run(monkeypatch, {
            "legal_domain": "Tax Law", "language": "tamil", "urgency": "ASAP",
            "needs_document": "yes", "key_issues": "single issue",
        })
        assert out["legal_domain"] == "general"
        assert out["language"] == "en"
        assert out["case_analysis"]["urgency"] == "low"
        assert out["needs_document"] is True
        assert out["case_analysis"]["key_issues"] == ["single issue"]

    def test_case_normalized(self, monkeypatch):
        out = self._run(monkeypatch, {"legal_domain": " Criminal ", "language": "HI"})
        assert out["legal_domain"] == "criminal"
        assert out["language"] == "hi"

    def test_non_json_falls_back(self, monkeypatch):
        out = self._run(monkeypatch, "not json at all")
        assert out["case_analysis"]["reasoning"] == "Fallback classification used"
        schemas.OrchestratorOutput(**out)
