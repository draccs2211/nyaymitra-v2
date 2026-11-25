"""
Jurisdiction-aware retrieval tests (Phase 2): state detection, explicit
override, filtering, labelling of state-specific sources. No paid APIs.
Run: pytest tests/test_jurisdiction.py -v
"""
import json

import pytest

from backend.rag.hybrid_search import filter_by_jurisdiction, hybrid_rag
from tests.test_agent_outputs import fake_openai
from tests.test_pipeline import make_state

DOCS = [
    {"id": "c", "metadata": {"jurisdiction": "IN"}},
    {"id": "up", "metadata": {"jurisdiction": "IN-UP"}},
    {"id": "dl", "metadata": {"jurisdiction": "IN-DL"}},
    {"id": "nometa"},
]


class TestFilter:
    def test_known_state_keeps_central_and_own(self):
        assert [d["id"] for d in filter_by_jurisdiction(DOCS, "IN-UP")] == ["c", "up", "nometa"]

    def test_unknown_state_keeps_all(self):
        assert filter_by_jurisdiction(DOCS, "IN") == DOCS
        assert filter_by_jurisdiction(DOCS, "") == DOCS


class TestDetection:
    def _orch(self, monkeypatch, detected, request_jur="IN"):
        from backend.agents import orchestrator
        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(json.dumps(
            {"legal_domain": "property", "jurisdiction": detected})))
        return orchestrator.orchestrator_agent(make_state(user_query="q", jurisdiction=request_jur))

    def test_detected_state_used(self, monkeypatch):
        assert self._orch(monkeypatch, "IN-UP")["jurisdiction"] == "IN-UP"

    def test_explicit_request_wins(self, monkeypatch):
        assert self._orch(monkeypatch, "IN-UP", request_jur="IN-DL")["jurisdiction"] == "IN-DL"

    @pytest.mark.parametrize("raw,expected", [("in-up", "IN-UP"), ("UP", "IN"), ("Uttar Pradesh", "IN"),
                                              ("", "IN"), (None, "IN")])
    def test_normalised(self, monkeypatch, raw, expected):
        assert self._orch(monkeypatch, raw)["jurisdiction"] == expected


class TestWiring:
    def test_research_passes_jurisdiction(self, monkeypatch):
        from backend.agents import research
        seen = []
        monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None, **kw: seen.append(kw) or [])
        research.research_agent(make_state(user_query="q", jurisdiction="IN-MH"))
        assert seen[0]["jurisdiction"] == "IN-MH"

    def test_analysis_labels_state_sources_when_state_unknown(self):
        from backend.agents.analysis import _build_prompt
        prompt = _build_prompt(make_state(user_query="q", research_results=[
            {"id": "a", "content": "UP Rent Act text", "metadata": {"jurisdiction": "IN-UP"}},
            {"id": "b", "content": "Central Act text", "metadata": {"jurisdiction": "IN"}}]))
        assert "[applies only in IN-UP] UP Rent Act text" in prompt
        assert "- Central Act text" in prompt
        assert "not specified" in prompt

    def test_analysis_states_known_jurisdiction(self):
        from backend.agents.analysis import _build_prompt
        assert "Jurisdiction: IN-UP" in _build_prompt(make_state(user_query="q", jurisdiction="IN-UP"))


class TestRealCorpus:
    """Uses the shipped corpus (3 UP-specific tenancy docs) and the local model."""

    QUERY = "landlord not returning security deposit after tenant vacated"

    @pytest.fixture(autouse=True)
    def _corpus(self):
        if hybrid_rag.collection.count() == 0:
            pytest.skip("corpus not loaded — run python -m backend.data_loader")

    def _jurs(self, jurisdiction):
        return {(d.get("metadata") or {}).get("jurisdiction") for d in
                hybrid_rag.search(self.QUERY, "property", jurisdiction=jurisdiction)}

    def test_delhi_never_gets_up_law(self):
        assert "IN-UP" not in self._jurs("IN-DL")

    def test_up_gets_up_law(self):
        assert "IN-UP" in self._jurs("IN-UP")
