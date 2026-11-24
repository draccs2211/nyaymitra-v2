"""
Human approval tests (CLAUDE.md issue #5) — LangGraph interrupt/resume.
All LLM and retrieval calls mocked; no paid APIs.
Run: pytest tests/test_approval.py -v
"""
import json

import pytest
from pydantic import ValidationError

from backend.agents.schemas import ApprovalDecision
from tests.test_agent_outputs import analysis_json, fake_openai, mock_drafting


@pytest.fixture
def mocked_agents(monkeypatch):
    """Mock LLMs + retrieval; `actions` controls whether a draft (=> approval) happens."""
    from backend.agents import orchestrator, analysis, research

    def configure(actions):
        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(json.dumps({
            "legal_domain": "property", "requested_actions": actions, "language": "en",
        })))
        monkeypatch.setattr(analysis, "OpenAI", fake_openai(analysis_json()))
        mock_drafting(monkeypatch)
        monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None, **kw: [
            {"id": "act:1", "content": "Section 406 IPC", "metadata": {"domain": "criminal"}},
        ] * 5)
    return configure


def approve(**kw):
    return {"decision": "approved", "reviewer_id": "rev-1", **kw}


class TestApprovalFlow:
    @pytest.mark.asyncio
    async def test_draft_pauses_for_approval(self, mocked_agents):
        from backend.pipeline import run_pipeline
        mocked_agents(["research", "drafting"])
        r = await run_pipeline("draft notice", "s", "u")
        assert r["approval_status"] == "pending"
        assert r["run_id"]
        assert r["draft_document"]
        assert r["approval"] == {}

    @pytest.mark.asyncio
    async def test_approve_completes_run(self, mocked_agents):
        from backend.pipeline import run_pipeline, resume_pipeline, get_run
        mocked_agents(["research", "drafting"])
        r = await run_pipeline("draft notice", "s", "u")
        done = await resume_pipeline(r["run_id"], approve(notes="looks fine"))
        assert done["approval_status"] == "approved"
        assert done["approval"]["reviewer_id"] == "rev-1"
        assert done["approval"]["edited"] is False
        assert done["final_response"] == r["final_response"]   # unchanged
        assert (await get_run(r["run_id"]))["approval_status"] == "approved"

    @pytest.mark.asyncio
    async def test_approve_with_edit_replaces_response(self, mocked_agents):
        from backend.pipeline import run_pipeline, resume_pipeline
        mocked_agents(["research", "drafting"])
        r = await run_pipeline("draft notice", "s", "u")
        done = await resume_pipeline(r["run_id"], approve(edited_response="Corrected text"))
        assert done["final_response"] == "Corrected text"
        assert done["approval"]["edited"] is True

    @pytest.mark.asyncio
    async def test_reject(self, mocked_agents):
        from backend.pipeline import run_pipeline, resume_pipeline
        mocked_agents(["research", "drafting"])
        r = await run_pipeline("draft notice", "s", "u")
        done = await resume_pipeline(r["run_id"], {
            "decision": "rejected", "reviewer_id": "rev-1", "notes": "wrong section"})
        assert done["approval_status"] == "rejected"
        assert done["approval"]["notes"] == "wrong section"

    @pytest.mark.asyncio
    async def test_cannot_resume_twice(self, mocked_agents):
        from backend.pipeline import run_pipeline, resume_pipeline
        mocked_agents(["research", "drafting"])
        r = await run_pipeline("draft notice", "s", "u")
        await resume_pipeline(r["run_id"], approve())
        with pytest.raises(ValueError):
            await resume_pipeline(r["run_id"], approve())

    @pytest.mark.asyncio
    async def test_invalid_decision_keeps_run_pending(self, mocked_agents):
        from backend.pipeline import run_pipeline, resume_pipeline, get_run
        mocked_agents(["research", "drafting"])
        r = await run_pipeline("draft notice", "s", "u")
        with pytest.raises(ValidationError):
            await resume_pipeline(r["run_id"], {"decision": "maybe", "reviewer_id": "x"})
        assert (await get_run(r["run_id"]))["approval_status"] == "pending"
        done = await resume_pipeline(r["run_id"], approve())
        assert done["approval_status"] == "approved"

    @pytest.mark.asyncio
    async def test_unknown_run(self):
        from backend.pipeline import resume_pipeline, get_run
        assert await get_run("does-not-exist") is None
        with pytest.raises(KeyError):
            await resume_pipeline("does-not-exist", approve())

    @pytest.mark.asyncio
    async def test_runs_are_isolated(self, mocked_agents):
        from backend.pipeline import run_pipeline, resume_pipeline, get_run
        mocked_agents(["research", "drafting"])
        a = await run_pipeline("draft A", "s", "u")
        b = await run_pipeline("draft B", "s", "u")
        await resume_pipeline(a["run_id"], approve())
        assert (await get_run(b["run_id"]))["approval_status"] == "pending"

    @pytest.mark.asyncio
    async def test_guardian_unavailable_still_completes(self, mocked_agents):
        """Guardian LLM is blocked (unmocked) in tests -> fails open, run still finishes."""
        from backend.pipeline import run_pipeline
        mocked_agents(["research"])
        r = await run_pipeline("what is section 406", "s", "u")
        assert r["guardian_status"] == "unavailable"
        assert r["is_emergency"] is False
        assert r["approval_status"] in ("not_required", "pending")
        assert r["case_analysis"]

class TestApprovalDecision:
    def test_reject_requires_notes(self):
        with pytest.raises(ValidationError):
            ApprovalDecision(decision="rejected", reviewer_id="r")

    def test_edit_only_on_approve(self):
        with pytest.raises(ValidationError):
            ApprovalDecision(decision="rejected", reviewer_id="r", notes="x", edited_response="y")

    def test_reviewer_required(self):
        with pytest.raises(ValidationError):
            ApprovalDecision(decision="approved", reviewer_id="")
