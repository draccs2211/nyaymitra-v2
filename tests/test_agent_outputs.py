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
            "legal_domain": "property", "requested_actions": ["research", "drafting"],
            "language": "hinglish", "urgency": "medium", "key_issues": ["deposit"],
            "suggested_acts": ["UP Rent Act"], "reasoning": "tenant issue",
        })
        assert out["legal_domain"] == "property"
        assert out["requested_actions"] == ["research", "drafting"]
        assert out["language"] == "hinglish"
        assert out["classification"]["key_issues"] == ["deposit"]
        assert "case_analysis" not in out
        schemas.OrchestratorOutput(**out)

    def test_invalid_enum_values_coerced(self, monkeypatch):
        out = self._run(monkeypatch, {
            "legal_domain": "Tax Law", "language": "tamil", "urgency": "ASAP",
            "requested_actions": ["Drafting", "drafting", "translate", "REFERRAL"],
            "key_issues": "single issue",
        })
        assert out["legal_domain"] == "general"
        assert out["language"] == "en"
        assert out["classification"]["urgency"] == "low"
        assert out["requested_actions"] == ["drafting", "referral"]
        assert out["classification"]["key_issues"] == ["single issue"]

    @pytest.mark.parametrize("actions", [[], None, ["unknown"], "nonsense"])
    def test_empty_or_unknown_actions_default_to_research(self, monkeypatch, actions):
        out = self._run(monkeypatch, {"legal_domain": "civil", "requested_actions": actions})
        assert out["requested_actions"] == ["research"]

    def test_single_string_action_accepted(self, monkeypatch):
        out = self._run(monkeypatch, {"requested_actions": "referral"})
        assert out["requested_actions"] == ["referral"]

    def test_case_normalized(self, monkeypatch):
        out = self._run(monkeypatch, {"legal_domain": " Criminal ", "language": "HI"})
        assert out["legal_domain"] == "criminal"
        assert out["language"] == "hi"

    def test_non_json_falls_back(self, monkeypatch):
        out = self._run(monkeypatch, "not json at all")
        assert out["classification"]["reasoning"] == "Fallback classification used"
        assert out["requested_actions"] == ["research"]
        schemas.OrchestratorOutput(**out)


# ─── Research ─────────────────────────────────────────────────────────────────

class TestResearchOutput:
    def test_rag_results_validated(self, monkeypatch):
        from backend.agents import research
        docs = [
            {"id": "act:abc", "content": "Section 406 IPC breach of trust",
             "metadata": {"domain": "criminal"}, "score": 0.8, "source": "dense",
             "rrf_score": 0.03},
            {"id": "legal_guidance:def", "content": "Security deposit guidance",
             "score": 2.1, "source": "sparse"},
        ]
        monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None: docs)
        out = research.research_agent(make_state(user_query="deposit", legal_domain="property"))
        assert [r["id"] for r in out["research_results"]] == ["act:abc", "legal_guidance:def"]
        assert out["research_results"][1]["metadata"] == {}
        assert "rrf_score" not in out["research_results"][1]   # None dropped
        assert out["kanoon_cases"] == []
        assert [s["id"] for s in out["ipc_sections"]] == ["act:abc"]
        schemas.ResearchOutput(**out)

    def test_kanoon_case_coerces_ids(self):
        c = schemas.KanoonCase(title="X v Y", tid=12345, date=None)
        assert c.tid == "12345" and c.date == ""
        assert "verify" in c.source.lower()


# ─── Analysis ─────────────────────────────────────────────────────────────────

class TestAnalysisOutput:
    def test_llm_success_validates(self, monkeypatch):
        from backend.agents import analysis
        text = "Applicable: Section 406 IPC.\n1. Send legal notice to landlord\n2. File in Rent Control Court"
        monkeypatch.setattr(analysis, "OpenAI", fake_openai(text))
        out = analysis.analysis_agent(make_state(user_query="deposit", legal_domain="property"))
        assert out["final_response"] == text
        assert out["case_analysis"]["domain"] == "property"
        assert "Send legal notice to landlord" in out["next_steps"]
        schemas.AnalysisOutput(**out)

    def test_llm_failure_fallback_validates(self, monkeypatch):
        from backend.agents import analysis

        class Boom:
            def __init__(self, *a, **kw):
                raise RuntimeError("api down")
        monkeypatch.setattr(analysis, "OpenAI", Boom)
        out = analysis.analysis_agent(make_state(user_query="q"))
        assert out["case_analysis"]["error"] == "api down"
        assert out["case_strength"] == "medium"
        schemas.AnalysisOutput(**out)


# ─── Drafting / Referral / Verification ──────────────────────────────────────

class TestDraftingOutput:
    @pytest.mark.parametrize("query,lang,expected", [
        ("Draft legal notice to landlord", "en", "legal_notice"),
        ("RTI application for scheme status", "en", "rti_application"),
        ("मकान मालिक को कानूनी नोटिस", "hi", "legal_notice"),
    ])
    def test_validates(self, query, lang, expected):
        from backend.agents.drafting import drafting_agent
        out = drafting_agent(make_state(user_query=query, language=lang))
        assert out["document_type"] == expected
        expected_draft = out["document_hindi"] if lang == "hi" else out["document_english"]
        assert out["draft_document"] == expected_draft
        schemas.DraftingOutput(**out)


class TestReferralOutput:
    def test_appends_and_validates(self):
        from backend.agents.referral import referral_agent
        out = referral_agent(make_state(final_response="Prior analysis"))
        assert out["final_response"].startswith("Prior analysis")
        assert out["dlsa_offices"][0]["helpline"] == "15100"
        schemas.ReferralOutput(**out)


class TestVerificationOutput:
    def test_validates(self):
        from backend.agents.verification import verification_agent
        out = verification_agent(make_state(final_response="x" * 400))
        assert 0.0 <= out["confidence_score"] <= 1.0
        schemas.VerificationOutput(**out)


# ─── End-to-end (all external calls mocked) ──────────────────────────────────

class TestRouting:
    """Pure routing functions — no agents executed."""

    @pytest.mark.parametrize("actions,after_orch,after_analysis,after_drafting", [
        (["research"],                          "research",      "verification", "verification"),
        (["drafting"],                          "research",      "drafting",     "verification"),
        (["referral"],                          "referral_only", "referral",     "referral"),
        (["research", "referral"],              "research",      "referral",     "referral"),
        (["drafting", "referral"],              "research",      "drafting",     "referral"),
        (["referral", "drafting", "research"],  "research",      "drafting",     "referral"),
        ([],                                    "research",      "verification", "verification"),
    ])
    def test_routes(self, actions, after_orch, after_analysis, after_drafting):
        from backend import pipeline
        st = make_state(requested_actions=actions)
        assert pipeline.route_after_orchestrator(st) == after_orch
        assert pipeline.route_after_analysis(st) == after_analysis
        assert pipeline.route_after_drafting(st) == after_drafting


class TestPipelineWithSchemas:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("actions", [
        ["research"],
        ["research", "drafting", "referral"],
        ["referral"],
    ])
    async def test_full_graph(self, monkeypatch, actions):
        from backend.agents import orchestrator, analysis, research
        from backend.pipeline import run_pipeline

        prompts = []

        def analysis_llm(content):
            cls = fake_openai(content)
            class Spy(cls):
                def __init__(self, *a, **kw):
                    super().__init__()
                    create = self.chat.completions.create
                    def spy(**kw):
                        prompts.append(kw["messages"][-1]["content"])
                        return create(**kw)
                    self.chat.completions.create = spy
            return Spy

        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(json.dumps({
            "legal_domain": "property", "requested_actions": actions,
            "language": "en", "urgency": "low", "key_issues": ["unreturned deposit"],
            "suggested_acts": ["UP Rent Act"],
        })))
        monkeypatch.setattr(analysis, "OpenAI", analysis_llm("Analysis text " * 30))
        monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None: [
            {"id": "act:1", "content": "Section 406 IPC", "metadata": {"domain": "criminal"}},
        ])

        result = await run_pipeline("landlord kept deposit", "s1", "u1")
        assert result["legal_domain"] == "property"
        assert result["requested_actions"] == actions
        # classification survives analysis (no longer overwritten)
        assert result["classification"]["key_issues"] == ["unreturned deposit"]
        assert "Disclaimer" in result["final_response"]          # verification always runs

        ran_research = actions != ["referral"]
        assert bool(result["research_results"]) is ran_research
        assert bool(result["case_analysis"]) is ran_research
        if ran_research:
            assert "unreturned deposit" in prompts[0]           # issues fed to analysis
        assert bool(result["draft_document"]) is ("drafting" in actions)
        assert bool(result["dlsa_offices"]) is ("referral" in actions)
