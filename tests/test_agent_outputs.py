"""
Agent output schema tests — no paid APIs (OpenAI / Cohere are mocked).
Run: pytest tests/test_agent_outputs.py -v
"""
import json
from types import SimpleNamespace

import pytest

from backend.agents import schemas
from tests.test_pipeline import make_state


def fake_openai(content):
    """
    OpenAI-like class. `content` is the reply string, or a callable that
    receives the create(**kwargs) and returns the reply string.
    """
    def create(**kw):
        text = content(kw) if callable(content) else content
        msg = SimpleNamespace(content=text)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class _Client:
        def __init__(self, *a, **kw):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))
    return _Client


EXTRACTED = {
    "document_type": "legal_notice",
    "client_name": "Ramesh Kumar", "client_address": "12 MG Road, Lucknow",
    "opposite_party_name": "Suresh Gupta", "opposite_party_address": None,
    "facts": "Client vacated the flat 3 months ago; deposit of Rs 50,000 not returned.",
    "relief_sought": "Return of Rs 50,000 security deposit", "amount": "50000",
}
DRAFTED = {
    "english": "LEGAL NOTICE\nTo Suresh Gupta, [Opposite Party Address]. Under Section 406 IPC ...",
    "hindi": "कानूनी नोटिस\nसुरेश गुप्ता, [विपक्षी का पता]। धारा 406 के अंतर्गत ...",
}


def mock_drafting(monkeypatch, extraction=EXTRACTED, draft=DRAFTED):
    """Mock the drafting agent's two LLM calls (extraction, then drafting).
    Pass an Exception instance to make that call fail."""
    from backend.agents import drafting

    def reply(kw):
        system = kw["messages"][0]["content"]
        out = extraction if system == drafting.EXTRACTION_PROMPT else draft
        if isinstance(out, Exception):
            raise out
        return out if isinstance(out, str) else json.dumps(out)
    monkeypatch.setattr(drafting, "OpenAI", fake_openai(reply))


# ─── Guardian ─────────────────────────────────────────────────────────────────

def mock_guardian(monkeypatch, reply):
    """reply: dict (LLM JSON), str, or Exception instance."""
    from backend.agents import guardian

    def _reply(kw):
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)
    monkeypatch.setattr(guardian, "OpenAI", fake_openai(_reply))


class TestGuardianOutput:
    def test_normal(self, monkeypatch):
        from backend.agents.guardian import guardian_agent
        mock_guardian(monkeypatch, {"is_emergency": False, "emergency_type": "", "reasoning": "routine"})
        out = guardian_agent(make_state(user_query="landlord not returning deposit"))
        assert out["is_emergency"] is False and out["urgent_alert"] == ""
        assert out["guardian_status"] == "checked"
        schemas.GuardianOutput(**out)

    def test_critical_emergency_alert(self, monkeypatch):
        from backend.agents.guardian import guardian_agent
        mock_guardian(monkeypatch, {"is_emergency": True, "emergency_type": "domestic_violence"})
        out = guardian_agent(make_state(user_query="client facing domestic violence"))
        assert out["urgency_level"] == "CRITICAL"
        assert "PWDVA" in out["urgent_alert"]
        assert "112" in out["urgent_alert"] and "181" in out["urgent_alert"]
        assert "final_response" not in out        # guardian no longer writes the response
        schemas.GuardianOutput(**out)

    def test_non_critical_has_no_safety_helplines(self, monkeypatch):
        from backend.agents.guardian import guardian_agent
        mock_guardian(monkeypatch, {"is_emergency": True, "emergency_type": "court_emergency"})
        out = guardian_agent(make_state(user_query="hearing tomorrow"))
        assert out["urgency_level"] == "MEDIUM"
        assert "112" not in out["urgent_alert"]

    def test_unknown_type_becomes_other(self, monkeypatch):
        from backend.agents.guardian import guardian_agent
        mock_guardian(monkeypatch, {"is_emergency": True, "emergency_type": "flood",
                                    "reasoning": "limitation expires tomorrow"})
        out = guardian_agent(make_state(user_query="limitation expires tomorrow"))
        assert out["emergency_type"] == "other"
        assert "limitation expires tomorrow" in out["urgent_alert"]

    def test_type_dropped_when_not_emergency(self):
        m = schemas.GuardianLLMOutput(is_emergency=False, emergency_type="physical_threat")
        assert m.emergency_type == ""

    @pytest.mark.parametrize("failure", [RuntimeError("api down"), "not json"])
    def test_fails_open(self, monkeypatch, failure):
        from backend.agents.guardian import guardian_agent
        mock_guardian(monkeypatch, failure)
        out = guardian_agent(make_state(user_query="anything"))
        assert out["is_emergency"] is False
        assert out["guardian_status"] == "unavailable"


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
        monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None, **kw: docs)
        out = research.research_agent(make_state(user_query="deposit", legal_domain="property"))
        assert [r["id"] for r in out["research_results"]] == ["act:abc", "legal_guidance:def"]
        assert out["research_results"][1]["metadata"] == {}
        assert "rrf_score" not in out["research_results"][1]   # None dropped
        assert out["kanoon_cases"] == []
        # statutory refs come from doc_type metadata (none here), not a text match on "section"
        assert out["ipc_sections"] == []
        assert out["source_status"]["indian_kanoon"] == "not_configured"
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

RESEARCH = [{"id": "ipc_section:1", "content": "Section 406 IPC / Section 316 BNS - Criminal breach of trust"}]


class TestDraftingOutput:
    def test_llm_draft_with_extracted_fields(self, monkeypatch):
        from backend.agents.drafting import drafting_agent
        mock_drafting(monkeypatch)
        out = drafting_agent(make_state(user_query="draft notice", research_results=RESEARCH))
        assert out["draft_method"] == "llm"
        assert out["document_type"] == "legal_notice"
        assert out["draft_document"] == DRAFTED["english"]
        assert out["draft_missing_fields"] == ["opposite_party_address"]
        assert out["draft_unverified_citations"] == []      # 406 is in research
        schemas.DraftingOutput(**out)

    def test_hindi_language_selects_hindi(self, monkeypatch):
        from backend.agents.drafting import drafting_agent
        mock_drafting(monkeypatch)
        out = drafting_agent(make_state(user_query="नोटिस", language="hi", research_results=RESEARCH))
        assert out["draft_document"] == DRAFTED["hindi"]

    def test_invented_citation_flagged(self, monkeypatch):
        from backend.agents.drafting import drafting_agent
        mock_drafting(monkeypatch, draft={
            "english": "Notice under Section 420 IPC and Section 406 IPC, see also Article 21.",
            "hindi": "धारा 420 और धारा 406 के अंतर्गत नोटिस, अनुच्छेद 21 भी देखें।",
        })
        out = drafting_agent(make_state(user_query="draft notice", research_results=RESEARCH))
        assert out["draft_unverified_citations"] == ["21", "420"]

    def test_rti_template_provision_allowed(self, monkeypatch):
        from backend.agents.drafting import drafting_agent
        mock_drafting(monkeypatch,
                      extraction={"document_type": "rti_application", "facts": "status of PMAY application"},
                      draft={"english": "Application under Section 6(1) of the RTI Act, 2005 ...",
                             "hindi": "सूचना का अधिकार अधिनियम की धारा 6(1) के अंतर्गत आवेदन ..."})
        out = drafting_agent(make_state(user_query="RTI for PMAY"))
        assert out["document_type"] == "rti_application"
        assert out["draft_unverified_citations"] == []

    def test_drafting_failure_falls_back_to_template(self, monkeypatch):
        from backend.agents.drafting import drafting_agent
        mock_drafting(monkeypatch, draft=RuntimeError("api down"))
        out = drafting_agent(make_state(user_query="draft notice"))
        assert out["draft_method"] == "template_fallback"
        assert "Ramesh Kumar" in out["document_english"]           # extracted value used
        assert "[Opposite Party Address]" in out["document_english"]  # missing kept as placeholder
        assert "{" not in out["document_english"]                 # no unfilled template keys
        schemas.DraftingOutput(**out)

    def test_total_failure_gives_placeholder_template(self, monkeypatch):
        from backend.agents.drafting import drafting_agent, REQUIRED_FIELDS
        mock_drafting(monkeypatch, extraction=RuntimeError("down"), draft=RuntimeError("down"))
        out = drafting_agent(make_state(user_query="draft notice"))
        assert out["draft_method"] == "template_fallback"
        assert out["document_type"] == "legal_notice"
        assert out["draft_missing_fields"] == REQUIRED_FIELDS["legal_notice"]
        assert "[Client Name]" in out["document_english"]

    def test_short_llm_draft_rejected(self, monkeypatch):
        from backend.agents.drafting import drafting_agent
        mock_drafting(monkeypatch, draft={"english": "ok", "hindi": "ok"})
        out = drafting_agent(make_state(user_query="draft notice"))
        assert out["draft_method"] == "template_fallback"

    def test_other_document_type_fallback(self, monkeypatch):
        from backend.agents.drafting import drafting_agent
        mock_drafting(monkeypatch,
                      extraction={"document_type": "affidavit", "document_title": "Affidavit of residence"},
                      draft=RuntimeError("down"))
        out = drafting_agent(make_state(user_query="draft affidavit"))
        assert out["document_type"] == "other"
        assert "AFFIDAVIT OF RESIDENCE" in out["document_english"]

    def test_date_is_current_not_import_time(self, monkeypatch):
        from backend.agents import drafting
        monkeypatch.setattr(drafting, "_today", lambda: "01/01/2030")
        mock_drafting(monkeypatch, draft=RuntimeError("down"))
        out = drafting.drafting_agent(make_state(user_query="draft notice"))
        assert "01/01/2030" in out["document_english"]


class TestDraftFields:
    @pytest.mark.parametrize("raw", ["", "  ", "null", "Unknown", "N/A", None])
    def test_blank_markers_become_none(self, raw):
        assert schemas.DraftFields(client_name=raw).client_name is None

    def test_unknown_doc_type_becomes_other(self):
        assert schemas.DraftFields(document_type="Affidavit").document_type == "other"

    def test_doc_type_normalized(self):
        assert schemas.DraftFields(document_type=" RTI_Application ").document_type == "rti_application"

    def test_numbers_coerced_to_str(self):
        assert schemas.DraftFields(amount=50000).amount == "50000"


class TestCitationGuard:
    def test_hindi_and_english_forms(self):
        from backend.agents.drafting import find_unverified_citations
        ctx = "Section 498A IPC; धारा 125"
        assert find_unverified_citations(["section 498a and धारा 125 and s. 302"], ctx) == ["302"]

    def test_subsections_match_base(self):
        from backend.agents.drafting import find_unverified_citations
        assert find_unverified_citations(["Section 6(1)"], "Section 6(1) of RTI") == []


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


class TestEmergencyContinues:
    @pytest.mark.asyncio
    async def test_emergency_runs_full_pipeline_with_alert_first(self, monkeypatch):
        from backend.agents import orchestrator, analysis, research
        from backend.pipeline import run_pipeline
        mock_guardian(monkeypatch, {"is_emergency": True, "emergency_type": "anticipatory_bail_urgent"})
        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(json.dumps({
            "legal_domain": "criminal", "requested_actions": ["research"]})))
        monkeypatch.setattr(analysis, "OpenAI", fake_openai("Bail analysis " * 30))
        monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None, **kw: [
            {"id": "g:1", "content": "Bail types in India: anticipatory bail under Section 438 CrPC",
             "metadata": {"domain": "criminal"}}])

        r = await run_pipeline("police may arrest my client tomorrow", "s", "u")
        assert r["is_emergency"] is True and r["urgency_level"] == "HIGH"
        assert r["research_results"] and r["case_analysis"]           # not cut short
        assert r["final_response"].startswith(r["urgent_alert"])
        assert "Bail analysis" in r["final_response"]
        assert "Disclaimer" in r["final_response"]


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
        mock_drafting(monkeypatch)
        monkeypatch.setattr(research.hybrid_rag, "search", lambda q, d=None, **kw: [
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
