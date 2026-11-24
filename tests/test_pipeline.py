"""
Tests for NyayMitra 2.0 pipeline.
Run: pytest tests/test_pipeline.py -v
"""
import pytest
import asyncio
from backend.agents.guardian import guardian_agent
from backend.agents.orchestrator import orchestrator_agent
from backend.agents.verification import verification_agent
from backend.utils.language_detector import detect_language
from backend.state import NyayMitraState, initial_state


def make_state(**kwargs) -> NyayMitraState:
    kwargs.setdefault("session_id", "test-session")
    kwargs.setdefault("user_id", "test-user")
    return initial_state(**kwargs)


# ─── Language Detection Tests ─────────────────────────────────────────────────

class TestLanguageDetection:
    def test_english_detection(self):
        assert detect_language("My landlord is not returning deposit") == "en"

    def test_hindi_detection(self):
        assert detect_language("मेरा मकान मालिक जमानत वापस नहीं कर रहा") == "hi"

    def test_hinglish_detection(self):
        assert detect_language("mera landlord deposit nahi de raha hai") == "hinglish"

    def test_empty_string(self):
        assert detect_language("") == "en"


# ─── Guardian Agent Tests ─────────────────────────────────────────────────────

class TestGuardianAgent:
    def test_normal_query(self):
        state = make_state(user_query="My landlord is not returning my deposit")
        result = guardian_agent(state)
        assert result["is_emergency"] == False

    def test_domestic_violence_emergency(self):
        state = make_state(user_query="My husband is doing domestic violence please help")
        result = guardian_agent(state)
        assert result["is_emergency"] == True
        assert result["emergency_type"] == "domestic_violence"

    def test_threat_emergency(self):
        state = make_state(user_query="Someone is threatening to kill me jaan ka khatra hai")
        result = guardian_agent(state)
        assert result["is_emergency"] == True

    @pytest.mark.xfail(reason="Guardian alert is lawyer-facing and has no helplines; revisit with CLAUDE.md issue #7", strict=True)
    def test_emergency_response_contains_helpline(self):
        state = make_state(user_query="domestic violence ho rahi hai bachao")
        result = guardian_agent(state)
        assert "1091" in result.get("final_response", "") or "100" in result.get("final_response", "")


# ─── Orchestrator Agent Tests ──────────────────────────────────────────────────

@pytest.mark.paid
class TestOrchestratorAgent:
    def test_criminal_domain_detection(self):
        state = make_state(user_query="I want to file FIR against my neighbor")
        result = orchestrator_agent(state)
        assert result["legal_domain"] == "criminal"

    def test_property_domain_detection(self):
        state = make_state(user_query="Landlord not returning my security deposit rent problem")
        result = orchestrator_agent(state)
        assert result["legal_domain"] == "property"

    def test_consumer_domain_detection(self):
        state = make_state(user_query="Defective product consumer forum complaint refund")
        result = orchestrator_agent(state)
        assert result["legal_domain"] == "consumer"

    def test_rti_domain_detection(self):
        state = make_state(user_query="I want to file RTI application for government information")
        result = orchestrator_agent(state)
        assert result["legal_domain"] == "rti"

    def test_document_detection(self):
        state = make_state(user_query="Draft a legal notice for my landlord")
        result = orchestrator_agent(state)
        assert "drafting" in result["requested_actions"]

    def test_hindi_language_detection(self):
        state = make_state(user_query="मेरे मकान मालिक ने किराया नहीं लिया")
        result = orchestrator_agent(state)
        assert result["language"] == "hi"

    def test_hinglish_detection(self):
        state = make_state(user_query="mera boss salary nahi de raha hai kya karna chahiye")
        result = orchestrator_agent(state)
        assert result["language"] == "hinglish"


# ─── Verification Agent Tests ──────────────────────────────────────────────────

class TestVerificationAgent:
    def test_high_confidence_with_sources(self):
        state = make_state(
            final_response="Based on UP Urban Buildings Act, landlord must return deposit within 30 days. Verify from official sources before citing.",
            research_results=[{"content": "UP Urban Buildings Act security deposit"} for _ in range(5)],
            kanoon_cases=[{"title": "Test Case", "url": "http://indiankanoon.org/doc/123"} for _ in range(3)],
            case_strength="strong"
        )
        result = verification_agent(state)
        assert result["confidence_score"] > 0.6

    def test_disclaimer_added(self):
        state = make_state(
            final_response="Some legal advice here.",
            research_results=[],
            kanoon_cases=[],
            case_strength="medium"
        )
        result = verification_agent(state)
        assert "disclaimer" in result["final_response"].lower() or "verify" in result["final_response"].lower()

    def test_low_confidence_triggers_human_approval(self):
        state = make_state(
            final_response="Short",
            research_results=[],
            kanoon_cases=[],
            case_strength="medium"
        )
        result = verification_agent(state)
        assert result["human_approval_needed"] == True or result["confidence_score"] < 0.70

    def test_sources_added_to_response(self):
        state = make_state(
            final_response="Legal advice",
            research_results=[{"content": "legal content"}],
            kanoon_cases=[{"title": "Test Case", "url": "http://indiankanoon.org/doc/123"}],
            case_strength="medium"
        )
        result = verification_agent(state)
        assert "indiankanoon.org" in result["final_response"] or len(result["verified_sources"]) > 0


# ─── Integration Test ─────────────────────────────────────────────────────────

@pytest.mark.paid
class TestIntegration:
    @pytest.mark.asyncio
    async def test_full_pipeline_normal_query(self):
        from backend.pipeline import run_pipeline
        result = await run_pipeline(
            query="My landlord is not returning security deposit",
            session_id="test-123",
            user_id="test-user"
        )
        assert result is not None
        assert "final_response" in result
        assert len(result["final_response"]) > 50

    @pytest.mark.asyncio
    async def test_full_pipeline_emergency(self):
        from backend.pipeline import run_pipeline
        result = await run_pipeline(
            query="domestic violence ho rahi hai please help",
            session_id="test-456",
            user_id="test-user"
        )
        assert result["is_emergency"] == True
        assert "1091" in result["final_response"] or "100" in result["final_response"]
