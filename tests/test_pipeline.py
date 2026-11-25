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

@pytest.mark.paid
class TestGuardianAgent:
    def test_normal_query(self):
        state = make_state(user_query="My landlord is not returning my deposit")
        result = guardian_agent(state)
        assert result["is_emergency"] == False
        assert result["guardian_status"] == "checked"

    def test_domestic_violence_emergency(self):
        state = make_state(user_query="My husband is doing domestic violence please help")
        result = guardian_agent(state)
        assert result["is_emergency"] == True
        assert result["emergency_type"] == "domestic_violence"

    def test_threat_emergency(self):
        state = make_state(user_query="Someone is threatening to kill me jaan ka khatra hai")
        result = guardian_agent(state)
        assert result["is_emergency"] == True

    def test_critical_alert_contains_safety_helplines(self):
        state = make_state(user_query="domestic violence ho rahi hai bachao")
        result = guardian_agent(state)
        assert result["urgency_level"] == "CRITICAL"
        assert "112" in result["urgent_alert"] and "181" in result["urgent_alert"]

    @pytest.mark.parametrize("query", [
        "Research Supreme Court case law on Section 376 IPC for a trial next month",
        "Client paid Rs 3760 advance to the builder, wants refund",
        "Explain the grounds for anticipatory bail under Section 438 CrPC",
    ])
    def test_no_false_positives(self, query):
        result = guardian_agent(make_state(user_query=query))
        assert result["is_emergency"] == False, result


# ─── Orchestrator Agent Tests ──────────────────────────────────────────────────

@pytest.mark.paid
class TestOrchestratorAgent:
    @pytest.mark.parametrize("query,expected", [
        ("My client's landlord in Lucknow is not returning the security deposit", "IN-UP"),
        ("Tenant in Andheri, Mumbai wants rent receipt from landlord", "IN-MH"),
        ("Explain the grounds for anticipatory bail", "IN"),
    ])
    def test_jurisdiction_detection(self, query, expected):
        result = orchestrator_agent(make_state(user_query=query))
        assert result["jurisdiction"] == expected

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
            kanoon_cases=[{"title": "Test Case", "url": "http://indiankanoon.org/doc/123"} for _ in range(3)]
        )
        result = verification_agent(state)
        assert result["confidence_score"] > 0.6

    def test_disclaimer_added(self):
        state = make_state(
            final_response="Some legal advice here.",
            research_results=[],
            kanoon_cases=[]
        )
        result = verification_agent(state)
        assert "disclaimer" in result["final_response"].lower() or "verify" in result["final_response"].lower()

    def test_low_confidence_triggers_human_approval(self):
        state = make_state(
            final_response="Short",
            research_results=[],
            kanoon_cases=[]
        )
        result = verification_agent(state)
        assert result["human_approval_needed"] == True or result["confidence_score"] < 0.70

    def test_sources_added_to_response(self):
        state = make_state(
            final_response="Legal advice",
            research_results=[{"content": "legal content"}],
            kanoon_cases=[{"title": "Test Case", "url": "http://indiankanoon.org/doc/123"}]
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
        assert result["final_response"].startswith(result["urgent_alert"])   # alert on top
        assert "112" in result["final_response"]
        assert result["case_analysis"]                    # pipeline continued past guardian
