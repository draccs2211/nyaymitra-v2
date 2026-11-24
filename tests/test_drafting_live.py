"""
Live drafting checks against real OpenAI (gpt-4o-mini). ~2 calls per test.
Run only deliberately: pytest tests/test_drafting_live.py -m paid -v
"""
import re

import pytest

from tests.test_pipeline import make_state

pytestmark = pytest.mark.paid

RESEARCH = [{"id": "ipc_section:1",
             "content": "Section 406 IPC / Section 316 BNS - Criminal breach of trust"}]


def test_legal_notice_uses_stated_facts_only():
    from backend.agents.drafting import drafting_agent
    out = drafting_agent(make_state(
        user_query=("Draft a legal notice for my client Ramesh Kumar of 12 MG Road, Lucknow "
                    "to his former landlord Suresh Gupta, who has not returned the Rs 50,000 "
                    "security deposit 3 months after Ramesh vacated the flat."),
        research_results=RESEARCH,
    ))
    print("\n--- EN ---\n", out["document_english"][:1200])
    print("missing:", out["draft_missing_fields"], "unverified:", out["draft_unverified_citations"])
    assert out["draft_method"] == "llm"
    assert out["document_type"] == "legal_notice"
    en = out["document_english"]
    assert "Ramesh Kumar" in en and "Suresh Gupta" in en
    assert "50,000" in en or "50000" in en
    # landlord's address and client's phone were never given -> must stay missing, not invented
    assert "opposite_party_address" in out["draft_missing_fields"]
    assert re.search(r"\[[^\]]*address[^\]]*\]", en, re.I), "missing address line dropped instead of placeholder"
    assert not re.search(r"\b[6-9]\d{9}\b", en), "invented Indian mobile number"
    assert re.search(r"[ऀ-ॿ]", out["document_hindi"]), "Hindi draft not in Devanagari"


def test_rti_application_detected():
    from backend.agents.drafting import drafting_agent
    out = drafting_agent(make_state(
        user_query="Draft an RTI application asking the Lucknow Development Authority for the status of my PM Awas Yojana application",
    ))
    print("\n--- EN ---\n", out["document_english"][:800])
    print("missing:", out["draft_missing_fields"], "unverified:", out["draft_unverified_citations"])
    assert out["document_type"] == "rti_application"
    assert out["draft_method"] == "llm"
    assert "client_name" in out["draft_missing_fields"]
