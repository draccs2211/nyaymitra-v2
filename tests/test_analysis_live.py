"""
Live analysis checks against real OpenAI (gpt-4o-mini), 1 call per test.
Run only deliberately: pytest tests/test_analysis_live.py -m paid -v -s
"""
import re

import pytest

from tests.test_pipeline import make_state

pytestmark = pytest.mark.paid

RESEARCH = [
    {"id": "ipc_section:1", "content": "Section 406 IPC / Section 316 BNS - Criminal breach of trust: "
     "imprisonment up to 3 years or fine or both."},
    {"id": "legal_guidance:1", "content": "Security deposit return: tenant can file complaint in Rent "
     "Control Court; legal notice may be sent before filing."},
]

VERDICT = re.compile(r"\b(strong|weak)\s+case\b|\bcase is (strong|weak)\b|\blikely to (win|lose|succeed|fail)\b",
                     re.IGNORECASE)


@pytest.mark.parametrize("query,language", [
    ("Client vacated a rented flat in Lucknow 3 months ago; landlord refuses to return Rs 50,000 deposit. "
     "How strong is our case?", "en"),
    ("मुवक्किल ने 3 महीने पहले लखनऊ में किराए का फ्लैट खाली किया; मकान मालिक 50,000 रुपये जमा राशि नहीं लौटा रहा।", "hi"),
])
def test_structured_no_verdict(query, language):
    from backend.agents.analysis import analysis_agent
    out = analysis_agent(make_state(user_query=query, language=language, legal_domain="property",
                                    research_results=RESEARCH))
    print("\n" + out["final_response"][:1500])
    print("unverified:", out["analysis_unverified_citations"])
    ca = out["case_analysis"]
    assert "error" not in ca, ca.get("error")
    assert ca["summary"] and ca["missing_facts"] and ca["questions_for_lawyer"]
    assert not VERDICT.search(out["final_response"]), "verdict language in analysis"
    if language == "hi":
        assert re.search(r"[ऀ-ॿ]", ca["summary"])
