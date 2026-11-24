"""
Verification Agent - NyayMitra 2.0
Checks confidence, detects hallucinations, adds disclaimers and sources.
"""
from backend.state import NyayMitraState, initial_state
from backend.config import CONFIDENCE_THRESHOLD
from backend.agents.schemas import VerificationOutput, to_state
from langsmith import traceable
from typing import List, Dict
import re


HALLUCINATION_PATTERNS = [
    r"section \d+ of ipc",
    r"section \d+ of bns",
    r"section \d+ of crpc",
    r"\d{4} scc \d+",
    r"air \d{4}",
]


def check_citation_format(text: str) -> List[str]:
    found = []
    for pattern in HALLUCINATION_PATTERNS:
        matches = re.findall(pattern, text.lower())
        found.extend(matches)
    return found


def calculate_confidence(
    research_results: List[Dict],
    kanoon_cases: List[Dict],
    final_response: str,
    case_strength: str
) -> float:
    score = 0.5

    if len(research_results) >= 5:
        score += 0.15
    elif len(research_results) >= 2:
        score += 0.08

    if len(kanoon_cases) >= 3:
        score += 0.15
    elif len(kanoon_cases) >= 1:
        score += 0.08

    if len(final_response) > 300:
        score += 0.10
    elif len(final_response) < 100:
        score -= 0.15

    if case_strength in ["strong", "weak"]:
        score += 0.05

    citations = check_citation_format(final_response)
    if citations:
        score += min(0.10, len(citations) * 0.03)

    if any(word in final_response.lower() for word in [
        "verify", "consult", "lawyer", "dlsa", "legal aid", "official source"
    ]):
        score += 0.05

    return min(0.98, max(0.30, round(score, 2)))


def build_sources_list(
    kanoon_cases: List[Dict],
    research_results: List[Dict]
) -> List[str]:
    sources = []
    for case in kanoon_cases[:3]:
        url = case.get("url", "")
        title = case.get("title", "Unknown Case")
        if url:
            sources.append(f"{title} — {url} (Discovery only — verify before citing)")
    if research_results:
        sources.append("NyayMitra Legal Database (IPC/BNS sections and legal corpus)")
    if not sources:
        sources.append("General legal knowledge — verify from official sources")
    return sources


@traceable(name="Verification Agent")
def verification_agent(state: NyayMitraState) -> dict:
    final_response = state.get("final_response", "")
    research_results = state.get("research_results", [])
    kanoon_cases = state.get("kanoon_cases", [])
    case_strength = state.get("case_strength", "medium")
    draft_document = state.get("draft_document", "")
    language = state.get("language", "en")

    # Calculate confidence
    confidence = calculate_confidence(
        research_results, kanoon_cases, final_response, case_strength
    )

    # Check hallucination risk
    hallucination_detected = False
    citations_in_response = check_citation_format(final_response)
    if citations_in_response and not kanoon_cases and not research_results:
        hallucination_detected = True
        confidence = min(confidence, 0.55)

    # Build sources
    sources = build_sources_list(kanoon_cases, research_results)

    # Add disclaimer
    disclaimer_hi = "\n\n⚠️ अस्वीकरण: यह जानकारी सामान्य कानूनी मार्गदर्शन के लिए है। न्यायालय में उद्धृत करने से पहले आधिकारिक स्रोतों से सत्यापित करें।"
    disclaimer_en = "\n\n⚠️ Disclaimer: This information is for general legal guidance only. Verify from official sources before citing in court. Consult a qualified lawyer for specific advice."

    enhanced_response = final_response + (disclaimer_hi if language == "hi" else disclaimer_en)

    # Add sources
    if sources:
        sources_text = "\n\n📚 Sources:\n" + "\n".join(f"• {s}" for s in sources)
        enhanced_response += sources_text

    human_approval_needed = (
        confidence < CONFIDENCE_THRESHOLD or
        hallucination_detected or
        bool(draft_document)
    )

    print(f"[Verification] Confidence: {confidence} | Hallucination: {hallucination_detected} | Human approval: {human_approval_needed}")

    return to_state(VerificationOutput(
        confidence_score=confidence,
        hallucination_detected=hallucination_detected,
        human_approval_needed=human_approval_needed,
        approval_status="pending" if human_approval_needed else "not_required",
        verified_sources=sources,
        final_response=enhanced_response,
    ))


# ── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    test_state = initial_state(
        user_query="My landlord has not returned security deposit of Rs 50000 after 3 months",
        legal_domain="property",
        requested_actions=["research"],
        research_results=[
            {"content": "UP Urban Buildings Act - security deposit must be returned within 30 days"},
            {"content": "Section 406 IPC - Criminal breach of trust"}
        ],
        kanoon_cases=[
            {"title": "Ram vs Shyam 2020", "url": "https://indiankanoon.org/doc/123"}
        ],
        final_response="Based on UP Urban Buildings Act, landlord must return deposit within 30 days. Section 406 IPC applicable. Verify from official sources before citing in court.",
        next_steps=["Send legal notice", "File in Rent Control Court"],
    )

    result = verification_agent(test_state)
    print("\n" + "=" * 60)
    print("VERIFICATION AGENT TEST")
    print("=" * 60)
    print(f"Confidence score     : {result['confidence_score']}")
    print(f"Human approval needed: {result['human_approval_needed']}")
    print(f"Hallucination        : {result['hallucination_detected']}")
    print(f"\nSources:")
    for s in result["verified_sources"]:
        print(f"  - {s}")
    print(f"\nFinal response preview:")
    print(result["final_response"][:500])