from typing import TypedDict, List, Optional, Dict, Any


class NyayMitraState(TypedDict):
    # Input
    user_query: str
    language: str                    # hi, en, hinglish
    session_id: str
    user_id: str
    jurisdiction: str                # IN or IN-<state>; recorded, not yet used for retrieval

    # Conversation
    conversation_history: List[Dict[str, str]]

    # Routing (set by orchestrator)
    legal_domain: str                # criminal, civil, family, etc.
    requested_actions: List[str]     # subset of research / drafting / referral
    search_query_en: str             # English rewrite of the query for retrieval ("" if none)
    classification: Dict[str, Any]   # urgency, key_issues, suggested_acts, reasoning

    # Emergency
    is_emergency: bool
    emergency_type: str
    urgency_level: str               # "", CRITICAL, HIGH, MEDIUM
    urgent_alert: str                # prepended to final response by verification
    guardian_status: str             # checked / unavailable (LLM failed, failed open)

    # Research results
    research_results: List[Dict[str, Any]]
    kanoon_cases: List[Dict[str, Any]]
    ipc_sections: List[Dict[str, Any]]
    source_status: Dict[str, str]    # per external source: ok / not_configured / unavailable / error / skipped

    # Analysis (set by analysis agent)
    case_analysis: Dict[str, Any]
    analysis_unverified_citations: List[str]  # provisions in analysis not in research context

    # Document
    draft_document: str
    document_type: str
    document_hindi: str
    document_english: str
    draft_missing_fields: List[str]        # facts the lawyer must fill in
    draft_unverified_citations: List[str]  # provisions cited but not in research context
    draft_method: str                      # llm / template_fallback

    # Verification
    confidence_score: float
    hallucination_detected: bool
    human_approval_needed: bool
    verified_sources: List[str]

    # Human approval (LangGraph interrupt)
    approval_status: str             # not_required / pending / approved / rejected
    approval: Dict[str, Any]         # reviewer_id, notes, decided_at, edited

    # Referral
    dlsa_offices: List[Dict[str, Any]]
    lawyer_referrals: List[Dict[str, Any]]

    # Final
    final_response: str
    final_response_hindi: str
    sources: List[str]
    next_steps: List[str]

    # Error handling
    error: Optional[str]
    retry_count: int


def initial_state(**overrides) -> NyayMitraState:
    """Fully-populated default state. Single source of truth for state keys."""
    state: NyayMitraState = {
        "user_query": "",
        "language": "en",
        "session_id": "",
        "user_id": "",
        "jurisdiction": "IN",
        "conversation_history": [],
        "legal_domain": "general",
        "requested_actions": [],
        "search_query_en": "",
        "classification": {},
        "is_emergency": False,
        "emergency_type": "",
        "urgency_level": "",
        "urgent_alert": "",
        "guardian_status": "",
        "research_results": [],
        "kanoon_cases": [],
        "ipc_sections": [],
        "source_status": {},
        "case_analysis": {},
        "analysis_unverified_citations": [],
        "draft_document": "",
        "document_type": "",
        "document_hindi": "",
        "document_english": "",
        "draft_missing_fields": [],
        "draft_unverified_citations": [],
        "draft_method": "",
        "confidence_score": 0.0,
        "hallucination_detected": False,
        "human_approval_needed": False,
        "verified_sources": [],
        "approval_status": "not_required",
        "approval": {},
        "dlsa_offices": [],
        "lawyer_referrals": [],
        "final_response": "",
        "final_response_hindi": "",
        "sources": [],
        "next_steps": [],
        "error": None,
        "retry_count": 0,
    }
    unknown = set(overrides) - set(state)
    if unknown:
        raise KeyError(f"Unknown state keys: {sorted(unknown)}")
    state.update(overrides)
    return state
