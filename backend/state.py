from typing import TypedDict, List, Optional, Dict, Any


class NyayMitraState(TypedDict):
    # Input
    user_query: str
    language: str                    # hi, en, hinglish
    session_id: str
    user_id: str

    # Conversation
    conversation_history: List[Dict[str, str]]

    # Routing (set by orchestrator)
    legal_domain: str                # criminal, civil, family, etc.
    requested_actions: List[str]     # subset of research / drafting / referral
    classification: Dict[str, Any]   # urgency, key_issues, suggested_acts, reasoning

    # Emergency
    is_emergency: bool
    emergency_type: str

    # Research results
    research_results: List[Dict[str, Any]]
    kanoon_cases: List[Dict[str, Any]]
    ipc_sections: List[Dict[str, Any]]

    # Analysis (set by analysis agent)
    case_analysis: Dict[str, Any]
    case_strength: str               # weak, medium, strong

    # Document
    draft_document: str
    document_type: str
    document_hindi: str
    document_english: str

    # Verification
    confidence_score: float
    hallucination_detected: bool
    human_approval_needed: bool
    verified_sources: List[str]

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
        "conversation_history": [],
        "legal_domain": "general",
        "requested_actions": [],
        "classification": {},
        "is_emergency": False,
        "emergency_type": "",
        "research_results": [],
        "kanoon_cases": [],
        "ipc_sections": [],
        "case_analysis": {},
        "case_strength": "medium",
        "draft_document": "",
        "document_type": "",
        "document_hindi": "",
        "document_english": "",
        "confidence_score": 0.0,
        "hallucination_detected": False,
        "human_approval_needed": False,
        "verified_sources": [],
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
