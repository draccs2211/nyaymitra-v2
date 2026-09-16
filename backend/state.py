from typing import TypedDict, List, Optional, Dict, Any


class NyayMitraState(TypedDict):
    # Input
    user_query: str
    language: str                    # hi, en, hinglish
    session_id: str
    user_id: str

    # Conversation
    conversation_history: List[Dict[str, str]]

    # Routing
    legal_domain: str                # criminal, civil, family, etc.
    needs_document: bool
    needs_research: bool
    needs_referral: bool

    # Emergency
    is_emergency: bool
    emergency_type: str

    # Research results
    research_results: List[Dict[str, Any]]
    kanoon_cases: List[Dict[str, Any]]
    ipc_sections: List[Dict[str, Any]]

    # Analysis
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
