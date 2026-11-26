"""
Pydantic output schemas for every NyayMitra agent.

Each agent builds one of the *Output models and returns `to_state(model)`,
a partial dict merged into NyayMitraState by LangGraph. Fields left as None
are dropped, so an agent only writes the state keys it actually sets.
"""
import re
from typing import Any, Dict, List, Literal, Optional, get_args

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.config import LEGAL_DOMAINS

# ─── Shared types ─────────────────────────────────────────────────────────────

Language = Literal["en", "hi", "hinglish"]
Urgency = Literal["critical", "high", "medium", "low"]
# Actions the user asked for; executed in fixed pipeline order, not list order
RequestedAction = Literal["research", "drafting", "referral"]
EmergencyType = Literal[
    "physical_threat", "domestic_violence", "sexual_assault", "custody_emergency",
    "illegal_detention", "anticipatory_bail_urgent", "property_emergency",
    "court_emergency", "other",
]
ApprovalStatus = Literal["not_required", "pending", "approved", "rejected"]
DocumentType = Literal[
    "legal_notice", "rti_application", "consumer_complaint",
    "police_complaint", "vakalatnama", "other",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


def to_state(model: BaseModel) -> Dict[str, Any]:
    """Serialize an agent output to a partial state update (None fields omitted)."""
    return model.model_dump(exclude_none=True)


def _coerce_choice(value: Any, choices: tuple, default: str) -> str:
    """Normalize an LLM-provided enum value; unknown values fall back to default."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in choices:
            return v
    return default


# ─── Guardian ─────────────────────────────────────────────────────────────────

class GuardianLLMOutput(BaseModel):
    """Raw urgency classification from the guardian LLM."""
    model_config = ConfigDict(extra="ignore")

    is_emergency: bool = False
    emergency_type: str = ""
    reasoning: str = ""

    @model_validator(mode="after")
    def _consistent(self):
        if not self.is_emergency:
            self.emergency_type = ""
        else:
            self.emergency_type = _coerce_choice(self.emergency_type, get_args(EmergencyType), "other")
        return self


class GuardianOutput(_Base):
    tokens_used: Optional[int] = None
    is_emergency: bool
    emergency_type: str = ""
    urgency_level: Literal["", "CRITICAL", "HIGH", "MEDIUM"] = ""
    urgent_alert: str = ""
    guardian_status: Literal["checked", "unavailable"]


# ─── Orchestrator ─────────────────────────────────────────────────────────────

class ClassificationLLMOutput(BaseModel):
    """Validates the raw JSON returned by the orchestrator LLM (lenient per field)."""
    model_config = ConfigDict(extra="ignore")

    legal_domain: str = "general"
    requested_actions: List[RequestedAction] = Field(default_factory=lambda: ["research"])
    language: Language = "en"
    search_query_en: str = ""
    jurisdiction: str = "IN"      # IN-XX if the query names an Indian state/UT (or a city in it)
    district: str = ""            # official district name if the query names a district/city
    urgency: Urgency = "low"
    key_issues: List[str] = Field(default_factory=list)
    suggested_acts: List[str] = Field(default_factory=list)
    reasoning: str = ""

    @field_validator("legal_domain", mode="before")
    @classmethod
    def _domain(cls, v):
        return _coerce_choice(v, tuple(LEGAL_DOMAINS), "general")

    @field_validator("district", mode="before")
    @classmethod
    def _district(cls, v):
        v = " ".join(str(v or "").split())[:80]
        return "" if v.lower() in ("", "none", "null", "unknown", "n/a") else v

    @field_validator("jurisdiction", mode="before")
    @classmethod
    def _jurisdiction(cls, v):
        v = str(v or "").strip().upper()
        return v if re.match(r"^IN(-[A-Z]{2})?$", v) else "IN"

    @field_validator("search_query_en", mode="before")
    @classmethod
    def _search_query(cls, v):
        return " ".join(str(v).split())[:300] if v else ""

    @field_validator("language", mode="before")
    @classmethod
    def _language(cls, v):
        return _coerce_choice(v, get_args(Language), "en")

    @field_validator("urgency", mode="before")
    @classmethod
    def _urgency(cls, v):
        return _coerce_choice(v, get_args(Urgency), "low")

    @field_validator("requested_actions", mode="before")
    @classmethod
    def _actions(cls, v):
        """Keep known actions, de-duplicated in order; never empty."""
        if isinstance(v, str):
            v = [v]
        allowed = get_args(RequestedAction)
        out = []
        for a in v or []:
            a = a.strip().lower() if isinstance(a, str) else a
            if a in allowed and a not in out:
                out.append(a)
        return out or ["research"]

    @field_validator("key_issues", "suggested_acts", mode="before")
    @classmethod
    def _str_list(cls, v):
        if v is None:
            return []
        if isinstance(v, str):
            return [v]
        return [str(x) for x in v if x]


class ClassificationMeta(_Base):
    urgency: Urgency
    key_issues: List[str] = Field(default_factory=list)
    suggested_acts: List[str] = Field(default_factory=list)
    reasoning: str = ""


class OrchestratorOutput(_Base):
    tokens_used: Optional[int] = None
    legal_domain: str
    requested_actions: List[RequestedAction] = Field(min_length=1)
    language: Language
    search_query_en: str = ""
    jurisdiction: Optional[str] = None
    district: Optional[str] = None
    classification: ClassificationMeta


# ─── Research ─────────────────────────────────────────────────────────────────

class RetrievedDoc(BaseModel):
    model_config = ConfigDict(extra="allow")  # retriever may add scores later

    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None
    source: Optional[str] = None            # dense / sparse
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None


class KanoonCase(_Base):
    title: str = ""
    citation: str = ""
    court: str = ""
    date: str = ""
    snippet: str = ""
    url: str = ""
    tid: str = ""
    source: str = "Indian Kanoon (Discovery only — verify before citing)"

    @field_validator("*", mode="before")
    @classmethod
    def _stringify(cls, v):
        return "" if v is None else str(v)


SourceStatus = Literal["ok", "not_configured", "unavailable", "error", "skipped"]


class ResearchOutput(_Base):
    research_results: List[RetrievedDoc] = Field(default_factory=list)
    kanoon_cases: List[KanoonCase] = Field(default_factory=list)
    ipc_sections: List[RetrievedDoc] = Field(default_factory=list)
    source_status: Dict[str, SourceStatus] = Field(default_factory=dict)


# ─── Analysis ─────────────────────────────────────────────────────────────────

def _to_str_list(v) -> List[str]:
    if v is None:
        return []
    if isinstance(v, str):
        v = [v]
    return [str(x).strip() for x in v if x is not None and str(x).strip()]


class AnalysisLLMOutput(BaseModel):
    """Structured analysis from the LLM. No verdict fields by design (CLAUDE.md)."""
    model_config = ConfigDict(extra="ignore")

    summary: str = Field(min_length=1)
    applicable_law: List[str] = Field(default_factory=list)
    action_plan: List[str] = Field(default_factory=list)
    evidence_needed: List[str] = Field(default_factory=list)
    missing_facts: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    questions_for_lawyer: List[str] = Field(default_factory=list)
    timeline: str = ""
    costs: str = ""

    @field_validator("applicable_law", "action_plan", "evidence_needed", "missing_facts",
                     "uncertainties", "questions_for_lawyer", mode="before")
    @classmethod
    def _lists(cls, v):
        return _to_str_list(v)

    @field_validator("summary", "timeline", "costs", mode="before")
    @classmethod
    def _strings(cls, v):
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        return "" if v is None else str(v).strip()


class AnalysisDetail(_Base):
    analysis: str
    domain: Optional[str] = None
    cases_found: Optional[int] = None
    rag_results: Optional[int] = None
    error: Optional[str] = None
    summary: Optional[str] = None
    applicable_law: List[str] = Field(default_factory=list)
    action_plan: List[str] = Field(default_factory=list)
    evidence_needed: List[str] = Field(default_factory=list)
    missing_facts: List[str] = Field(default_factory=list)
    uncertainties: List[str] = Field(default_factory=list)
    questions_for_lawyer: List[str] = Field(default_factory=list)
    timeline: Optional[str] = None
    costs: Optional[str] = None


class AnalysisOutput(_Base):
    tokens_used: Optional[int] = None
    case_analysis: AnalysisDetail
    next_steps: List[str] = Field(default_factory=list)
    final_response: str
    analysis_unverified_citations: List[str] = Field(default_factory=list)


# ─── Drafting ─────────────────────────────────────────────────────────────────

_EMPTY_MARKERS = {"", "null", "none", "n/a", "na", "unknown", "not provided", "not mentioned"}


class DraftFields(BaseModel):
    """Facts extracted from the query by the LLM. None = not stated (never guessed)."""
    model_config = ConfigDict(extra="ignore")

    document_type: DocumentType = "legal_notice"
    document_title: str = ""
    client_name: Optional[str] = None
    client_address: Optional[str] = None
    client_phone: Optional[str] = None
    opposite_party_name: Optional[str] = None
    opposite_party_address: Optional[str] = None
    authority_name: Optional[str] = None       # police station / PIO dept / forum / court
    authority_address: Optional[str] = None
    subject: Optional[str] = None
    facts: Optional[str] = None
    relief_sought: Optional[str] = None
    amount: Optional[str] = None
    incident_date: Optional[str] = None
    incident_location: Optional[str] = None

    @field_validator("document_type", mode="before")
    @classmethod
    def _doc_type(cls, v):
        return _coerce_choice(v, get_args(DocumentType), "other")

    @field_validator("*", mode="before")
    @classmethod
    def _blank_to_none(cls, v, info):
        if info.field_name in ("document_type", "document_title"):
            return v if v is not None else ""
        if v is None:
            return None
        v = str(v).strip()
        return None if v.lower() in _EMPTY_MARKERS else v


class DraftLLMOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")
    english: str = Field(min_length=20)
    hindi: str = Field(min_length=20)


class DraftingOutput(_Base):
    tokens_used: Optional[int] = None
    draft_document: str
    document_type: DocumentType
    document_hindi: str
    document_english: str
    draft_missing_fields: List[str] = Field(default_factory=list)
    draft_unverified_citations: List[str] = Field(default_factory=list)
    draft_method: Literal["llm", "template_fallback"]


# ─── Referral ─────────────────────────────────────────────────────────────────

class DlsaOffice(_Base):
    name: str
    district: str = ""
    state: str = ""
    secretary: str = ""
    phone: str = ""
    as_of: str = ""                 # date the contact was collected
    source_url: str = ""
    verified_by_human: bool = False
    helpline: str = "15100"


class ReferralOutput(_Base):
    dlsa_offices: List[DlsaOffice] = Field(default_factory=list)
    lawyer_referrals: List[Dict[str, Any]] = Field(default_factory=list)
    final_response: str


# ─── Verification ─────────────────────────────────────────────────────────────

class VerificationOutput(_Base):
    confidence_score: float = Field(ge=0.0, le=1.0)
    hallucination_detected: bool
    human_approval_needed: bool
    approval_status: Literal["not_required", "pending"]
    verified_sources: List[str] = Field(default_factory=list)
    final_response: str


# ─── Human approval ───────────────────────────────────────────────────────────

class ApprovalDecision(_Base):
    """Reviewer input used to resume an interrupted run."""
    decision: Literal["approved", "rejected"]
    reviewer_id: str = Field(min_length=1)
    notes: str = ""
    edited_response: Optional[str] = None   # approved only: reviewer's corrected text

    @model_validator(mode="after")
    def _edit_only_on_approve(self):
        if self.edited_response is not None and self.decision != "approved":
            raise ValueError("edited_response is only allowed when approving")
        if self.decision == "rejected" and not self.notes.strip():
            raise ValueError("notes are required when rejecting")
        return self


class ApprovalRecord(_Base):
    reviewer_id: str
    notes: str = ""
    decided_at: str              # ISO-8601 UTC
    edited: bool = False


class HumanApprovalOutput(_Base):
    approval_status: Literal["approved", "rejected"]
    approval: ApprovalRecord
    final_response: Optional[str] = None    # set only when reviewer edited
