"""
Drafting Agent - NyayMitra 2.0
1. LLM extracts document type + stated facts (never guessed; missing -> placeholder)
2. LLM drafts English + Hindi using the reviewed template (templates MCP server,
   data/templates/document_templates.json) as format reference,
   citing only provisions present in the research context
3. Citation guard flags provisions not found in that context
Falls back to deterministic template filling if the LLM fails.
"""
from backend.state import NyayMitraState, initial_state
from backend.llm import Usage, chat_json
from backend.agents.schemas import DraftFields, DraftingOutput, DraftLLMOutput, to_state
from backend.mcp_client import MCPCallError, call_tool, run_sync
from backend.templates import get_template as load_template
from backend.utils.citations import cited_numbers, find_unverified_citations, research_context  # noqa: F401
from langsmith import traceable
import logging

logger = logging.getLogger("nyaymitra.drafting")
from openai import OpenAI
from datetime import datetime
import json


def fetch_template(doc_type: str):
    """Reviewed template {english, hindi} via the templates MCP server; if the server
    call fails, read the same file directly. None if there is no template for doc_type."""
    try:
        data = run_sync(call_tool("templates", "get_template", {"doc_type": doc_type, "language": "both"}))
    except MCPCallError as e:
        logger.warning(f"templates MCP unavailable, reading file directly: {e}")
        return load_template(doc_type)
    if data.get("status") != "success":
        return None
    return {"english": data["template_english"], "hindi": data["template_hindi"]}


PLACEHOLDERS = {
    "sender_name": "[Client Name]",
    "sender_address": "[Client Address]",
    "sender_phone": "[Client Phone]",
    "recipient_name": "[Opposite Party Name]",
    "recipient_address": "[Opposite Party Address]",
    "subject": "[Subject]",
    "facts": "[Describe the facts of the case in detail]",
    "relief": "[State what action you want them to take]",
    "days": "15",
    "applicant_name": "[Applicant Name]",
    "applicant_address": "[Applicant Address]",
    "applicant_phone": "[Applicant Phone]",
    "department_name": "[Department Name]",
    "department_address": "[Department Address]",
    "information_requested": "[Describe the specific information you need]",
    "complainant_name": "[Complainant Name]",
    "complainant_address": "[Complainant Address]",
    "complainant_phone": "[Complainant Phone]",
    "opposite_party": "[Opposite Party Name]",
    "opposite_address": "[Opposite Party Address]",
    "complaint_details": "[Describe your complaint in detail]",
    "relief_sought": "[State what relief you are seeking]",
    "forum_name": "[Consumer Forum Name]",
    "forum_address": "[Forum Address]",
    "police_station": "[Police Station Name]",
    "incident_date": "[Date of Incident]",
    "incident_location": "[Location of Incident]",
    "incident_details": "[Describe the incident in detail]",
    "accused_name": "[Name of Accused if known]",
    "court_name": "[Court Name]",
    "client_name": "[Client Name]",
    "client_address": "[Client Address]",
    "lawyer_name": "[Advocate Name]",
    "lawyer_enrollment": "[Enrollment Number]",
    "case_details": "[Case Title and Number]",
}

# template placeholder -> DraftFields attribute (fallback filling)
FIELD_MAP = {
    "sender_name": "client_name", "sender_address": "client_address", "sender_phone": "client_phone",
    "applicant_name": "client_name", "applicant_address": "client_address", "applicant_phone": "client_phone",
    "complainant_name": "client_name", "complainant_address": "client_address", "complainant_phone": "client_phone",
    "client_name": "client_name", "client_address": "client_address",
    "recipient_name": "opposite_party_name", "recipient_address": "opposite_party_address",
    "opposite_party": "opposite_party_name", "opposite_address": "opposite_party_address",
    "accused_name": "opposite_party_name",
    "department_name": "authority_name", "department_address": "authority_address",
    "forum_name": "authority_name", "forum_address": "authority_address",
    "police_station": "authority_name", "court_name": "authority_name",
    "subject": "subject", "case_details": "subject",
    "facts": "facts", "information_requested": "facts", "complaint_details": "facts",
    "incident_details": "facts",
    "relief": "relief_sought", "relief_sought": "relief_sought",
    "incident_date": "incident_date", "incident_location": "incident_location",
}

# Facts a lawyer must supply before the document can be sent
REQUIRED_FIELDS = {
    "legal_notice": ["client_name", "client_address", "opposite_party_name",
                     "opposite_party_address", "facts", "relief_sought"],
    "rti_application": ["client_name", "client_address", "authority_name", "facts"],
    "consumer_complaint": ["client_name", "client_address", "opposite_party_name",
                           "opposite_party_address", "authority_name", "facts", "relief_sought"],
    "police_complaint": ["client_name", "client_address", "authority_name",
                         "incident_date", "incident_location", "facts"],
    "vakalatnama": ["client_name", "client_address", "authority_name", "subject"],
    "other": ["client_name", "facts"],
}

EXTRACTION_PROMPT = """You extract facts for drafting an Indian legal document.

Return JSON with exactly these keys:
{
  "document_type": "<one of: legal_notice, rti_application, consumer_complaint, police_complaint, vakalatnama, other>",
  "document_title": "<short title if document_type is other, else empty>",
  "client_name": null, "client_address": null, "client_phone": null,
  "opposite_party_name": null, "opposite_party_address": null,
  "authority_name": "<police station / public authority / consumer forum / court, if stated>",
  "authority_address": null,
  "subject": null, "facts": "<facts stated by the user, in neutral third person>",
  "relief_sought": null, "amount": null, "incident_date": null, "incident_location": null
}

Rules:
- Use ONLY information explicitly present in the query or conversation. If a value is not stated, use null.
- NEVER invent names, addresses, phone numbers, dates, amounts or authorities.
- Pick document_type from what the user asks to draft; use "other" (with document_title) if none fits.
Return ONLY JSON."""

DRAFTING_PROMPT = """You are drafting an Indian legal document for review by a lawyer.

Return JSON: {"english": "<full document in formal English>", "hindi": "<the same document in formal Hindi (Devanagari)>"}

Rules:
- Follow the structure of the FORMAT REFERENCE when one is given.
- Use only the facts in EXTRACTED FIELDS. Where a field is null, keep a bracketed placeholder such as [Client Name] - never invent it.
- Keep EVERY line of the format reference (addresses, phone, signature, etc.). Never drop a line because its value is unknown - write the placeholder, e.g. [Opposite Party Address], [Client Phone].
- Cite statutory provisions ONLY if they appear in ALLOWED LEGAL CONTEXT. If none fit, cite none. Never cite case law.
- Do not state conclusions about case strength or likely outcome.
- Use today's date given below.
Return ONLY JSON."""

def _today() -> str:
    return datetime.now().strftime("%d/%m/%Y")


def fill_template(template: str, values: dict) -> str:
    result = template
    for key, value in values.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


def _allowed_context(state: NyayMitraState, tmpl) -> str:
    parts = [research_context(state)]
    if tmpl:  # curated templates may cite their own enabling provision (e.g. RTI s.6(1))
        parts += [tmpl["english"], tmpl["hindi"]]
    return "\n".join(p for p in parts if p)


def _llm_json(system: str, user: str, max_tokens: int):
    """Returns (dict, Usage)."""
    return chat_json(OpenAI, system, user, max_tokens=max_tokens, temperature=0.2)


def extract_fields(state: NyayMitraState) -> DraftFields:
    history = state.get("conversation_history", [])[-4:]
    convo = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in history)
    user = f"Query: {state.get('user_query', '')}\n"
    if convo:
        user += f"\nRecent conversation:\n{convo}\n"
    data, usage = _llm_json(EXTRACTION_PROMPT, user, 700)
    return DraftFields.model_validate(data), usage


def missing_fields(fields: DraftFields) -> list:
    return [f for f in REQUIRED_FIELDS.get(fields.document_type, []) if getattr(fields, f) is None]


def template_fallback(fields: DraftFields, tmpl) -> tuple:
    """Deterministic fill of the reviewed template with whatever was extracted."""
    values = dict(PLACEHOLDERS, date=_today())
    for key, attr in FIELD_MAP.items():
        val = getattr(fields, attr)
        if val is not None:
            values[key] = val
    if tmpl is None:  # "other": no reviewed template - minimal skeleton
        title = fields.document_title or "[Document Title]"
        body = (f"{title.upper()}\n\nDate: {values['date']}\n\n"
                f"{values['facts']}\n\nRelief sought: {values['relief_sought']}\n\n"
                f"{values['client_name']}\n{values['client_address']}")
        return body, body
    return fill_template(tmpl["english"], values), fill_template(tmpl["hindi"], values)


@traceable(name="Drafting Agent")
def drafting_agent(state: NyayMitraState) -> dict:
    language = state.get("language", "en")

    # 1. Field extraction (LLM). On failure we still draft a placeholder template.
    usage = Usage()
    try:
        fields, u = extract_fields(state)
        usage += u
    except Exception as e:
        logger.warning(f"Extraction failed: {e}")
        fields = DraftFields()

    doc_type = fields.document_type
    tmpl = fetch_template(doc_type)
    allowed = _allowed_context(state, tmpl)

    # 2. Drafting (LLM)
    method = "llm"
    try:
        analysis = (state.get("case_analysis") or {}).get("analysis", "")[:1500]
        title = f" ({fields.document_title})" if fields.document_title else ""
        user = (
            f"Today's date: {_today()}\n"
            f"Document type: {doc_type}{title}\n\n"
            f"EXTRACTED FIELDS:\n{fields.model_dump_json(indent=1)}\n\n"
            f"FORMAT REFERENCE:\n{tmpl['english'] if tmpl else '(none - use standard Indian format)'}\n\n"
            f"ALLOWED LEGAL CONTEXT:\n{allowed[:4000] or '(none)'}\n\n"
            f"LEGAL ANALYSIS (for context, do not copy conclusions):\n{analysis or '(none)'}"
        )
        data, u = _llm_json(DRAFTING_PROMPT, user, 3500)
        usage += u
        draft = DraftLLMOutput.model_validate(data)
        doc_english, doc_hindi = draft.english, draft.hindi
    except Exception as e:
        logger.warning(f"LLM drafting failed, using template: {e}")
        method = "template_fallback"
        doc_english, doc_hindi = template_fallback(fields, tmpl)

    unverified = find_unverified_citations([doc_english, doc_hindi], allowed)
    missing = missing_fields(fields)
    draft_doc = doc_hindi if language == "hi" else doc_english

    logger.info(f"Draft: type={doc_type} method={method} missing={missing} unverified={unverified}")

    return to_state(DraftingOutput(
        draft_document=draft_doc,
        document_type=doc_type,
        document_hindi=doc_hindi,
        document_english=doc_english,
        draft_missing_fields=missing,
        draft_unverified_citations=unverified,
        draft_method=method,
        tokens_used=usage.total,
    ))


# ── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_cases = [
        {
            "query": "Draft legal notice to landlord for not returning security deposit of Rs 50000",
            "domain": "property",
            "language": "en"
        },
        {
            "query": "Need RTI application for PM Awas Yojana status",
            "domain": "rti",
            "language": "en"
        },
        {
            "query": "Consumer forum complaint for defective washing machine",
            "domain": "consumer",
            "language": "en"
        },
        {
            "query": "Police complaint against neighbor for threatening",
            "domain": "criminal",
            "language": "en"
        },
        {
            "query": "मकान मालिक को कानूनी नोटिस भेजना है",
            "domain": "property",
            "language": "hi"
        },
    ]

    base_state = initial_state()

    print("=" * 60)
    print("DRAFTING AGENT TEST")
    print("=" * 60)

    for tc in test_cases:
        base_state["user_query"] = tc["query"]
        base_state["legal_domain"] = tc["domain"]
        base_state["language"] = tc["language"]

        result = drafting_agent(base_state)

        print(f"\nQuery  : {tc['query'][:50]}")
        print(f"Doc type: {result['document_type']}")
        print(f"\nDocument preview:")
        doc = result["document_hindi"] if tc["language"] == "hi" else result["document_english"]
        print(doc[:400])
        print("=" * 60)