"""
Drafting Agent - NyayMitra 2.0
1. LLM extracts document type + stated facts (never guessed; missing -> placeholder)
2. LLM drafts English + Hindi using the reviewed templates as format reference,
   citing only provisions present in the research context
3. Citation guard flags provisions not found in that context
Falls back to deterministic template filling if the LLM fails.
"""
from backend.state import NyayMitraState, initial_state
from backend.config import OPENAI_API_KEY
from backend.agents.schemas import DraftFields, DraftingOutput, DraftLLMOutput, to_state
from backend.utils.citations import cited_numbers, find_unverified_citations, research_context  # noqa: F401
from langsmith import traceable
from openai import OpenAI
from datetime import datetime
import json


DOCUMENT_TEMPLATES = {
    "legal_notice": {
        "hindi": """कानूनी नोटिस

दिनांक: {date}

प्रति,
{recipient_name}
{recipient_address}

विषय: {subject}

महोदय/महोदया,

मैं {sender_name}, निवासी {sender_address}, आपको इस नोटिस के माध्यम से सूचित करता/करती हूं कि:

{facts}

अतः आपसे अनुरोध है कि इस नोटिस की प्राप्ति के {days} दिनों के भीतर {relief} करें।

यदि आप ऐसा करने में विफल रहते हैं, तो मुझे कानूनी कार्रवाई करने के लिए बाध्य होना पड़ेगा।

भवदीय,
{sender_name}
{sender_address}
{sender_phone}""",

        "english": """LEGAL NOTICE

Date: {date}

To,
{recipient_name}
{recipient_address}

Subject: {subject}

Sir/Madam,

I, {sender_name}, resident of {sender_address}, hereby serve you this legal notice:

{facts}

You are hereby called upon to {relief} within {days} days of receipt of this notice.

Failing which, I shall be constrained to initiate appropriate legal proceedings against you without further notice.

Yours faithfully,
{sender_name}
{sender_address}
{sender_phone}"""
    },

    "rti_application": {
        "hindi": """सूचना का अधिकार अधिनियम, 2005 के अंतर्गत सूचना हेतु आवेदन

सेवा में,
जन सूचना अधिकारी
{department_name}
{department_address}

दिनांक: {date}

विषय: RTI अधिनियम 2005 की धारा 6(1) के अंतर्गत सूचना हेतु आवेदन

महोदय/महोदया,

मैं {applicant_name}, निवासी {applicant_address}, आपसे निम्नलिखित सूचना प्राप्त करना चाहता/चाहती हूं:

{information_requested}

आवेदन शुल्क: ₹10/- (BPL श्रेणी से छूट)

आवेदक का नाम: {applicant_name}
पता: {applicant_address}
दूरभाष: {applicant_phone}
हस्ताक्षर: ________________
दिनांक: {date}""",

        "english": """APPLICATION UNDER RIGHT TO INFORMATION ACT, 2005

To,
The Public Information Officer
{department_name}
{department_address}

Date: {date}

Subject: Application for Information under Section 6(1) of RTI Act 2005

Sir/Madam,

I, {applicant_name}, resident of {applicant_address}, wish to obtain the following information:

{information_requested}

Application Fee: Rs. 10/- (Exempted for BPL category)

Name: {applicant_name}
Address: {applicant_address}
Phone: {applicant_phone}
Signature: ________________
Date: {date}"""
    },

    "consumer_complaint": {
        "hindi": """उपभोक्ता फोरम में शिकायत

{forum_name}
{forum_address}

दिनांक: {date}

शिकायतकर्ता: {complainant_name}
पता: {complainant_address}

विरुद्ध

प्रतिपक्षी: {opposite_party}
पता: {opposite_address}

शिकायत का विवरण:
{complaint_details}

राहत की मांग:
{relief_sought}

शपथपत्र:
मैं {complainant_name} शपथपूर्वक कहता/कहती हूं कि उपरोक्त तथ्य सत्य हैं।

हस्ताक्षर: ________________
दिनांक: {date}""",

        "english": """COMPLAINT BEFORE CONSUMER FORUM

{forum_name}
{forum_address}

Date: {date}

Complainant: {complainant_name}
Address: {complainant_address}

Versus

Opposite Party: {opposite_party}
Address: {opposite_address}

Facts of Complaint:
{complaint_details}

Relief Sought:
{relief_sought}

Affidavit:
I, {complainant_name}, solemnly affirm that the above facts are true.

Signature: ________________
Date: {date}"""
    },

    "police_complaint": {
        "hindi": """थाना प्रभारी को शिकायत पत्र

सेवा में,
थाना प्रभारी
{police_station}

दिनांक: {date}

विषय: {subject} के संबंध में शिकायत

महोदय,

मैं {complainant_name}, निवासी {complainant_address}, आपको सूचित करना चाहता/चाहती हूं कि:

घटना की तारीख: {incident_date}
घटना का स्थान: {incident_location}

घटना का विवरण:
{incident_details}

आरोपी का नाम (यदि ज्ञात): {accused_name}

अतः आपसे प्रार्थना है कि उचित कार्रवाई की जाए।

प्रार्थी,
{complainant_name}
{complainant_address}
{complainant_phone}""",

        "english": """COMPLAINT TO POLICE STATION

To,
The Station House Officer
{police_station}

Date: {date}

Subject: Complaint regarding {subject}

Sir,

I, {complainant_name}, resident of {complainant_address}, wish to report that:

Date of Incident: {incident_date}
Place of Incident: {incident_location}

Details of Incident:
{incident_details}

Name of Accused (if known): {accused_name}

I request you to take appropriate action in the matter.

Complainant,
{complainant_name}
{complainant_address}
{complainant_phone}"""
    },

    "vakalatnama": {
        "hindi": """वकालतनामा

न्यायालय: {court_name}
दिनांक: {date}

मैं {client_name}, निवासी {client_address}, एतद्द्वारा {lawyer_name} 
(नामांकन संख्या: {lawyer_enrollment}) को अपना अधिवक्ता नियुक्त करता/करती हूं।

मामला: {case_details}

मैं उपरोक्त अधिवक्ता को मेरी ओर से सभी कानूनी कार्रवाई करने का अधिकार देता/देती हूं।

मुवक्किल हस्ताक्षर: ________________
दिनांक: {date}""",

        "english": """VAKALATNAMA

Court: {court_name}
Date: {date}

I, {client_name}, resident of {client_address}, hereby appoint {lawyer_name}
(Enrollment No: {lawyer_enrollment}) as my Advocate.

Matter: {case_details}

I authorize the above advocate to act on my behalf in all legal proceedings.

Client Signature: ________________
Date: {date}"""
    }
}


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


def _allowed_context(state: NyayMitraState, doc_type: str) -> str:
    parts = [research_context(state)]
    tmpl = DOCUMENT_TEMPLATES.get(doc_type)
    if tmpl:  # curated templates may cite their own enabling provision (e.g. RTI s.6(1))
        parts += [tmpl["english"], tmpl["hindi"]]
    return "\n".join(p for p in parts if p)


def _llm_json(system: str, user: str, max_tokens: int) -> dict:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        max_tokens=max_tokens,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    return json.loads(response.choices[0].message.content)


def extract_fields(state: NyayMitraState) -> DraftFields:
    history = state.get("conversation_history", [])[-4:]
    convo = "\n".join(f"{m['role'].upper()}: {m['content'][:300]}" for m in history)
    user = f"Query: {state.get('user_query', '')}\n"
    if convo:
        user += f"\nRecent conversation:\n{convo}\n"
    return DraftFields.model_validate(_llm_json(EXTRACTION_PROMPT, user, 700))


def missing_fields(fields: DraftFields) -> list:
    return [f for f in REQUIRED_FIELDS.get(fields.document_type, []) if getattr(fields, f) is None]


def template_fallback(fields: DraftFields) -> tuple:
    """Deterministic fill of the reviewed template with whatever was extracted."""
    values = dict(PLACEHOLDERS, date=_today())
    for key, attr in FIELD_MAP.items():
        val = getattr(fields, attr)
        if val is not None:
            values[key] = val
    tmpl = DOCUMENT_TEMPLATES.get(fields.document_type)
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
    try:
        fields = extract_fields(state)
    except Exception as e:
        print(f"[Drafting] Extraction failed: {e}")
        fields = DraftFields()

    doc_type = fields.document_type
    allowed = _allowed_context(state, doc_type)
    tmpl = DOCUMENT_TEMPLATES.get(doc_type)

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
        draft = DraftLLMOutput.model_validate(_llm_json(DRAFTING_PROMPT, user, 3500))
        doc_english, doc_hindi = draft.english, draft.hindi
    except Exception as e:
        print(f"[Drafting] LLM drafting failed, using template: {e}")
        method = "template_fallback"
        doc_english, doc_hindi = template_fallback(fields)

    unverified = find_unverified_citations([doc_english, doc_hindi], allowed)
    missing = missing_fields(fields)
    draft_doc = doc_hindi if language == "hi" else doc_english

    print(f"[Drafting] type={doc_type} method={method} missing={missing} unverified={unverified}")

    return to_state(DraftingOutput(
        draft_document=draft_doc,
        document_type=doc_type,
        document_hindi=doc_hindi,
        document_english=doc_english,
        draft_missing_fields=missing,
        draft_unverified_citations=unverified,
        draft_method=method,
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