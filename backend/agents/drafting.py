"""
Drafting Agent - NyayMitra 2.0
Generates legal documents using templates.
Upgrade planned: OpenAI-based intelligent drafting after testing.
"""
from backend.state import NyayMitraState, initial_state
from backend.agents.schemas import DraftingOutput, to_state
from langsmith import traceable
from datetime import datetime


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


DEFAULT_VALUES = {
    "date": datetime.now().strftime("%d/%m/%Y"),
    "sender_name": "[Client Name]",
    "sender_address": "[Client Address]",
    "sender_phone": "[Client Phone]",
    "recipient_name": "[Opposite Party Name]",
    "recipient_address": "[Opposite Party Address]",
    "subject": "[Subject of Notice]",
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


def identify_document_type(query: str, domain: str) -> str:
    query_lower = query.lower()

    if any(kw in query_lower for kw in [
        "rti", "right to information", "suchna adhikar",
        "government information", "public information"
    ]):
        return "rti_application"

    if any(kw in query_lower for kw in [
        "consumer", "product defect", "refund", "service issue",
        "consumer forum", "consumer court"
    ]):
        return "consumer_complaint"

    if any(kw in query_lower for kw in [
        "fir", "police complaint", "police station", "thana",
        "first information report"
    ]):
        return "police_complaint"

    if any(kw in query_lower for kw in [
        "vakalatnama", "appoint lawyer", "authorize advocate",
        "power of attorney"
    ]):
        return "vakalatnama"

    return "legal_notice"


def fill_template(template: str, values: dict) -> str:
    result = template
    for key, value in values.items():
        result = result.replace(f"{{{key}}}", str(value))
    return result


@traceable(name="Drafting Agent")
def drafting_agent(state: NyayMitraState) -> dict:
    query = state.get("user_query", "")
    domain = state.get("legal_domain", "general")
    language = state.get("language", "en")

    # Identify document type
    doc_type = identify_document_type(query, domain)

    # Get template
    template = DOCUMENT_TEMPLATES.get(doc_type, DOCUMENT_TEMPLATES["legal_notice"])

    # Fill with default values
    values = DEFAULT_VALUES.copy()

    # Fill documents
    try:
        doc_hindi = fill_template(template["hindi"], values)
    except Exception:
        doc_hindi = template["hindi"]

    try:
        doc_english = fill_template(template["english"], values)
    except Exception:
        doc_english = template["english"]

    draft = doc_hindi if language == "hi" else doc_english

    print(f"[Drafting] Document type: {doc_type} | Language: {language}")

    return to_state(DraftingOutput(
        draft_document=draft,
        document_type=doc_type,
        document_hindi=doc_hindi,
        document_english=doc_english,
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