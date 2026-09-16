from fastmcp import FastMCP
from backend.database.connection import get_redis
from datetime import datetime
import json
import hashlib

mcp = FastMCP("Document Templates MCP Server")
redis_client = get_redis()

TEMPLATES = {
    "legal_notice": {
        "name": "Legal Notice",
        "description": "General legal notice to any party",
        "fields": ["sender_name", "sender_address", "sender_phone", "recipient_name",
                   "recipient_address", "subject", "facts", "relief", "days"],
        "hindi": """कानूनी नोटिस

दिनांक: {date}

प्रति,
{recipient_name}
{recipient_address}

विषय: {subject}

महोदय/महोदया,

मैं {sender_name}, निवासी {sender_address}, आपको सूचित करता/करती हूं:

{facts}

अतः {days} दिनों के भीतर {relief} करें अन्यथा कानूनी कार्रवाई की जाएगी।

भवदीय,
{sender_name}
{sender_phone}""",
        "english": """LEGAL NOTICE

Date: {date}

To,
{recipient_name}
{recipient_address}

Subject: {subject}

Sir/Madam,

I, {sender_name}, resident of {sender_address}, hereby notify you:

{facts}

You are called upon to {relief} within {days} days failing which legal proceedings shall be initiated.

Yours faithfully,
{sender_name}
{sender_phone}"""
    },
    "rti_application": {
        "name": "RTI Application",
        "description": "Right to Information application",
        "fields": ["applicant_name", "applicant_address", "applicant_phone",
                   "department_name", "department_address", "information_requested"],
        "hindi": """सूचना का अधिकार अधिनियम 2005 - आवेदन

सेवा में,
जन सूचना अधिकारी
{department_name}
{department_address}

दिनांक: {date}

विषय: RTI अधिनियम 2005 की धारा 6(1) के तहत सूचना हेतु

महोदय,

मुझे निम्न सूचना चाहिए:
{information_requested}

शुल्क: ₹10 संलग्न

आवेदक: {applicant_name}
पता: {applicant_address}
फोन: {applicant_phone}
हस्ताक्षर: ________________""",
        "english": """RTI APPLICATION - Right to Information Act 2005

To,
The Public Information Officer
{department_name}
{department_address}

Date: {date}

Subject: Application for Information under Section 6(1) RTI Act 2005

Sir/Madam,

I require the following information:
{information_requested}

Fee: ₹10 enclosed

Applicant: {applicant_name}
Address: {applicant_address}
Phone: {applicant_phone}
Signature: ________________"""
    },
    "consumer_complaint": {
        "name": "Consumer Forum Complaint",
        "description": "Complaint to consumer forum",
        "fields": ["complainant_name", "complainant_address", "opposite_party",
                   "opposite_address", "complaint_details", "relief_sought", "forum_name"],
        "hindi": """उपभोक्ता फोरम शिकायत

{forum_name}

दिनांक: {date}

शिकायतकर्ता: {complainant_name}
पता: {complainant_address}

विरुद्ध

प्रतिपक्षी: {opposite_party}
पता: {opposite_address}

शिकायत:
{complaint_details}

मांगी गई राहत:
{relief_sought}

मैं शपथपूर्वक कहता/कहती हूं कि उपरोक्त तथ्य सत्य हैं।

हस्ताक्षर: ________________
दिनांक: {date}""",
        "english": """CONSUMER COMPLAINT

{forum_name}

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

I solemnly affirm the above facts are true.

Signature: ________________
Date: {date}"""
    },
    "police_complaint": {
        "name": "Police Complaint",
        "description": "Complaint to police station",
        "fields": ["complainant_name", "complainant_address", "complainant_phone",
                   "police_station", "subject", "incident_date", "incident_location",
                   "incident_details", "accused_name"],
        "hindi": """थाना प्रभारी को शिकायत

सेवा में,
थाना प्रभारी
{police_station}

दिनांक: {date}

विषय: {subject}

महोदय,

मैं {complainant_name}, निवासी {complainant_address} सूचित करता/करती हूं:

घटना दिनांक: {incident_date}
घटना स्थान: {incident_location}

विवरण:
{incident_details}

आरोपी: {accused_name}

उचित कार्रवाई की प्रार्थना है।

{complainant_name}
{complainant_phone}""",
        "english": """POLICE COMPLAINT

To,
The Station House Officer
{police_station}

Date: {date}

Subject: {subject}

Sir,

I, {complainant_name}, resident of {complainant_address} report that:

Date of Incident: {incident_date}
Place of Incident: {incident_location}

Details:
{incident_details}

Accused: {accused_name}

Kindly take appropriate action.

{complainant_name}
{complainant_phone}"""
    },
    "vakalatnama": {
        "name": "Vakalatnama",
        "description": "Lawyer appointment document",
        "fields": ["client_name", "client_address", "lawyer_name",
                   "lawyer_enrollment", "court_name", "case_details"],
        "hindi": """वकालतनामा

न्यायालय: {court_name}
दिनांक: {date}

मैं {client_name}, निवासी {client_address}, एतद्द्वारा {lawyer_name} (नामांकन संख्या: {lawyer_enrollment}) को अपना अधिवक्ता नियुक्त करता/करती हूं।

मामला: {case_details}

मैं उपरोक्त अधिवक्ता को मेरी ओर से सभी कानूनी कार्रवाई करने का अधिकार देता/देती हूं।

मुवक्किल हस्ताक्षर: ________________
दिनांक: {date}""",
        "english": """VAKALATNAMA

Court: {court_name}
Date: {date}

I, {client_name}, resident of {client_address}, hereby appoint {lawyer_name} (Enrollment No: {lawyer_enrollment}) as my Advocate.

Matter: {case_details}

I authorize the above advocate to act on my behalf in all legal proceedings.

Client Signature: ________________
Date: {date}"""
    }
}


def get_cache_key(prefix: str, data: str) -> str:
    return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()}"


@mcp.tool()
async def get_template(doc_type: str, language: str = "both") -> dict:
    """
    Get a legal document template.

    Args:
        doc_type: Type of document (legal_notice, rti_application, consumer_complaint, police_complaint, vakalatnama)
        language: Language preference (hindi, english, both)
    """
    template = TEMPLATES.get(doc_type)

    if not template:
        return {
            "status": "error",
            "message": f"Template not found: {doc_type}",
            "available_templates": list(TEMPLATES.keys())
        }

    result = {
        "status": "success",
        "doc_type": doc_type,
        "name": template["name"],
        "description": template["description"],
        "required_fields": template["fields"],
    }

    if language in ["hindi", "both"]:
        result["template_hindi"] = template["hindi"]
    if language in ["english", "both"]:
        result["template_english"] = template["english"]

    return result


@mcp.tool()
async def fill_template(
    doc_type: str,
    user_details: dict,
    language: str = "both"
) -> dict:
    """
    Fill a legal document template with user details.

    Args:
        doc_type: Type of document
        user_details: Dictionary with field values
        language: Output language (hindi, english, both)
    """
    template = TEMPLATES.get(doc_type)
    if not template:
        return {"status": "error", "message": f"Template not found: {doc_type}"}

    # Add date if not provided
    if "date" not in user_details:
        user_details["date"] = datetime.now().strftime("%d/%m/%Y")

    # Fill placeholders with user details or keep placeholder text
    def fill(template_text: str, details: dict) -> str:
        result = template_text
        for key, value in details.items():
            result = result.replace(f"{{{key}}}", str(value))
        # Keep unfilled placeholders visible
        return result

    output = {
        "status": "success",
        "doc_type": doc_type,
        "filled_fields": list(user_details.keys()),
        "missing_fields": [f for f in template["fields"] if f not in user_details]
    }

    if language in ["hindi", "both"]:
        output["document_hindi"] = fill(template["hindi"], user_details)
    if language in ["english", "both"]:
        output["document_english"] = fill(template["english"], user_details)

    return output


@mcp.tool()
async def list_templates() -> dict:
    """List all available document templates."""
    templates_list = []
    for key, tmpl in TEMPLATES.items():
        templates_list.append({
            "id": key,
            "name": tmpl["name"],
            "description": tmpl["description"],
            "required_fields": tmpl["fields"]
        })
    return {"status": "success", "templates": templates_list, "total": len(templates_list)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
