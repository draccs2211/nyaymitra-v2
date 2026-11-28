from fastmcp import FastMCP
from datetime import datetime

from backend.templates import load_templates, placeholders
import json

mcp = FastMCP("Document Templates MCP Server")

# Templates live in data/templates/document_templates.json (shared with the drafting agent)


@mcp.tool()
async def get_template(doc_type: str, language: str = "both") -> dict:
    """
    Get a legal document template.

    Args:
        doc_type: Type of document (legal_notice, rti_application, consumer_complaint, police_complaint, vakalatnama)
        language: Language preference (hindi, english, both)
    """
    template = load_templates().get(doc_type)

    if not template:
        return {
            "status": "error",
            "message": f"Template not found: {doc_type}",
            "available_templates": list(load_templates())
        }

    result = {
        "status": "success",
        "doc_type": doc_type,
        "name": template["name"],
        "description": template["description"],
        "required_fields": placeholders(template),
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
    template = load_templates().get(doc_type)
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
        "missing_fields": [f for f in placeholders(template) if f not in user_details]
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
    for key, tmpl in load_templates().items():
        templates_list.append({
            "id": key,
            "name": tmpl["name"],
            "description": tmpl["description"],
            "required_fields": placeholders(tmpl)
        })
    return {"status": "success", "templates": templates_list, "total": len(templates_list)}


if __name__ == "__main__":
    mcp.run(transport="stdio")
