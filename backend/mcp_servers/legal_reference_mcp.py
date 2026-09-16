from fastmcp import FastMCP
from backend.database.connection import get_redis
import json
import hashlib

mcp = FastMCP("Legal Reference MCP Server")
redis_client = get_redis()

# IPC to BNS mapping (key sections)
IPC_TO_BNS = {
    "302": {"bns": "101", "description": "Murder", "punishment": "Death or life imprisonment + fine"},
    "304": {"bns": "105", "description": "Culpable homicide not amounting to murder", "punishment": "Life imprisonment or 10 years + fine"},
    "307": {"bns": "109", "description": "Attempt to murder", "punishment": "10 years to life imprisonment"},
    "354": {"bns": "74", "description": "Assault on woman with intent to outrage modesty", "punishment": "1-5 years + fine"},
    "376": {"bns": "63", "description": "Rape", "punishment": "7 years to life imprisonment"},
    "379": {"bns": "303", "description": "Theft", "punishment": "3 years or fine or both"},
    "380": {"bns": "305", "description": "Theft in dwelling house", "punishment": "7 years + fine"},
    "392": {"bns": "309", "description": "Robbery", "punishment": "10 years + fine"},
    "395": {"bns": "310", "description": "Dacoity", "punishment": "Life imprisonment or 10 years + fine"},
    "406": {"bns": "316", "description": "Criminal breach of trust", "punishment": "3 years or fine or both"},
    "420": {"bns": "318", "description": "Cheating and dishonestly inducing delivery of property", "punishment": "7 years + fine"},
    "498A": {"bns": "85", "description": "Cruelty by husband or relatives", "punishment": "3 years + fine"},
    "504": {"bns": "352", "description": "Intentional insult to provoke breach of peace", "punishment": "2 years or fine or both"},
    "506": {"bns": "351", "description": "Criminal intimidation", "punishment": "2-7 years + fine"},
    "509": {"bns": "79", "description": "Word or gesture intended to insult modesty of woman", "punishment": "3 years or fine or both"},
}

# Key legal provisions
LEGAL_PROVISIONS = {
    "crpc_154": {
        "act": "CrPC / BNSS",
        "section": "Section 154 CrPC / Section 173 BNSS",
        "title": "First Information Report (FIR)",
        "description": "Every information relating to a cognizable offence must be reduced to writing by the officer in charge of a police station and shall be signed by the informant.",
        "key_points": [
            "FIR is mandatory for cognizable offences",
            "Police cannot refuse to register FIR",
            "Can approach Superintendent of Police if refused",
            "Zero FIR can be filed at any police station"
        ]
    },
    "consumer_jurisdiction": {
        "act": "Consumer Protection Act 2019",
        "section": "Section 34, 47, 58",
        "title": "Jurisdiction of Consumer Forums",
        "description": "Jurisdiction based on value of goods/services and compensation claimed.",
        "key_points": [
            "District Commission: Up to ₹50 lakh",
            "State Commission: ₹50 lakh to ₹2 crore",
            "National Commission: Above ₹2 crore",
            "Complaint can be filed online: edaakhil.nic.in"
        ]
    },
    "rti_process": {
        "act": "RTI Act 2005",
        "section": "Section 6, 7, 19",
        "title": "RTI Application Process",
        "description": "Process for filing RTI applications and appeals.",
        "key_points": [
            "Application fee: ₹10 (Free for BPL)",
            "Response within 30 days",
            "First Appeal to First Appellate Authority",
            "Second Appeal to CIC/SIC",
            "Online filing: rtionline.gov.in"
        ]
    },
    "tenant_rights_up": {
        "act": "UP Urban Buildings Act 1972",
        "section": "Section 20, 21, 34",
        "title": "Tenant Rights in UP",
        "description": "Rights of tenants under UP Urban Buildings (Regulation of Letting, Rent and Eviction) Act 1972.",
        "key_points": [
            "Landlord cannot evict without court order",
            "Security deposit must be returned within 30 days of vacating",
            "Rent cannot be increased arbitrarily",
            "Tenant has right to get receipt for rent paid",
            "Disputes go to Rent Control Court"
        ]
    },
    "domestic_violence": {
        "act": "Protection of Women from Domestic Violence Act 2005",
        "section": "Section 12, 18, 19, 20",
        "title": "Domestic Violence Protection",
        "description": "Protection orders and remedies available under PWDVA 2005.",
        "key_points": [
            "Protection order against abuser",
            "Right to reside in shared household",
            "Monetary relief including compensation",
            "Custody orders for children",
            "Contact Protection Officers or NGOs for help"
        ]
    },
    "bail_types": {
        "act": "CrPC / BNSS",
        "section": "Section 436, 437, 438, 439 CrPC",
        "title": "Types of Bail",
        "description": "Different types of bail available in criminal matters.",
        "key_points": [
            "Regular Bail: After arrest - Section 437/439",
            "Anticipatory Bail: Before arrest - Section 438",
            "Interim Bail: Temporary bail",
            "Statutory Bail: When chargesheet not filed in time",
            "Default Bail: Right when investigation incomplete"
        ]
    }
}


def get_cache_key(prefix: str, data: str) -> str:
    return f"{prefix}:{hashlib.md5(data.encode()).hexdigest()}"


@mcp.tool()
async def get_ipc_section(section_number: str) -> dict:
    """
    Get details of an IPC section and its equivalent BNS section.

    Args:
        section_number: IPC section number e.g. "302", "420", "498A"
    """
    cache_key = get_cache_key("ipc", section_number)
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    section = IPC_TO_BNS.get(section_number.strip())
    if section:
        result = {
            "status": "found",
            "ipc_section": f"Section {section_number} IPC",
            "bns_equivalent": f"Section {section['bns']} BNS 2023",
            "description": section["description"],
            "punishment": section["punishment"],
            "note": "India has replaced IPC with Bharatiya Nyaya Sanhita (BNS) 2023. Both may be applicable depending on when the offence was committed.",
            "official_source": "https://legislative.gov.in"
        }
    else:
        result = {
            "status": "not_found",
            "ipc_section": f"Section {section_number} IPC",
            "message": "Section not in quick reference. Please check India Code website.",
            "official_source": "https://www.indiacode.nic.in"
        }

    redis_client.setex(cache_key, 604800, json.dumps(result))
    return result


@mcp.tool()
async def get_legal_provision(topic: str) -> dict:
    """
    Get key legal provisions for common legal topics.

    Args:
        topic: Legal topic e.g. "fir", "consumer", "rti", "tenant", "bail", "domestic_violence"
    """
    cache_key = get_cache_key("provision", topic)
    cached = redis_client.get(cache_key)
    if cached:
        return json.loads(cached)

    # Try to match topic
    topic_lower = topic.lower()
    matched = None

    if any(kw in topic_lower for kw in ["fir", "first information", "police complaint"]):
        matched = LEGAL_PROVISIONS["crpc_154"]
    elif any(kw in topic_lower for kw in ["consumer", "product", "refund", "service"]):
        matched = LEGAL_PROVISIONS["consumer_jurisdiction"]
    elif any(kw in topic_lower for kw in ["rti", "information", "right to information"]):
        matched = LEGAL_PROVISIONS["rti_process"]
    elif any(kw in topic_lower for kw in ["tenant", "rent", "landlord", "kiraya"]):
        matched = LEGAL_PROVISIONS["tenant_rights_up"]
    elif any(kw in topic_lower for kw in ["domestic", "violence", "hinsa", "498"]):
        matched = LEGAL_PROVISIONS["domestic_violence"]
    elif any(kw in topic_lower for kw in ["bail", "zamanat", "arrest"]):
        matched = LEGAL_PROVISIONS["bail_types"]

    if matched:
        result = {"status": "found", **matched}
    else:
        result = {
            "status": "not_found",
            "message": f"No specific provision found for: {topic}",
            "suggestion": "Please check India Code: https://www.indiacode.nic.in or contact DLSA: 15100"
        }

    redis_client.setex(cache_key, 604800, json.dumps(result))
    return result


@mcp.tool()
async def search_up_laws(query: str) -> dict:
    """
    Search UP specific laws and regulations.

    Args:
        query: Search query for UP specific laws
    """
    UP_LAWS = {
        "rent": {
            "name": "UP Urban Buildings (Regulation of Letting, Rent and Eviction) Act, 1972",
            "key_sections": ["Section 20 - Eviction grounds", "Section 21 - Rent fixation", "Section 34 - Deposit return"],
            "court": "Rent Control Court",
            "contact": "District Court, Lucknow"
        },
        "land": {
            "name": "UP Revenue Code 2006 + UP Zamindari Abolition Act 1950",
            "key_sections": ["Land records", "Mutation process", "Encroachment removal"],
            "court": "Revenue Court / Civil Court",
            "contact": "District Collector / Tehsildar Office"
        },
        "police": {
            "name": "UP Police Act 2007 + IPC/BNS",
            "key_sections": ["FIR registration", "Complaint to SP", "Police accountability"],
            "court": "Criminal Court",
            "contact": "UP Police: 112, SP Office"
        }
    }

    query_lower = query.lower()
    results = []

    for key, law in UP_LAWS.items():
        if key in query_lower or any(word in query_lower for word in key.split()):
            results.append(law)

    if results:
        return {"status": "found", "query": query, "laws": results}
    else:
        return {
            "status": "general",
            "query": query,
            "message": "For specific UP laws, contact:",
            "resources": [
                "UP Government Portal: uppolice.gov.in",
                "UP Revenue Department: upbhulekh.gov.in",
                "Allahabad High Court: allahabadhighcourt.in",
                "DLSA Lucknow: 0522-2611818"
            ]
        }


if __name__ == "__main__":
    mcp.run(transport="stdio")
