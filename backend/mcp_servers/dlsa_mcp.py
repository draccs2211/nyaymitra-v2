from fastmcp import FastMCP
import json

mcp = FastMCP("DLSA MCP Server")

DLSA_DATA = {
    "lucknow": {
        "name": "DLSA Lucknow",
        "full_name": "District Legal Services Authority, Lucknow",
        "address": "District Court Complex, Lucknow - 226001",
        "phone": "0522-2611818",
        "email": "dlsalucknow@gmail.com",
        "timing": "10:00 AM - 5:00 PM (Mon-Sat)",
        "services": ["Free legal aid", "Lok Adalat", "Mediation", "Legal awareness"],
        "eligibility": "Income below ₹3 lakh/year, SC/ST, Women, Children, Disabled persons"
    },
    "allahabad": {
        "name": "DLSA Allahabad/Prayagraj",
        "full_name": "District Legal Services Authority, Prayagraj",
        "address": "Civil Court Compound, Prayagraj - 211001",
        "phone": "0532-2420711",
        "timing": "10:00 AM - 5:00 PM (Mon-Sat)",
        "services": ["Free legal aid", "Lok Adalat", "Mediation"],
        "eligibility": "Income below ₹3 lakh/year, SC/ST, Women, Children, Disabled persons"
    },
    "agra": {
        "name": "DLSA Agra",
        "full_name": "District Legal Services Authority, Agra",
        "address": "District Court, Agra",
        "phone": "0562-2521234",
        "timing": "10:00 AM - 5:00 PM (Mon-Sat)",
        "services": ["Free legal aid", "Lok Adalat"],
        "eligibility": "Income below ₹3 lakh/year, SC/ST, Women, Children, Disabled persons"
    }
}

NALSA_INFO = {
    "name": "National Legal Services Authority (NALSA)",
    "helpline": "15100",
    "website": "nalsa.gov.in",
    "description": "Provides free legal services to eligible persons",
    "eligibility": [
        "Annual income below ₹3 lakh",
        "SC/ST community members",
        "Women and children",
        "Persons with disabilities",
        "Victims of trafficking",
        "Industrial workmen",
        "Persons in custody"
    ],
    "services": [
        "Free legal representation",
        "Legal advice",
        "Drafting of legal documents",
        "Lok Adalat services",
        "Mediation"
    ]
}


@mcp.tool()
async def find_dlsa(district: str = "lucknow") -> dict:
    """
    Find DLSA office for a district.

    Args:
        district: District name (lucknow, allahabad, agra, etc.)
    """
    district_lower = district.lower().strip()

    # Find matching DLSA
    dlsa = None
    for key, data in DLSA_DATA.items():
        if key in district_lower or district_lower in key:
            dlsa = data
            break

    if dlsa:
        return {
            "status": "found",
            "district": district,
            "dlsa": dlsa,
            "national_helpline": "15100",
            "note": "Free legal aid available for eligible persons"
        }
    else:
        return {
            "status": "general",
            "message": f"Specific DLSA data not available for {district}",
            "national_helpline": "15100",
            "nalsa": NALSA_INFO,
            "instruction": "Call 15100 to find nearest DLSA office"
        }


@mcp.tool()
async def get_eligibility_info() -> dict:
    """Get information about eligibility for free legal aid."""
    return {
        "status": "success",
        "nalsa": NALSA_INFO,
        "how_to_apply": [
            "Visit nearest DLSA office",
            "Call National Legal Aid Helpline: 15100",
            "Apply online at nalsa.gov.in",
            "Contact nearest District Court"
        ],
        "documents_needed": [
            "Income certificate",
            "Identity proof (Aadhaar/Voter ID)",
            "Caste certificate (if SC/ST)",
            "Any relevant case documents"
        ]
    }


@mcp.tool()
async def get_lok_adalat_info(district: str = "lucknow") -> dict:
    """
    Get Lok Adalat schedule and information.

    Args:
        district: District name
    """
    return {
        "status": "success",
        "what_is_lok_adalat": "Lok Adalat is an alternative dispute resolution mechanism where disputes are settled amicably",
        "benefits": [
            "No court fees",
            "Award is final — no appeal",
            "Faster resolution",
            "Mutual agreement based",
            "No strict procedural requirements"
        ],
        "suitable_for": [
            "Motor accident claims",
            "Matrimonial disputes (except divorce)",
            "Labour disputes",
            "Electricity disputes",
            "Pre-litigation disputes"
        ],
        "schedule": "Usually held on second Saturday of every month at District Courts",
        "contact": f"DLSA {district.title()}: 15100",
        "note": "Contact local DLSA for exact schedule"
    }


if __name__ == "__main__":
    mcp.run(transport="stdio")
