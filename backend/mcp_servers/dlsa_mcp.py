from fastmcp import FastMCP
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

mcp = FastMCP("DLSA MCP Server")

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


# ─── Directory data (data/referral/*.json, collected from official NALSA/SLSA pages) ───
_ROOT = Path(__file__).resolve().parents[2]
DLSA_DIRECTORIES = {"Uttar Pradesh": _ROOT / "data" / "referral" / "up_dlsa_directory.json"}
HELPLINES_PATH = _ROOT / "data" / "referral" / "helplines.json"
STATE_DIRECTORY_URLS = {"Uttar Pradesh": "https://uttarpradesh.nalsa.gov.in/dlsa-directory/"}
STATE_CODES = {"IN-UP": "Uttar Pradesh"}

# Former / common names -> official district name (exact lookup, not keyword matching)
DISTRICT_ALIASES = {
    "allahabad": "Prayagraj", "faizabad": "Ayodhya", "noida": "Gautam Buddha Nagar",
    "greater noida": "Gautam Buddha Nagar", "kanpur": "Kanpur Nagar", "sant ravidas nagar": "Bhadohi",
    "lakhimpur": "Lakhimpur Kheri", "kheri": "Lakhimpur Kheri", "rae bareli": "Raebareli",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z]", "", (name or "").lower())


@lru_cache(maxsize=None)
def _directory(state: str) -> dict:
    path = DLSA_DIRECTORIES.get(state)
    if not path or not path.exists():
        return {}
    rows = json.loads(path.read_text(encoding="utf-8")).get("dlsa", [])
    return {_norm(r["district"]): r for r in rows}


def lookup_dlsa(district: str, state: str = "Uttar Pradesh") -> Optional[dict]:
    if not district:
        return None
    official = DISTRICT_ALIASES.get(district.strip().lower(), district)
    return _directory(state).get(_norm(official))


@lru_cache(maxsize=1)
def load_helplines() -> List[dict]:
    if not HELPLINES_PATH.exists():
        return []
    return json.loads(HELPLINES_PATH.read_text(encoding="utf-8")).get("helplines", [])


@mcp.tool()
async def find_dlsa(district: str = "", jurisdiction: str = "IN") -> dict:
    """
    Find the District Legal Services Authority for a district.

    Args:
        district: Official district name (e.g. "Prayagraj"); common old names are accepted
        jurisdiction: ISO code of the state, e.g. "IN-UP" ("IN" if unknown)
    """
    state = STATE_CODES.get(jurisdiction)
    row = lookup_dlsa(district, state) if state else None
    if row:
        return {
            "status": "found",
            "dlsa": {
                "name": f"District Legal Services Authority, {row['district']}",
                "district": row["district"],
                "state": row.get("state") or state,
                "secretary": row.get("secretary") or "",
                "phone": row.get("phone") or "",
                "as_of": row.get("collected") or "",
                "source_url": row.get("source_url") or "",
                "verified_by_human": bool(row.get("verified_by_human")),
            },
            "national_helpline": "15100",
        }
    return {
        "status": "state_directory" if state else "national",
        "district": district,
        "state": state or "",
        "directory_url": STATE_DIRECTORY_URLS.get(state, "https://nalsa.gov.in"),
        "national_helpline": "15100",
        "message": ("District not found in the directory; use the state DLSA directory or call 15100."
                    if state else "Call the NALSA helpline 15100 to reach the nearest DLSA."),
    }


@mcp.tool()
async def list_helplines() -> dict:
    """Helpline numbers from data/referral/helplines.json (with source and verification flag)."""
    return {"status": "success", "helplines": load_helplines()}


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
