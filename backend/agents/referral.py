"""
Referral Agent - NyayMitra 2.0

Free legal aid referral from data, not hard-coded numbers:
- DLSA for the detected district via the DLSA MCP server
  (data/referral/*_dlsa_directory.json, collected from official NALSA/SLSA pages)
- helplines from data/referral/helplines.json (15100 always + one for the legal domain)
Falls back to the state DLSA directory link or the NALSA helpline when the
district is unknown or the MCP call fails.
"""
import logging
from typing import List, Optional

from langsmith import traceable

from backend.agents.schemas import DlsaOffice, ReferralOutput, to_state
from backend.mcp_client import MCPCallError, call_tool, run_sync
from backend.state import NyayMitraState

logger = logging.getLogger("nyaymitra.referral")

NATIONAL_HELPLINE = "15100"
# Legal domain (LLM-classified) -> helpline number shown in addition to 15100
DOMAIN_HELPLINE = {"consumer": "1915", "family": "181"}

LABELS = {
    "en": {
        "title": "Free legal aid", "dlsa": "District Legal Services Authority", "secretary": "Secretary",
        "phone": "Phone", "as_of": "as of", "source": "Source", "directory": "Find the DLSA for the district",
        "national": "NALSA legal aid helpline", "other": "Other helplines",
        "lok_adalat": "DLSAs also organise Lok Adalats for settling disputes without full litigation.",
        "verify": "Contacts change; confirm before sharing with a client.",
    },
    "hi": {
        "title": "मुफ्त कानूनी सहायता", "dlsa": "जिला विधिक सेवा प्राधिकरण", "secretary": "सचिव",
        "phone": "फोन", "as_of": "दिनांक", "source": "स्रोत", "directory": "जिले का DLSA खोजें",
        "national": "NALSA कानूनी सहायता हेल्पलाइन", "other": "अन्य हेल्पलाइन",
        "lok_adalat": "DLSA लोक अदालतें भी आयोजित करते हैं, जहाँ बिना लंबे मुकदमे के विवाद सुलझ सकते हैं।",
        "verify": "संपर्क बदलते रहते हैं; मुवक्किल को देने से पहले पुष्टि करें।",
    },
}


def _find_dlsa(district: str, jurisdiction: str) -> dict:
    try:
        return run_sync(call_tool("dlsa", "find_dlsa", {"district": district, "jurisdiction": jurisdiction}))
    except MCPCallError as e:
        logger.warning("DLSA lookup unavailable: %s", e)
        return {"status": "national", "national_helpline": NATIONAL_HELPLINE, "directory_url": "https://nalsa.gov.in"}


def _helplines(domain: str) -> List[dict]:
    try:
        data = run_sync(call_tool("dlsa", "list_helplines", {}))
        rows = data.get("helplines", [])
    except MCPCallError as e:
        logger.warning("Helpline list unavailable: %s", e)
        rows = []
    wanted = DOMAIN_HELPLINE.get(domain)
    return [h for h in rows if wanted and h.get("number") == wanted]


def build_referral_text(result: dict, extra: List[dict], language: str) -> str:
    t = LABELS["hi" if language == "hi" else "en"]
    lines = [f"🏛️ {t['title']}"]
    dlsa = result.get("dlsa")
    if dlsa:
        lines.append(f"📍 {t['dlsa']}, {dlsa['district']} ({dlsa.get('state', '')})")
        if dlsa.get("secretary"):
            lines.append(f"{t['secretary']}: {dlsa['secretary']}")
        if dlsa.get("phone"):
            as_of = f" ({t['as_of']} {dlsa['as_of']})" if dlsa.get("as_of") else ""
            lines.append(f"{t['phone']}: {dlsa['phone']}{as_of}")
        if dlsa.get("source_url"):
            lines.append(f"{t['source']}: {dlsa['source_url']}")
    elif result.get("directory_url"):
        lines.append(f"📍 {t['directory']}: {result['directory_url']}")
    lines.append(f"📞 {t['national']}: {NATIONAL_HELPLINE}")
    if extra:
        lines.append(f"{t['other']}: " + ", ".join(f"{h['name']} {h['number']}" for h in extra))
    lines.append(f"💡 {t['lok_adalat']}")
    lines.append(f"ⓘ {t['verify']}")
    return "\n".join(lines)


def _office(result: dict) -> Optional[DlsaOffice]:
    d = result.get("dlsa")
    if not d:
        return None
    return DlsaOffice(
        name=d["name"], district=d.get("district", ""), state=d.get("state", ""),
        secretary=d.get("secretary", ""), phone=d.get("phone", ""), as_of=d.get("as_of", ""),
        source_url=d.get("source_url", ""), verified_by_human=bool(d.get("verified_by_human")),
    )


@traceable(name="Referral Agent")
def referral_agent(state: NyayMitraState) -> dict:
    language = state.get("language", "en")
    result = _find_dlsa(state.get("district", ""), state.get("jurisdiction") or "IN")
    extra = _helplines(state.get("legal_domain", "general"))
    referral_text = build_referral_text(result, extra, language)

    current = state.get("final_response", "")
    office = _office(result)
    logger.info("Referral: status=%s district=%s", result.get("status"), office.district if office else "-")
    return to_state(ReferralOutput(
        dlsa_offices=[office] if office else [],
        lawyer_referrals=[],
        final_response=f"{current}\n\n{referral_text}" if current else referral_text,
    ))
