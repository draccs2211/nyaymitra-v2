from backend.state import NyayMitraState
from backend.agents.schemas import DlsaOffice, ReferralOutput, to_state
from langsmith import traceable


DLSA_OFFICES = {
    "lucknow": {
        "name": "District Legal Services Authority, Lucknow",
        "address": "District Court Complex, Lucknow, UP - 226001",
        "phone": "0522-2611818",
        "helpline": "15100",
        "timing": "10 AM - 5 PM (Mon-Sat)"
    },
    "allahabad": {
        "name": "District Legal Services Authority, Allahabad",
        "address": "Civil Court Compound, Allahabad, UP",
        "phone": "0532-2420711",
        "helpline": "15100",
        "timing": "10 AM - 5 PM (Mon-Sat)"
    },
    "default": {
        "name": "District Legal Services Authority",
        "address": "Contact your nearest District Court",
        "phone": "15100 (National Legal Aid Helpline)",
        "helpline": "15100",
        "timing": "10 AM - 5 PM (Mon-Sat)"
    }
}

HELPLINES = {
    "legal_aid": "15100",
    "women": "1091",
    "police": "100",
    "child": "1098",
    "senior": "14567",
    "consumer": "1800-11-4000",
}


@traceable(name="Referral Agent")
def referral_agent(state: NyayMitraState) -> dict:
    language = state.get("language", "en")
    domain = state.get("legal_domain", "general")

    # Get DLSA info
    dlsa = DLSA_OFFICES.get("lucknow", DLSA_OFFICES["default"])

    dlsa_offices = [dlsa]

    # Build referral response
    if language == "hi":
        referral_text = f"""
🏛️ मुफ्त कानूनी सहायता उपलब्ध है:

📍 {dlsa['name']}
पता: {dlsa['address']}
फोन: {dlsa['phone']}
राष्ट्रीय हेल्पलाइन: {dlsa['helpline']}
समय: {dlsa['timing']}

📞 अन्य महत्वपूर्ण नंबर:
• कानूनी सहायता: 15100
• महिला हेल्पलाइन: 1091
• पुलिस: 100
• उपभोक्ता हेल्पलाइन: 1800-11-4000

💡 लोक अदालत: हर महीने जिला न्यायालय में लोक अदालत होती है जहां बिना मुकदमे के समझौता हो सकता है।
"""
    else:
        referral_text = f"""
🏛️ Free Legal Aid Available:

📍 {dlsa['name']}
Address: {dlsa['address']}
Phone: {dlsa['phone']}
National Helpline: {dlsa['helpline']}
Timing: {dlsa['timing']}

📞 Important Numbers:
• Legal Aid: 15100
• Women Helpline: 1091
• Police: 100
• Consumer Helpline: 1800-11-4000

💡 Lok Adalat: Monthly Lok Adalats at District Courts allow dispute resolution without full litigation.
"""

    current_response = state.get("final_response", "")
    final_response = current_response + "\n\n" + referral_text if current_response else referral_text

    return to_state(ReferralOutput(
        dlsa_offices=[DlsaOffice(**o) for o in dlsa_offices],
        lawyer_referrals=[],
        final_response=final_response,
    ))
