"""
Guardian Agent - NyayMitra 2.0
Urgency detector for law firm tool.
Detects emergency/urgent cases and flags them for immediate legal action.
Runs in parallel before any other agent.
"""
from backend.state import NyayMitraState
from backend.agents.schemas import GuardianOutput, to_state
from langsmith import traceable
from typing import Literal

# ─── Emergency Keywords ───────────────────────────────────────────────────────

EMERGENCY_KEYWORDS = {
    "physical_threat": [
    "immediate threat", "jaan ka khatra", "maar raha", "maar rahi",
    "killing", "threatened to kill", "physical abuse", "assault",
    "attack", "hamlaa", "pitai", "maarpeet",
    # Add these:
    "threatened to be killed", "kill my client", "death threat",
    "client threatened", "life threat", "jaan se maarne ki dhamki",
    "threat to life", "murder threat"
],
    "domestic_violence": [
        "domestic violence", "gharelu hinsa", "husband beating",
        "pati maar raha", "wife beating", "abuse at home",
        "498a urgent", "protection order urgent", "dv act urgent"
    ],
    "sexual_assault": [
        "rape", "balatkaar", "sexual assault", "molestation",
        "376", "354", "pocso"
    ],
    "custody_emergency": [
        "child abducted", "child kidnapped", "bachche ko le gaya",
        "bachche ko le gayi", "child taken away illegally",
        "habeas corpus child", "urgent custody"
    ],
    "illegal_detention": [
        "illegally detained", "wrongful arrest", "false imprisonment",
        "galat giraftari", "habeas corpus", "custody without warrant",
        "arrested without reason"
    ],
    "anticipatory_bail_urgent": [
        "arrest imminent", "giraftari hone wali", "anticipatory bail urgent",
        "police coming to arrest", "arrest warrant issued",
        "surrender tomorrow", "bail urgent tonight"
    ],
    "property_emergency": [
        "forceful eviction", "jbardasti nikala", "locked out of house",
        "property sealed illegally", "bulldozer", "demolition notice today",
        "vakf board illegal", "encroachment removal today"
    ],
    "court_emergency": [
        "hearing tomorrow", "court date tomorrow", "order passed today",
        "ex parte order", "contempt notice", "urgent stay required",
        "execution of decree today", "warrant issued"
    ]
}

# ─── Urgency Levels ───────────────────────────────────────────────────────────

URGENCY_LEVELS = {
    "physical_threat": "CRITICAL",
    "domestic_violence": "CRITICAL",
    "sexual_assault": "CRITICAL",
    "custody_emergency": "HIGH",
    "illegal_detention": "HIGH",
    "anticipatory_bail_urgent": "HIGH",
    "property_emergency": "MEDIUM",
    "court_emergency": "MEDIUM"
}

# ─── Legal Remedies for Each Emergency Type ───────────────────────────────────

IMMEDIATE_LEGAL_REMEDIES = {
    "physical_threat": {
        "remedies": [
            "File FIR immediately under IPC/BNS — cognizable offence",
            "Apply for Protection Order under PWDVA 2005 if domestic",
            "File for Anticipatory Bail if client accused",
            "Seek Police Protection Order from Magistrate"
        ],
        "sections": ["Section 154 CrPC/BNSS (FIR)", "Section 18 PWDVA (Protection Order)", "Section 438 CrPC (Anticipatory Bail)"],
        "court": "Duty Magistrate / Sessions Court",
        "time_sensitivity": "File today — do not delay"
    },
    "domestic_violence": {
        "remedies": [
            "File DV complaint with Protection Officer",
            "Apply for Emergency Protection Order — can be same day",
            "File FIR under Section 498A IPC / Section 85 BNS",
            "Apply for Residence Order to prevent eviction from shared household"
        ],
        "sections": ["Section 498A IPC / Section 85 BNS", "Section 12 PWDVA (Magistrate application)", "Section 18 PWDVA (Protection Order)", "Section 19 PWDVA (Residence Order)"],
        "court": "Judicial Magistrate First Class",
        "time_sensitivity": "Same day filing possible for emergency protection order"
    },
    "sexual_assault": {
        "remedies": [
            "File FIR immediately — police must register under Section 154 CrPC",
            "Zero FIR can be filed at any police station",
            "Medical examination within 24 hours critical for evidence",
            "Apply for compensation under victim compensation scheme"
        ],
        "sections": ["Section 376 IPC / Section 63 BNS (Rape)", "Section 354 IPC / Section 74 BNS (Molestation)", "POCSO Act if victim minor"],
        "court": "Special Court / Sessions Court",
        "time_sensitivity": "File FIR within 24 hours — evidence preservation critical"
    },
    "custody_emergency": {
        "remedies": [
            "File Habeas Corpus petition in High Court — emergency listing possible",
            "Apply for custody order from Family Court",
            "File complaint under Section 363 IPC if child taken by non-custodial parent",
            "Contact police for missing child if minor"
        ],
        "sections": ["Article 226 Constitution (Habeas Corpus HC)", "Section 363 IPC / Section 137 BNS (Kidnapping)", "Guardian and Wards Act 1890", "Section 25 Hindu Minority Act"],
        "court": "High Court (Habeas Corpus) / Family Court",
        "time_sensitivity": "Habeas Corpus — emergency listing same day possible"
    },
    "illegal_detention": {
        "remedies": [
            "File Habeas Corpus petition in High Court immediately",
            "Apply for bail before Sessions Court / High Court",
            "File complaint against police if unlawful detention",
            "Section 41 CrPC — arrest without warrant grounds must be justified"
        ],
        "sections": ["Article 21 Constitution (Right to Liberty)", "Article 226 (Habeas Corpus)", "Section 436/437/438/439 CrPC (Bail)", "Section 41 CrPC (Arrest without warrant)"],
        "court": "High Court / Sessions Court",
        "time_sensitivity": "Immediate — person in custody"
    },
    "anticipatory_bail_urgent": {
        "remedies": [
            "File anticipatory bail application in Sessions Court immediately",
            "If Sessions Court unavailable — file in High Court",
            "Prepare surety and documents in advance",
            "File interim anticipatory bail application for immediate relief"
        ],
        "sections": ["Section 438 CrPC / Section 482 BNSS (Anticipatory Bail)", "Section 439 CrPC (High Court Bail)"],
        "court": "Sessions Court / High Court",
        "time_sensitivity": "File before arrest — time critical"
    },
    "property_emergency": {
        "remedies": [
            "Apply for Stay Order / Injunction in Civil Court",
            "File for Status Quo Order",
            "Apply for Urgent Interim Relief in HC if municipal body involved",
            "Challenge demolition notice — file writ petition in HC"
        ],
        "sections": ["Order 39 Rule 1 CPC (Temporary Injunction)", "Order 39 Rule 2 CPC (Injunction)", "Article 226 Constitution (Writ Petition)"],
        "court": "Civil Court / High Court",
        "time_sensitivity": "Same day injunction possible in urgent cases"
    },
    "court_emergency": {
        "remedies": [
            "File for adjournment application immediately",
            "Apply for Stay of proceedings",
            "File urgent application before same court",
            "Mention matter before court for urgent listing"
        ],
        "sections": ["Order 17 CPC (Adjournment)", "Section 482 CrPC (HC inherent powers)", "Order 39 Rule 1 CPC (Stay)"],
        "court": "Same court where matter pending / High Court",
        "time_sensitivity": "Before next hearing date"
    }
}

# ─── Detect Emergency Type ────────────────────────────────────────────────────

def detect_emergency_type(query: str) -> tuple[bool, str, str]:
    """
    Detect if query is an emergency and what type.

    Returns:
        (is_emergency, emergency_type, urgency_level)
    """
    query_lower = query.lower()

    for emergency_type, keywords in EMERGENCY_KEYWORDS.items():
        for keyword in keywords:
            if keyword.lower() in query_lower:
                urgency_level = URGENCY_LEVELS.get(emergency_type, "MEDIUM")
                return True, emergency_type, urgency_level

    return False, "", ""


def build_lawyer_alert(emergency_type: str, urgency_level: str) -> str:
    """
    Build alert message for lawyer — not helplines, but legal action guidance.
    """
    remedies = IMMEDIATE_LEGAL_REMEDIES.get(emergency_type, {})

    if not remedies:
        return f"⚠️ URGENT CASE DETECTED — Requires immediate legal attention."

    urgency_emoji = "🚨" if urgency_level == "CRITICAL" else "⚡" if urgency_level == "HIGH" else "⚠️"

    alert = f"""{urgency_emoji} URGENT CASE — {urgency_level} PRIORITY

━━━━━━━━━━━━━━━━━━━━━━━━
IMMEDIATE LEGAL REMEDIES:
━━━━━━━━━━━━━━━━━━━━━━━━"""

    for i, remedy in enumerate(remedies.get("remedies", []), 1):
        alert += f"\n{i}. {remedy}"

    alert += f"""

📋 APPLICABLE SECTIONS:
{chr(10).join(f'• {s}' for s in remedies.get('sections', []))}

🏛️ COURT: {remedies.get('court', 'Appropriate Court')}

⏰ TIME SENSITIVITY: {remedies.get('time_sensitivity', 'Act immediately')}

━━━━━━━━━━━━━━━━━━━━━━━━
Note: Proceed with full legal research below alongside immediate action.
━━━━━━━━━━━━━━━━━━━━━━━━"""

    return alert


# ─── Guardian Agent ───────────────────────────────────────────────────────────

@traceable(name="Guardian Agent")
def guardian_agent(state: NyayMitraState) -> dict:
    """
    Guardian Agent — Urgency detector for law firm tool.

    Detects emergency/urgent cases and provides:
    - Urgency level (CRITICAL / HIGH / MEDIUM)
    - Immediate legal remedies
    - Applicable sections
    - Court to approach
    - Time sensitivity

    Does NOT replace full legal research — just flags urgency
    and provides immediate action guidance alongside normal pipeline.
    """
    query = state.get("user_query", "")
    language = state.get("language", "en")

    # Detect emergency
    is_emergency, emergency_type, urgency_level = detect_emergency_type(query)

    if is_emergency:
        # Build lawyer-focused alert
        alert = build_lawyer_alert(emergency_type, urgency_level)

        return to_state(GuardianOutput(
            is_emergency=True,
            emergency_type=emergency_type,
            final_response=alert,
            confidence_score=1.0,
            human_approval_needed=False,
            case_strength="urgent",
        ))

    # Normal case — no emergency
    return to_state(GuardianOutput(is_emergency=False, emergency_type=""))


# ─── Test ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_queries = [
        # Normal queries
        "My client's landlord is not returning security deposit",
        "Need to file RTI for government scheme",
        "Consumer forum complaint for defective product",

        # Emergency queries
        "Client facing domestic violence from husband — needs immediate help",
        "My client will be arrested tomorrow — anticipatory bail urgent",
        "Habeas corpus — client illegally detained by police",
        "Demolition notice received today — court order needed",
        "Custody — child abducted by father illegally",
        "Ex parte order passed — urgent stay required",
        "Client threatened to be killed — protection needed",
    ]

    print("=" * 60)
    print("GUARDIAN AGENT TEST")
    print("=" * 60)

    base_state = {
        "user_query": "",
        "language": "en",
        "session_id": "test",
        "user_id": "test",
        "conversation_history": [],
        "legal_domain": "general",
        "needs_document": False,
        "needs_research": True,
        "needs_referral": False,
        "is_emergency": False,
        "emergency_type": "",
        "research_results": [],
        "kanoon_cases": [],
        "ipc_sections": [],
        "case_analysis": {},
        "case_strength": "medium",
        "draft_document": "",
        "document_type": "",
        "document_hindi": "",
        "document_english": "",
        "confidence_score": 0.0,
        "hallucination_detected": False,
        "human_approval_needed": False,
        "verified_sources": [],
        "dlsa_offices": [],
        "lawyer_referrals": [],
        "final_response": "",
        "final_response_hindi": "",
        "sources": [],
        "next_steps": [],
        "error": None,
        "retry_count": 0,
    }

    for query in test_queries:
        base_state["user_query"] = query
        result = guardian_agent(base_state)

        print(f"\nQuery: {query[:60]}")
        print(f"Emergency: {result['is_emergency']}", end="")

        if result["is_emergency"]:
            print(f" | Type: {result['emergency_type']} | Urgency: {URGENCY_LEVELS.get(result['emergency_type'], 'N/A')}")
            print("Alert preview:")
            print(result["final_response"][:200])
        else:
            print(" — Normal case, proceed to full pipeline")
        print("-" * 60)