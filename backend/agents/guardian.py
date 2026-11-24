"""
Guardian Agent - NyayMitra 2.0
Urgency detector for the law firm tool.

LLM classifies whether the matter needs action within ~48 hours and which
emergency type it is (no keyword matching). The pipeline ALWAYS continues;
the alert is stored in `urgent_alert` and shown at the top of the final
response by the verification agent. If the LLM call fails, the guardian
fails open (no alert, guardian_status="unavailable").
"""
from backend.state import NyayMitraState, initial_state
from backend.config import OPENAI_API_KEY
from backend.agents.schemas import GuardianLLMOutput, GuardianOutput, to_state
from langsmith import traceable
from openai import OpenAI
import json

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

# ─── Client safety helplines (CRITICAL alerts only) ───────────────────────────

CRITICAL_SAFETY_HELPLINES = [
    "112 — National emergency number (police / ambulance)",
    "181 — Women helpline",
    "1098 — Child helpline",
    "15100 — NALSA free legal aid",
]

# ─── LLM classification ───────────────────────────────────────────────────────

GUARDIAN_PROMPT = """You are the urgency screener for an Indian law firm's legal research tool.
Lawyers (not victims) type the queries, often about their clients.

Decide if the matter needs legal action within about 48 hours, and classify it.

Emergency types:
- physical_threat: ongoing or imminent threat to someone's life or body
- domestic_violence: ongoing abuse in a domestic relationship needing protection now
- sexual_assault: recent sexual assault where FIR / medical evidence is time-critical
- custody_emergency: child taken or held illegally right now
- illegal_detention: someone currently detained/arrested unlawfully
- anticipatory_bail_urgent: arrest is imminent
- property_emergency: eviction, sealing or demolition happening now or within days
- court_emergency: hearing, order execution or contempt within about 48 hours
- other: time-critical but none of the above

NOT an emergency: legal research or drafting ABOUT such offences without present
time pressure (e.g. "research case law on Section 376 for a trial next month"),
past events with no ongoing risk, or amounts/numbers that merely resemble section numbers.

Return ONLY JSON:
{"is_emergency": true/false, "emergency_type": "<type above, or empty if not an emergency>", "reasoning": "<one line>"}"""


def classify_urgency(query: str) -> GuardianLLMOutput:
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": GUARDIAN_PROMPT},
            {"role": "user", "content": f"Query: {query}"},
        ],
        max_tokens=150,
        temperature=0,
        response_format={"type": "json_object"},
    )
    return GuardianLLMOutput.model_validate(json.loads(response.choices[0].message.content))


def build_lawyer_alert(emergency_type: str, urgency_level: str, reasoning: str = "") -> str:
    """
    Alert for the lawyer: immediate legal remedies, plus client-safety
    helplines when the urgency is CRITICAL (danger to a person).
    """
    remedies = IMMEDIATE_LEGAL_REMEDIES.get(emergency_type, {})

    if not remedies:
        alert = f"⚠️ URGENT CASE — {urgency_level} PRIORITY\nTime-sensitive matter: act promptly."
        if reasoning:
            alert += f"\nScreening note: {reasoning}"
        return alert + "\n\nFull research and analysis follow below."

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

⏰ TIME SENSITIVITY: {remedies.get('time_sensitivity', 'Act immediately')}"""

    if urgency_level == "CRITICAL":
        alert += "\n\n📞 FOR THE CLIENT'S IMMEDIATE SAFETY:\n"
        alert += "\n".join(f"• {h}" for h in CRITICAL_SAFETY_HELPLINES)

    alert += """

━━━━━━━━━━━━━━━━━━━━━━━━
Full research and analysis follow below.
━━━━━━━━━━━━━━━━━━━━━━━━"""

    return alert


# ─── Guardian Agent ───────────────────────────────────────────────────────────

@traceable(name="Guardian Agent")
def guardian_agent(state: NyayMitraState) -> dict:
    """
    Flags urgency and builds an alert; never stops the pipeline.
    Fails open: if classification is unavailable, no alert is raised.
    """
    query = state.get("user_query", "")

    try:
        result = classify_urgency(query)
    except Exception as e:
        print(f"[Guardian] Classification unavailable: {e}")
        return to_state(GuardianOutput(
            is_emergency=False, emergency_type="", urgency_level="",
            urgent_alert="", guardian_status="unavailable",
        ))

    if not result.is_emergency:
        return to_state(GuardianOutput(
            is_emergency=False, emergency_type="", urgency_level="",
            urgent_alert="", guardian_status="checked",
        ))

    urgency_level = URGENCY_LEVELS.get(result.emergency_type, "MEDIUM")
    print(f"[Guardian] Emergency: {result.emergency_type} ({urgency_level}) — {result.reasoning}")
    return to_state(GuardianOutput(
        is_emergency=True,
        emergency_type=result.emergency_type,
        urgency_level=urgency_level,
        urgent_alert=build_lawyer_alert(result.emergency_type, urgency_level, result.reasoning),
        guardian_status="checked",
    ))


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

        # Should NOT be emergencies (old keyword false positives)
        "Research Supreme Court case law on Section 376 IPC for a trial next month",
        "Client paid Rs 3760 advance to the builder, wants refund",
    ]

    print("=" * 60)
    print("GUARDIAN AGENT TEST")
    print("=" * 60)

    base_state = initial_state()

    for query in test_queries:
        base_state["user_query"] = query
        result = guardian_agent(base_state)

        print(f"\nQuery: {query[:60]}")
        print(f"Emergency: {result['is_emergency']}", end="")

        if result["is_emergency"]:
            print(f" | Type: {result['emergency_type']} | Urgency: {result['urgency_level']}")
            print("Alert preview:")
            print(result["urgent_alert"][:200])
        else:
            print(" — Normal case, proceed to full pipeline")
        print("-" * 60)