"""
Orchestrator Agent - NyayMitra 2.0
Uses OpenAI GPT-4o-mini to intelligently classify legal queries.
No hardcoded keywords — pure LLM-based classification.
"""
from backend.state import NyayMitraState
from backend.config import OPENAI_API_KEY
from langsmith import traceable
from openai import OpenAI
import json


CLASSIFICATION_PROMPT = """You are an expert Indian legal classifier for a law firm AI tool.

Analyze the given legal query and return a JSON response with the following fields:

{
    "legal_domain": "<one of: criminal, civil, family, property, consumer, rti, labor, constitutional, general>",
    "needs_document": <true if user needs a document drafted — legal notice, RTI, complaint, petition, vakalatnama, affidavit etc., else false>,
    "needs_referral": <true if user needs referral to lawyer, DLSA, legal aid, helpline, else false>,
    "language": "<one of: en, hi, hinglish>",
    "urgency": "<one of: critical, high, medium, low>",
    "query_type": "<one of: research, drafting, analysis, referral, general>",
    "key_issues": ["<issue 1>", "<issue 2>"],
    "suggested_acts": ["<Act 1>", "<Act 2>"],
    "reasoning": "<one line explanation of classification>"
}

Domain definitions:
- criminal: FIR, arrest, bail, IPC/BNS sections, police complaints, criminal cases
- civil: property disputes, contracts, suits, injunctions, recovery
- family: divorce, custody, maintenance, matrimonial disputes, adoption
- property: rent, tenant, landlord, real estate, land, possession, eviction
- consumer: product defects, service issues, refunds, consumer forum, insurance
- rti: Right to Information applications, government information requests
- labor: salary, employment, termination, PF, ESI, gratuity, labor disputes
- constitutional: fundamental rights, writ petitions, PIL, High Court, Supreme Court
- general: general legal advice not fitting above categories

Language detection:
- hi: Query predominantly in Hindi (Devanagari script)
- hinglish: Mix of Hindi and English, or Roman Hindi
- en: Query predominantly in English

Urgency levels:
- critical: Immediate physical danger, ongoing crime, immediate arrest
- high: Court date within 24 hours, anticipatory bail, custody emergency
- medium: Time-sensitive but not immediate — within a week
- low: General research, no time pressure

Return ONLY valid JSON. No explanation outside JSON."""


@traceable(name="Orchestrator Agent")
def orchestrator_agent(state: NyayMitraState) -> dict:
    """
    LLM-based orchestrator that classifies legal queries using OpenAI.
    Determines legal domain, document needs, language, and urgency.
    """
    query = state.get("user_query", "")
    conversation_history = state.get("conversation_history", [])

    # Build context from conversation history
    context = ""
    if conversation_history:
        recent = conversation_history[-4:]
        context = "\n".join([
            f"{msg['role'].upper()}: {msg['content'][:100]}"
            for msg in recent
        ])

    user_message = f"""Legal Query: {query}

{f'Recent conversation context:{chr(10)}{context}' if context else ''}

Classify this query and return JSON."""

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": CLASSIFICATION_PROMPT},
                {"role": "user", "content": user_message}
            ],
            max_tokens=500,
            temperature=0.1,
            response_format={"type": "json_object"}
        )

        result_text = response.choices[0].message.content
        result = json.loads(result_text)

        legal_domain = result.get("legal_domain", "general")
        needs_document = result.get("needs_document", False)
        needs_referral = result.get("needs_referral", False)
        language = result.get("language", "en")
        urgency = result.get("urgency", "low")
        query_type = result.get("query_type", "general")
        key_issues = result.get("key_issues", [])
        suggested_acts = result.get("suggested_acts", [])
        reasoning = result.get("reasoning", "")

        print(f"[Orchestrator] Domain: {legal_domain} | Lang: {language} | Urgency: {urgency} | Doc: {needs_document}")
        print(f"[Orchestrator] Reasoning: {reasoning}")

        return {
            "legal_domain": legal_domain,
            "needs_document": needs_document,
            "needs_research": True,
            "needs_referral": needs_referral,
            "language": language,
            "case_analysis": {
                "urgency": urgency,
                "query_type": query_type,
                "key_issues": key_issues,
                "suggested_acts": suggested_acts,
                "reasoning": reasoning
            }
        }

    except json.JSONDecodeError as e:
        print(f"[Orchestrator] JSON parse error: {e}")
        return _fallback_classification(query)

    except Exception as e:
        print(f"[Orchestrator] Error: {e}")
        return _fallback_classification(query)


def _fallback_classification(query: str) -> dict:
    """Simple fallback if OpenAI fails."""
    hindi_chars = sum(1 for c in query if '\u0900' <= c <= '\u097F')
    if hindi_chars > 5:
        language = "hi"
    elif any(w in query.lower() for w in ["kya", "mera", "hai", "nahi", "kaise", "chahiye"]):
        language = "hinglish"
    else:
        language = "en"

    print("[Orchestrator] Using fallback classification")

    return {
        "legal_domain": "general",
        "needs_document": False,
        "needs_research": True,
        "needs_referral": False,
        "language": language,
        "case_analysis": {
            "urgency": "low",
            "query_type": "general",
            "key_issues": [],
            "suggested_acts": [],
            "reasoning": "Fallback classification used"
        }
    }


if __name__ == "__main__":
    test_queries = [
        "My landlord has not returned my security deposit after 3 months",
        "I want to file FIR against my neighbor for threatening me",
        "Need to draft RTI application for PM Awas Yojana status",
        "Consumer forum complaint for defective washing machine",
        "Boss salary nahi de raha hai pichle 2 mahine se",
        "Draft legal notice to builder for possession delay",
        "मेरे मकान मालिक ने किराया नहीं दिया",
        "Divorce case — wife has left with children, need custody advice",
        "Anticipatory bail application needed — police may arrest tomorrow",
        "Writ petition for violation of fundamental rights Article 21",
        "PF not credited by employer for last 6 months",
    ]

    base_state = {
        "user_query": "", "language": "en", "session_id": "test",
        "user_id": "test", "conversation_history": [], "legal_domain": "general",
        "needs_document": False, "needs_research": True, "needs_referral": False,
        "is_emergency": False, "emergency_type": "", "research_results": [],
        "kanoon_cases": [], "ipc_sections": [], "case_analysis": {},
        "case_strength": "medium", "draft_document": "", "document_type": "",
        "document_hindi": "", "document_english": "", "confidence_score": 0.0,
        "hallucination_detected": False, "human_approval_needed": False,
        "verified_sources": [], "dlsa_offices": [], "lawyer_referrals": [],
        "final_response": "", "final_response_hindi": "", "sources": [],
        "next_steps": [], "error": None, "retry_count": 0,
    }

    print("=" * 60)
    print("ORCHESTRATOR AGENT TEST — LLM Based")
    print("=" * 60)

    for query in test_queries:
        base_state["user_query"] = query
        print(f"\nQuery: {query[:60]}")
        print("-" * 40)
        result = orchestrator_agent(base_state)
        print(f"Domain   : {result['legal_domain']}")
        print(f"Language : {result['language']}")
        print(f"Doc need : {result['needs_document']}")
        print(f"Referral : {result['needs_referral']}")
        issues = result.get('case_analysis', {}).get('key_issues', [])
        acts = result.get('case_analysis', {}).get('suggested_acts', [])
        if issues:
            print(f"Issues   : {', '.join(issues[:2])}")
        if acts:
            print(f"Acts     : {', '.join(acts[:2])}")
        print("=" * 60)