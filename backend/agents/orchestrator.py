"""
Orchestrator Agent - NyayMitra 2.0
Uses OpenAI GPT-4o-mini to intelligently classify legal queries.
No hardcoded keywords — pure LLM-based classification.
"""
from backend.state import NyayMitraState, initial_state
from backend.llm import chat_json
from backend.agents.schemas import (
    ClassificationLLMOutput, ClassificationMeta, OrchestratorOutput, to_state
)
from langsmith import traceable
import logging

logger = logging.getLogger("nyaymitra.orchestrator")
from openai import OpenAI
from pydantic import ValidationError
import json


CLASSIFICATION_PROMPT = """You are an expert Indian legal classifier for a law firm AI tool.

Analyze the given legal query and return a JSON response with the following fields:

{
    "legal_domain": "<one of: criminal, civil, family, property, consumer, rti, labor, constitutional, general>",
    "requested_actions": ["<every action the query asks for — one or more of: research, drafting, referral>"],
    "language": "<one of: en, hi, hinglish>",
    "search_query_en": "<the legal issue restated as a short English search query>",
    "urgency": "<one of: critical, high, medium, low>",
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

Requested actions (a query can ask for several — include ALL that apply):
- research: legal research / analysis of the situation, applicable law, options
- drafting: a document should be drafted — legal notice, RTI application, complaint, petition, vakalatnama, affidavit etc.
- referral: referral to legal aid, DLSA, Lok Adalat, a lawyer or helpline is asked for or clearly needed
Example: "Draft a legal notice to my landlord and tell me where to get free legal aid" -> ["research", "drafting", "referral"]
Example: "Only give me the DLSA contact for Lucknow" -> ["referral"]
If unsure, include "research".

Search query (search_query_en):
- Restate the user's legal issue as a concise English search query using standard Indian legal terms.
- Translate Hindi / Hinglish faithfully (e.g. "patni pati se har mahine kharcha chahti hai" -> "wife claiming monthly maintenance from husband").
- Do NOT add facts, section numbers or Acts that the query does not mention.

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
        data, usage = chat_json(OpenAI, CLASSIFICATION_PROMPT, user_message,
                                max_tokens=500, temperature=0.1)
        result = ClassificationLLMOutput.model_validate(data)

        logger.info(f"Classified: domain={result.legal_domain} | Lang: {result.language} | Urgency: {result.urgency} | Actions: {result.requested_actions}")
        logger.debug("Classification reasoning: %s", result.reasoning)

        return to_state(OrchestratorOutput(
            legal_domain=result.legal_domain,
            requested_actions=result.requested_actions,
            language=result.language,
            search_query_en=result.search_query_en,
            tokens_used=usage.total,
            classification=ClassificationMeta(
                urgency=result.urgency,
                key_issues=result.key_issues,
                suggested_acts=result.suggested_acts,
                reasoning=result.reasoning,
            ),
        ))

    except (json.JSONDecodeError, ValidationError) as e:
        logger.warning(f"Invalid LLM output: {e}")
        return _fallback_classification(query)

    except Exception as e:
        logger.warning(f"Classification failed: {e}")
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

    logger.warning("Using fallback classification")

    return to_state(OrchestratorOutput(
        legal_domain="general",
        requested_actions=["research"],
        language=language,
        classification=ClassificationMeta(
            urgency="low",
            reasoning="Fallback classification used",
        ),
    ))


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

    base_state = initial_state()

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
        print(f"Actions  : {result['requested_actions']}")
        issues = result['classification']['key_issues']
        acts = result['classification']['suggested_acts']
        if issues:
            print(f"Issues   : {', '.join(issues[:2])}")
        if acts:
            print(f"Acts     : {', '.join(acts[:2])}")
        print("=" * 60)