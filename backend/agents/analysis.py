"""
Analysis Agent - NyayMitra 2.0

Structured legal analysis for a lawyer. No case-strength verdicts or outcome
predictions (CLAUDE.md): instead it surfaces missing facts, uncertainties and
questions for the lawyer. The LLM returns JSON (AnalysisLLMOutput) which is
rendered to the response text here; provisions cited that are not in the
retrieved context are flagged by the citation guard.
"""
import logging

from langsmith import traceable
from openai import OpenAI

from backend.agents.schemas import AnalysisDetail, AnalysisLLMOutput, AnalysisOutput, to_state
from backend.llm import chat_json
from backend.state import NyayMitraState
from backend.utils.citations import find_unverified_citations, research_context

logger = logging.getLogger("nyaymitra.analysis")

SYSTEM_PROMPT = """You are NyayMitra, a legal research assistant for Indian law firms. Your reader is a lawyer.

Return ONLY JSON with these keys (lists of short strings unless noted):
{
  "summary": "<2-4 sentence neutral summary of the legal issue (string)>",
  "applicable_law": ["<provision / Act and why it may be relevant>"],
  "action_plan": ["<concrete procedural step, in order>"],
  "evidence_needed": ["<document or evidence to collect>"],
  "missing_facts": ["<fact not stated in the query that materially affects the analysis>"],
  "uncertainties": ["<legal or factual point that is unclear or contested>"],
  "questions_for_lawyer": ["<question the lawyer should resolve or ask the client>"],
  "timeline": "<realistic procedural timeline (string)>",
  "costs": "<cost notes incl. free legal aid eligibility if relevant (string)>"
}

Rules:
- NEVER give a verdict on case strength (strong/weak/likely to win/lose) or predict outcomes.
  Express doubt through missing_facts, uncertainties and questions_for_lawyer.
- Cite statutory provisions ONLY if they appear in the LEGAL CONTEXT. Never cite case law
  that is not listed there. If another provision might apply, put it in uncertainties as
  something to verify, without asserting it.
- Write all values in the user's language (Hindi in Devanagari for "hi"; English for "en" and "hinglish").
- Be concise and practical."""

HEADINGS = {
    "en": {
        "summary": "Summary", "applicable_law": "Applicable law (verify before citing)",
        "action_plan": "Recommended steps", "evidence_needed": "Evidence to collect",
        "missing_facts": "Missing facts", "uncertainties": "Uncertainties",
        "questions_for_lawyer": "Questions for the lawyer", "timeline": "Timeline",
        "costs": "Costs & legal aid",
    },
    "hi": {
        "summary": "सारांश", "applicable_law": "लागू कानून (उद्धृत करने से पहले सत्यापित करें)",
        "action_plan": "सुझाए गए कदम", "evidence_needed": "आवश्यक साक्ष्य",
        "missing_facts": "अनुपलब्ध तथ्य", "uncertainties": "अनिश्चितताएँ",
        "questions_for_lawyer": "वकील के लिए प्रश्न", "timeline": "समय-सीमा",
        "costs": "लागत और कानूनी सहायता",
    },
}

_LIST_FIELDS = ("applicable_law", "action_plan", "evidence_needed",
                "missing_facts", "uncertainties", "questions_for_lawyer")


def render(result: AnalysisLLMOutput, language: str) -> str:
    """Deterministic markdown rendering of the structured analysis."""
    h = HEADINGS["hi" if language == "hi" else "en"]
    parts = [f"**{h['summary']}**\n{result.summary}"]
    for field in _LIST_FIELDS:
        items = getattr(result, field)
        if items:
            numbered = field == "action_plan"
            lines = [f"{i}. {x}" if numbered else f"- {x}" for i, x in enumerate(items, 1)]
            parts.append(f"**{h[field]}**\n" + "\n".join(lines))
    for field in ("timeline", "costs"):
        value = getattr(result, field)
        if value:
            parts.append(f"**{h[field]}**\n{value}")
    return "\n\n".join(parts)


def _build_prompt(state: NyayMitraState) -> str:
    query = state.get("user_query", "")
    domain = state.get("legal_domain", "general")
    language = state.get("language", "en")
    research_results = state.get("research_results", [])
    kanoon_cases = state.get("kanoon_cases", [])

    context_parts = []
    if kanoon_cases:
        context_parts.append("Relevant Cases Found (discovery only):")
        for case in kanoon_cases[:3]:
            context_parts.append(f"- {case.get('title', '')} ({case.get('court', '')})")
            context_parts.append(f"  {case.get('snippet', '')}")
    if research_results:
        context_parts.append("\nRelevant Legal Information:")
        for result in research_results[:3]:
            context_parts.append(f"- {result.get('content', '')[:400]}")
    mcp_sections = [s for s in state.get("ipc_sections", []) if s.get("source") == "mcp"]
    if mcp_sections:
        context_parts.append("\nIPC -> BNS references (quick reference, verify official text):")
        for s in mcp_sections[:5]:
            context_parts.append(f"- {s.get('content', '')[:250]}")
    context = "\n".join(context_parts) if context_parts else "(no matching legal context found)"

    classification = state.get("classification") or {}
    key_issues = classification.get("key_issues") or []
    suggested_acts = classification.get("suggested_acts") or []
    issues_block = ""
    if key_issues or suggested_acts:
        issues_block = (
            f"Key issues identified: {'; '.join(key_issues) or 'none'}\n"
            f"Possibly relevant Acts (unverified, from classifier): {'; '.join(suggested_acts) or 'none'}\n"
        )

    return (
        f"Legal Domain: {domain}\nUser Query: {query}\nLanguage: {language}\n{issues_block}\n"
        f"LEGAL CONTEXT:\n{context}"
    )


@traceable(name="Analysis Agent")
def analysis_agent(state: NyayMitraState) -> dict:
    domain = state.get("legal_domain", "general")
    language = state.get("language", "en")
    research_results = state.get("research_results", [])
    kanoon_cases = state.get("kanoon_cases", [])

    try:
        data, usage = chat_json(OpenAI, SYSTEM_PROMPT, _build_prompt(state),
                                max_tokens=1800, temperature=0.3)
        result = AnalysisLLMOutput.model_validate(data)
        text = render(result, language)
        unverified = find_unverified_citations([text], research_context(state))

        return to_state(AnalysisOutput(
            case_analysis=AnalysisDetail(
                analysis=text,
                domain=domain,
                cases_found=len(kanoon_cases),
                rag_results=len(research_results),
                **result.model_dump(),
            ),
            next_steps=result.action_plan[:5],
            final_response=text,
            analysis_unverified_citations=unverified,
            tokens_used=usage.total,
        ))

    except Exception as e:
        logger.exception("Analysis failed")
        fallback = (
            f"Automated analysis is unavailable right now for this {domain} matter.\n\n"
            "1. Document everything — dates, incidents, communications\n"
            "2. Gather all relevant documents\n"
            "3. Consult a lawyer or DLSA for free legal aid\n"
            "4. Legal Aid Helpline: 15100"
        )
        return to_state(AnalysisOutput(
            case_analysis=AnalysisDetail(analysis=fallback, error=type(e).__name__),
            next_steps=["Document everything", "Consult a lawyer", "Contact DLSA: 15100"],
            final_response=fallback,
        ))
