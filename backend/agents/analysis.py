from backend.state import NyayMitraState
from backend.config import OPENAI_API_KEY
from langsmith import traceable
from openai import OpenAI


SYSTEM_PROMPT = """You are NyayMitra, an expert Indian legal advisor for law firms.
Analyze the legal situation and provide:
1. Case strength (weak/medium/strong) with reasoning
2. Key legal provisions applicable (IPC/BNS sections, relevant Acts)
3. Step by step action plan
4. Evidence required
5. Realistic timeline
6. Cost estimate (free legal aid if available)
7. Next steps

Be honest and realistic. If case is weak, say so clearly.
Respond in the same language as the query (Hindi/English/Hinglish).
Keep response concise and practical."""


@traceable(name="Analysis Agent")
def analysis_agent(state: NyayMitraState) -> dict:
    query = state.get("user_query", "")
    domain = state.get("legal_domain", "general")
    research_results = state.get("research_results", [])
    kanoon_cases = state.get("kanoon_cases", [])
    language = state.get("language", "en")

    # Build context from research
    context_parts = []

    if kanoon_cases:
        context_parts.append("Relevant Cases Found:")
        for case in kanoon_cases[:3]:
            context_parts.append(f"- {case.get('title', '')} ({case.get('court', '')})")
            context_parts.append(f"  {case.get('snippet', '')}")

    if research_results:
        context_parts.append("\nRelevant Legal Information:")
        for result in research_results[:3]:
            context_parts.append(f"- {result.get('content', '')[:200]}")

    context = "\n".join(context_parts) if context_parts else "No specific cases found — providing general legal guidance."

    user_prompt = f"""
Legal Domain: {domain}
User Query: {query}
Language preference: {language}

Context from legal database:
{context}

Please provide a comprehensive legal analysis.
"""

    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=1500,
            temperature=0.3
        )
        analysis_text = response.choices[0].message.content

        # Determine case strength
        case_strength = "medium"
        if any(word in analysis_text.lower() for word in ["strong case", "strong", "mazboot"]):
            case_strength = "strong"
        elif any(word in analysis_text.lower() for word in ["weak case", "weak", "kamzor", "difficult"]):
            case_strength = "weak"

        # Extract next steps
        next_steps = []
        for line in analysis_text.split("\n"):
            if any(char in line for char in ["1.", "2.", "3.", "•", "-"]):
                clean = line.strip().lstrip("0123456789.-•").strip()
                if clean and len(clean) > 10:
                    next_steps.append(clean)

        return {
            "case_analysis": {
                "analysis": analysis_text,
                "domain": domain,
                "cases_found": len(kanoon_cases),
                "rag_results": len(research_results)
            },
            "case_strength": case_strength,
            "next_steps": next_steps[:5],
            "final_response": analysis_text,
        }

    except Exception as e:
        fallback = f"""Based on your query about {domain} law:

1. Document everything — dates, incidents, communications
2. Gather all relevant documents
3. Consult a lawyer or DLSA for free legal aid
4. Contact: Legal Aid Helpline: 15100

Error: {str(e)}"""

        return {
            "case_analysis": {"analysis": fallback, "error": str(e)},
            "case_strength": "medium",
            "next_steps": ["Document everything", "Consult a lawyer", "Contact DLSA: 15100"],
            "final_response": fallback,
        }