from backend.state import NyayMitraState
from backend.rag.hybrid_search import hybrid_rag
from backend.config import INDIAN_KANOON_API_KEY, INDIAN_KANOON_BASE_URL, GROQ_API_KEY
from langsmith import traceable
import httpx
import asyncio
import json


@traceable(name="IndianKanoon Search")
async def search_indian_kanoon(query: str, domain: str = None) -> list:
    if not INDIAN_KANOON_API_KEY:
        return []
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            params = {
                "formInput": query,
                "pagenum": 0,
            }
            headers = {"Authorization": f"Token {INDIAN_KANOON_API_KEY}"}
            response = await client.get(
                f"{INDIAN_KANOON_BASE_URL}/search/",
                params=params,
                headers=headers
            )
            if response.status_code == 200:
                data = response.json()
                docs = data.get("docs", [])
                results = []
                for doc in docs[:5]:
                    results.append({
                        "title": doc.get("title", ""),
                        "citation": doc.get("citation", ""),
                        "court": doc.get("docsource", ""),
                        "date": doc.get("publishdate", ""),
                        "snippet": doc.get("headline", ""),
                        "url": f"https://indiankanoon.org/doc/{doc.get('tid', '')}",
                        "tid": doc.get("tid", ""),
                        "source": "Indian Kanoon (Discovery only — verify before citing)"
                    })
                return results
    except Exception as e:
        print(f"IndianKanoon API error: {e}")
    return []


@traceable(name="Research Agent")
def research_agent(state: NyayMitraState) -> dict:
    query = state.get("user_query", "")
    domain = state.get("legal_domain", "general")

    # Hybrid RAG search
    rag_results = hybrid_rag.search(query, domain)

    # Indian Kanoon search (async call)
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            kanoon_results = []
        else:
            kanoon_results = loop.run_until_complete(
                search_indian_kanoon(query, domain)
            )
    except Exception:
        kanoon_results = []

    # IPC/BNS sections from RAG
    ipc_query = f"IPC BNS section {domain} law"
    ipc_results = hybrid_rag.search(ipc_query, domain)
    ipc_sections = [r for r in ipc_results if "section" in r.get("content", "").lower()][:3]

    return {
        "research_results": rag_results,
        "kanoon_cases": kanoon_results,
        "ipc_sections": ipc_sections,
    }
