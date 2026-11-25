"""
Research Agent - NyayMitra 2.0

1. Hybrid RAG over the curated corpus (Chroma + BM25 + RRF + rerank)
2. In parallel via MCP (timeout / retry / fallback):
   - indiankanoon.search_case_laws  -> case-law DISCOVERY only (verify before citing)
   - legal_reference.get_ipc_section -> IPC -> BNS mapping for IPC sections found
     in corpus metadata or cited (IPC-qualified) in the query
3. source_status records what each source returned, so the lawyer knows
   when e.g. no case law was searched.
"""
import asyncio
import re
from typing import Dict, List, Tuple

from langsmith import traceable
import logging

logger = logging.getLogger("nyaymitra.research")

from backend.agents.schemas import KanoonCase, ResearchOutput, RetrievedDoc, to_state
from backend.mcp_client import MCPCallError, call_tool, run_sync
from backend.rag.hybrid_search import hybrid_rag
from backend.state import NyayMitraState

MAX_KANOON_RESULTS = 5
MAX_SECTION_LOOKUPS = 5

# IPC-qualified section references in the query: "Section 406 IPC", "s. 498A of the IPC",
# "420 IPC", "IPC 302", "धारा 376 IPC". Bare numbers are ignored (amounts, other Acts).
_SEC = r"(\d{1,3}[A-Z]?)"
_IPC = r"(?:IPC|I\.P\.C\.?|Indian Penal Code)"
_QUERY_IPC_PATTERNS = [
    re.compile(rf"(?:section|sec\.?|s\.|धारा)\s*{_SEC}\s*(?:of\s+(?:the\s+)?)?{_IPC}", re.IGNORECASE),
    re.compile(rf"\b{_SEC}\s+{_IPC}\b", re.IGNORECASE),
    re.compile(rf"\b{_IPC}\s+(?:section\s+|sec\.?\s*)?{_SEC}\b", re.IGNORECASE),
]


def ipc_sections_to_lookup(query: str, rag_results: List[dict]) -> List[str]:
    """IPC section numbers from corpus metadata + IPC-qualified query mentions (deduped, capped)."""
    found: List[str] = []

    def add(n: str):
        n = n.strip().upper()
        if n and n not in found:
            found.append(n)

    for pattern in _QUERY_IPC_PATTERNS:
        for m in pattern.findall(query or ""):
            add(m)
    for r in rag_results:
        meta = r.get("metadata") or {}
        if "Indian Penal Code" in meta.get("act", "") and meta.get("section"):
            add(meta["section"])
    return found[:MAX_SECTION_LOOKUPS]


async def _search_case_law(query: str) -> Tuple[List[dict], str]:
    try:
        data = await call_tool("indiankanoon", "search_case_laws",
                               {"query": query, "max_results": MAX_KANOON_RESULTS})
    except MCPCallError as e:
        logger.warning(f"Indian Kanoon unavailable: {e}")
        return [], "unavailable"
    if data.get("status") != "success":
        return [], "not_configured" if data.get("code") == "not_configured" else "error"
    cases = [
        {k: r.get(k, "") for k in ("title", "citation", "court", "date", "snippet", "url", "tid")}
        for r in data.get("results", [])
    ]
    return cases, "ok"


async def _lookup_ipc(section: str) -> Tuple[dict, str]:
    try:
        data = await call_tool("legal_reference", "get_ipc_section", {"section_number": section})
    except MCPCallError as e:
        logger.warning(f"legal_reference unavailable for {section}: {e}")
        return {}, "unavailable"
    if data.get("status") != "found":
        return {}, "ok"          # server answered; section just isn't in its quick reference
    content = (
        f"{data['ipc_section']} corresponds to {data['bns_equivalent']}: "
        f"{data.get('description', '')}. Punishment: {data.get('punishment', '')}. "
        f"{data.get('note', '')}"
    ).strip()
    return {
        "id": f"mcp:legal_reference:ipc:{section}",
        "content": content,
        "metadata": {
            "doc_type": "ipc_section",
            "act": "The Indian Penal Code, 1860",
            "section": section,
            "source_name": "legal_reference MCP quick reference (verify official text)",
            "source_url": data.get("official_source", ""),
        },
        "source": "mcp",
    }, "ok"


def _combine(statuses: List[str]) -> str:
    """Worst status wins: unavailable > error > ok; 'skipped' if nothing was called."""
    if not statuses:
        return "skipped"
    for s in ("unavailable", "error"):
        if s in statuses:
            return s
    return "ok"


async def _gather_mcp(query: str, sections: List[str]):
    return await asyncio.gather(
        _search_case_law(query),
        *[_lookup_ipc(s) for s in sections],
    )


@traceable(name="Research Agent")
def research_agent(state: NyayMitraState) -> dict:
    query = state.get("user_query", "")
    domain = state.get("legal_domain", "general")

    # Original query + orchestrator's English rewrite, fused (helps Hindi/Hinglish
    # against the English corpus; see backend/evaluation/retrieval_eval.py)
    rewrite = state.get("search_query_en", "")
    rag_results = hybrid_rag.search(query, domain, extra_queries=[rewrite] if rewrite else None,
                                    jurisdiction=state.get("jurisdiction") or "IN")
    sections = ipc_sections_to_lookup(query, rag_results)

    (kanoon_cases, kanoon_status), *ipc_lookups = run_sync(_gather_mcp(query, sections))
    mcp_sections = [doc for doc, _ in ipc_lookups if doc]

    # Statutory references: corpus IPC docs from this search + MCP IPC->BNS mappings
    corpus_sections = [
        r for r in rag_results if (r.get("metadata") or {}).get("doc_type") == "ipc_section"
    ]
    source_status: Dict[str, str] = {
        "indian_kanoon": kanoon_status,
        "legal_reference": _combine([s for _, s in ipc_lookups]),
    }
    logger.info(f"RAG={len(rag_results)} kanoon={len(kanoon_cases)} "
          f"ipc_lookups={sections} status={source_status}")

    return to_state(ResearchOutput(
        research_results=[RetrievedDoc(**r) for r in rag_results],
        kanoon_cases=[KanoonCase(**c) for c in kanoon_cases],
        ipc_sections=[RetrievedDoc(**r) for r in corpus_sections + mcp_sections],
        source_status=source_status,
    ))
