"""
Citation guard shared by analysis and drafting (CLAUDE.md: never invent citations).

Finds provision numbers (Section / s. / धारा / Article / अनुच्छेद N) in generated
text that never appear in the retrieved legal context. It compares numbers
only, not which Act they belong to, so it catches invented provisions but not
a real number attributed to the wrong Act.
"""
import re
from typing import Iterable, List, Set

_CITE_RE = re.compile(
    r"(?:section|sec\.|s\.|धारा|article|अनुच्छेद)\s*(\d+[A-Z]?)(?:\(\d+\))?",
    re.IGNORECASE,
)


def cited_numbers(text: str) -> Set[str]:
    return {m.upper() for m in _CITE_RE.findall(text or "")}


def find_unverified_citations(texts: Iterable[str], allowed_context: str) -> List[str]:
    """Provision numbers cited in `texts` that never appear in `allowed_context`."""
    allowed = cited_numbers(allowed_context)
    cited: Set[str] = set()
    for t in texts:
        cited |= cited_numbers(t)
    return sorted(cited - allowed)


def research_context(state: dict) -> str:
    """Text a generated answer may cite from: corpus results, statutory refs, case snippets."""
    parts = [r.get("content", "") for r in state.get("research_results", [])]
    parts += [r.get("content", "") for r in state.get("ipc_sections", [])]
    parts += [c.get("snippet", "") for c in state.get("kanoon_cases", [])]
    return "\n".join(p for p in parts if p)
