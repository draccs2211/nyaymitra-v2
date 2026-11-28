"""
Exact lookup of provisions the user cites ("BNS 318", "section 6 of the RTI Act").

The orchestrator (LLM) extracts {act, section} pairs with the Act's full official name;
this module resolves that name to an Act in the corpus and returns the section's
chunks in order. Nothing is guessed: an Act or section not in the corpus is reported
as unresolved so analysis can say it was not verified.
"""
import difflib
import re
from typing import Dict, List, Optional, Tuple

MAX_CITED_PROVISIONS = 5

_YEAR = re.compile(r"\b(1[89]\d\d|20\d\d)\b")
_PART = re.compile(r"\(part (\d+)/\d+\)")


def _norm(name: str) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", (name or "").lower())
    s = re.sub(r"^\s*the\s+", "", s)
    return " ".join(s.split())


def _split_year(name: str) -> Tuple[str, Optional[str]]:
    n = _norm(name)
    m = _YEAR.search(n)
    return (" ".join(_YEAR.sub(" ", n).split()), m.group(1) if m else None)


def resolve_act(name: str, corpus_acts: List[str], cutoff: float = 0.85) -> Optional[str]:
    """Corpus Act name for an LLM-supplied Act name, or None. Years must agree when both given."""
    if not name:
        return None
    target, year = _split_year(name)
    if not target:
        return None
    candidates: Dict[str, str] = {}
    for act in corpus_acts:
        base, act_year = _split_year(act)
        if year and act_year and year != act_year:
            continue
        if _norm(act) == _norm(name) or base == target:
            return act
        candidates[base] = act
    close = difflib.get_close_matches(target, list(candidates), n=1, cutoff=cutoff)
    return candidates[close[0]] if close else None


def normalize_section(section: str) -> str:
    s = re.sub(r"(?i)^\s*(section|sec\.?|s\.|धारा)\s*", "", str(section or "")).strip()
    return s.upper().replace(" ", "")


def section_chunks(act: str, section: str, docs: List[dict]) -> List[dict]:
    """Chunks of one section of one Act, in part order (docs: {'id','content','metadata'})."""
    hits = [d for d in docs
            if (d.get("metadata") or {}).get("act") == act
            and str((d.get("metadata") or {}).get("section", "")).upper() == section]

    def part(d):
        m = _PART.search(d.get("content", "").split("\n", 1)[0])
        return int(m.group(1)) if m else 0
    return sorted(hits, key=part)


def lookup_cited(provisions: List[dict], docs: List[dict]) -> Tuple[List[dict], List[str]]:
    """-> (chunks of every resolved provision, 'Act s.N' labels that are not in the corpus)."""
    corpus_acts = sorted({(d.get("metadata") or {}).get("act", "") for d in docs} - {""})
    found: List[dict] = []
    unresolved: List[str] = []
    seen = set()
    for p in provisions[:MAX_CITED_PROVISIONS]:
        section = normalize_section(p.get("section", ""))
        if not section:
            continue
        act = resolve_act(p.get("act", ""), corpus_acts)
        key = (act or p.get("act", ""), section)
        if key in seen:
            continue
        seen.add(key)
        chunks = section_chunks(act, section, docs) if act else []
        if chunks:
            found += [{**c, "source": "exact"} for c in chunks]
        else:
            unresolved.append(f"{p.get('act') or 'unknown Act'} s.{section}")
    return found, unresolved
