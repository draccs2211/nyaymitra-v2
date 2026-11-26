"""Exact IPC -> BNS lookup backed by data/reference/ipc_bns_mapping.json and the BNS text in
data/legal_corpus/bns_2023.md. No keyword matching, no hand-typed punishments: the provision
text returned is the corpus text itself, so it can only be as wrong as the corpus (which is
flagged for lawyer review)."""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

_ROOT = Path(__file__).resolve().parents[2]
MAPPING_PATH = _ROOT / "data" / "reference" / "ipc_bns_mapping.json"
BNS_PATH = _ROOT / "data" / "legal_corpus" / "bns_2023.md"


@lru_cache(maxsize=1)
def _mapping() -> dict:
    with open(MAPPING_PATH, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def _bns_sections() -> Dict[str, dict]:
    text = BNS_PATH.read_text(encoding="utf-8")
    out = {}
    for m in re.finditer(r"^## Section (\S+) — ([^\n]*)\n(.*?)(?=^## Section |\Z)", text, re.M | re.S):
        out[m.group(1)] = {"title": m.group(2).strip(), "text": m.group(3).strip()}
    return out


def _punishment_excerpt(body: str, limit: int = 3, width: int = 260) -> str:
    flat = " ".join(body.split())
    hits = [h.strip() for h in re.findall(r"[^.;]*shall be punished[^.;]*", flat)]
    hits = [h if len(h) <= width else h[:width].rsplit(" ", 1)[0] + "..." for h in hits[:limit]]
    return " | ".join(hits)


def normalise(section: str) -> str:
    return re.sub(r"\s+", "", (section or "")).upper()


def lookup_ipc(section: str) -> Optional[dict]:
    """Return None if the IPC section isn't in the mapping at all."""
    s = normalise(section)
    m = _mapping()
    if s in m.get("ipc_deleted_or_not_carried", []):
        return {"ipc": s, "carried_into_bns": False, "bns": []}
    bns_nums: List[str] = m["ipc_to_bns"].get(s) or []
    if not bns_nums:
        return None
    secs = _bns_sections()
    bns = []
    for n in bns_nums:
        sec = secs.get(n)
        bns.append({
            "section": n,
            "title": sec["title"] if sec else None,
            "punishment_text": _punishment_excerpt(sec["text"]) if sec else "",
        })
    return {"ipc": s, "carried_into_bns": True, "bns": bns}
