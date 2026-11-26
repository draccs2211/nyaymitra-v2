"""
Load the legal corpus from data/legal_corpus/ into ChromaDB + BM25.

    python -m backend.data_loader                 # load everything (idempotent)
    python -m backend.data_loader --dry-run       # validate + count, write nothing
    python -m backend.data_loader --no-prune      # keep docs that are no longer in the files

Corpus files (format details: data/legal_corpus/README.md):
- *.jsonl  one document per line: {"content": ..., "metadata": {...}}
- *.md     official text of an Act: header block + "## Section N — Title" sections,
           split per section (long sections into overlapping parts)

Every document's metadata is validated against backend.rag.metadata.DocMetadata.
Document IDs are deterministic ('<doc_type>:<sha256(content)[:16]>'), so re-running
is idempotent and editing a document's text replaces it.
"""
import argparse
import hashlib
import io
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import ValidationError

from backend.rag.metadata import DocMetadata

CORPUS_DIR = Path(__file__).resolve().parents[1] / "data" / "legal_corpus"
# ~ what the embedding model reads (128 tokens); longer sections are split with overlap
MAX_CHUNK_CHARS = 700

_SECTION_RE = re.compile(r"^##\s+(?:Section|Sec\.?|धारा)\s+(\d+[A-Z]?)\b\s*[—:-]?\s*(.*)$", re.IGNORECASE)


class CorpusError(Exception):
    """A corpus file is malformed or a document fails metadata validation."""


def make_canonical_id(doc: dict) -> str:
    """
    Deterministic ID: '<doc_type>:<sha256(normalized content)[:16]>'.
    Same content -> same ID on every run, so loads are idempotent.
    Metadata is excluded so metadata fixes update in place via upsert.
    """
    doc_type = doc.get("metadata", {}).get("doc_type", "doc")
    normalized = re.sub(r"\s+", " ", doc["content"]).strip().lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{doc_type}:{digest}"


# ─── Readers ──────────────────────────────────────────────────────────────────

def read_jsonl(path: Path) -> List[dict]:
    """Bad lines become {"_error": ...} entries so every problem in a file is reported at once."""
    docs = []
    with io.open(path, encoding="utf-8-sig") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                docs.append({"_error": f"{path.name}:{n}: invalid JSON ({e.msg})"})
                continue
            if not isinstance(item, dict) or not str(item.get("content", "")).strip():
                docs.append({"_error": f"{path.name}:{n}: each line needs a non-empty 'content'"})
                continue
            docs.append({"content": item["content"].strip(), "metadata": item.get("metadata") or {},
                         "_where": f"{path.name}:{n}"})
    return docs


_CLAUSE_SPLIT = re.compile(r"(?<=[;:.])\s+(?=\(|[A-Z0-9“\"])")


def _units(text: str, limit: int) -> List[str]:
    """Break text into pieces <= limit: blank-line paragraphs, then lines, then
    clause/sentence boundaries, then (last resort) word boundaries."""
    out: List[str] = []
    for para in (p.strip() for p in re.split(r"\n\s*\n", text)):
        if not para:
            continue
        if len(para) <= limit:
            out.append(para)
            continue
        for line in (l.strip() for l in para.split("\n")):
            if not line:
                continue
            if len(line) <= limit:
                out.append(line)
                continue
            for clause in (c.strip() for c in _CLAUSE_SPLIT.split(line)):
                while len(clause) > limit:                      # no boundary: cut at a space
                    cut = clause.rfind(" ", 0, limit)
                    cut = cut if cut > limit // 2 else limit
                    out.append(clause[:cut].strip())
                    clause = clause[cut:].strip()
                if clause:
                    out.append(clause)
    return out


def _split_long(text: str, limit: int = MAX_CHUNK_CHARS) -> List[str]:
    """Pack units into chunks <= limit, carrying the previous unit as overlap when it fits."""
    units = _units(text, limit)
    if not units:
        return [text]
    chunks, current = [], []
    for u in units:
        if current and len("\n".join(current + [u])) > limit:
            chunks.append("\n".join(current))
            last = current[-1]
            current = [last] if len(last) + 1 + len(u) <= limit else []   # overlap if it fits
        current.append(u)
    if current:
        chunks.append("\n".join(current))
    return chunks


def read_act_markdown(path: Path) -> List[dict]:
    """Official Act text: '---' header block of `key: value`, then '## Section N — Title' blocks."""
    text = io.open(path, encoding="utf-8-sig").read().replace("\r\n", "\n")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    if not m:
        raise CorpusError(f"{path.name}: must start with a '---' header block")
    header: Dict[str, str] = {}
    for line in m.group(1).splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            key, sep, value = line.partition(":")
            if not sep:
                raise CorpusError(f"{path.name}: header line without ':': {line!r}")
            header[key.strip()] = value.strip()
    act = header.get("act", "")

    sections: List[Tuple[str, str, List[str]]] = []
    for line in m.group(2).splitlines():
        hm = _SECTION_RE.match(line.strip())
        if hm:
            sections.append((hm.group(1).upper(), hm.group(2).strip(), []))
        elif sections:
            sections[-1][2].append(line)
    if not sections:
        raise CorpusError(f"{path.name}: no '## Section N — Title' headings found")

    docs = []
    for number, title, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body:
            raise CorpusError(f"{path.name}: Section {number} has no text")
        heading = f"{act} — Section {number}" + (f": {title}" if title else "")
        parts = _split_long(body)
        for i, part in enumerate(parts, 1):
            label = heading + (f" (part {i}/{len(parts)})" if len(parts) > 1 else "")
            docs.append({
                "content": f"{label}\n{part}",
                "metadata": {
                    "doc_type": "statute_section",
                    "domain": header.get("domain", "general"),
                    "act": act,
                    "section": number,
                    "date": header.get("date", ""),
                    "source_url": header.get("source_url", ""),
                    "source_name": header.get("source_name", "Official text"),
                    "jurisdiction": header.get("jurisdiction", "IN"),
                    "topic": title,
                },
                "_where": f"{path.name}: Section {number}",
            })
    return docs


READERS = {".jsonl": read_jsonl, ".md": read_act_markdown}


def build_documents(corpus_dir: Path = CORPUS_DIR) -> List[dict]:
    """Read + validate every corpus file. Raises CorpusError listing all problems."""
    corpus_dir = Path(corpus_dir)
    files = sorted(p for p in corpus_dir.iterdir()
                   if p.suffix in READERS and p.name.lower() != "readme.md")
    if not files:
        raise CorpusError(f"no corpus files (*.jsonl, *.md) in {corpus_dir}")

    docs, errors, seen = [], [], {}
    for path in files:
        try:
            raw = READERS[path.suffix](path)
        except CorpusError as e:
            errors.append(str(e))
            continue
        for d in raw:
            if "_error" in d:
                errors.append(d["_error"])
                continue
            where = d.pop("_where")
            try:
                DocMetadata.model_validate(d["metadata"])
            except ValidationError as e:
                errors.append(f"{where}: {e.errors()[0]['msg']}")
                continue
            d["id"] = make_canonical_id(d)
            if d["id"] in seen:                       # identical text in two places
                errors.append(f"{where}: duplicate of {seen[d['id']]}")
                continue
            seen[d["id"]] = where
            d["_file"] = path.name
            docs.append(d)
    if errors:
        raise CorpusError("corpus validation failed:\n  " + "\n  ".join(errors))
    return docs


def load_data(prune: bool = True, dry_run: bool = False, corpus_dir: Path = CORPUS_DIR) -> Dict:
    """
    Idempotent load. With prune=True, removes indexed docs that are no longer in the
    corpus files (e.g. edited or deleted documents).
    """
    docs = build_documents(corpus_dir)
    per_file: Dict[str, int] = {}
    for d in docs:
        name = d.pop("_file")
        per_file[name] = per_file.get(name, 0) + 1
    counts = {"documents": len(docs), "per_file": per_file}
    print(f"Corpus: {len(docs)} documents validated")
    for name, n in per_file.items():
        print(f"  {name}: {n}")
    if dry_run:
        print("Dry run: nothing written")
        return counts

    from backend.rag.hybrid_search import hybrid_rag  # lazy: heavy imports

    hybrid_rag.add_documents(docs)
    if prune:
        counts["removed"] = hybrid_rag.delete_ids_not_in([d["id"] for d in docs])
        print(f"Removed {counts['removed']} stale docs")
    stats = hybrid_rag.get_stats()
    print(f"Stats: {stats}")
    if not stats["ids_in_sync"]:
        raise RuntimeError("ChromaDB and BM25 IDs are out of sync")
    return counts


def main():
    parser = argparse.ArgumentParser(description="Load data/legal_corpus into the search indexes")
    parser.add_argument("--dry-run", action="store_true", help="validate and count only")
    parser.add_argument("--no-prune", action="store_true", help="keep docs no longer in the files")
    parser.add_argument("--dir", default=str(CORPUS_DIR), help="corpus directory")
    args = parser.parse_args()
    try:
        load_data(prune=not args.no_prune, dry_run=args.dry_run, corpus_dir=Path(args.dir))
    except CorpusError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)
    print("Data loading complete.")


if __name__ == "__main__":
    main()
