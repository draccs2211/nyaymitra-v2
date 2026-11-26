"""
Corpus loader tests (Phase 2): JSONL + official Act markdown, per-section
chunking, validation errors with locations. Uses tmp dirs; no index writes.
Run: pytest tests/test_corpus_loader.py -v
"""
import json

import pytest

from backend.data_loader import MAX_CHUNK_CHARS, CorpusError, build_documents, load_data

ACT_HEADER = """---
act: The Example Act, 2024
domain: criminal
jurisdiction: IN
source_url: https://www.indiacode.nic.in/handle/123/example
source_name: India Code (official text)
date: 2024-01-01
---
"""


def write(tmp_path, name, text):
    (tmp_path / name).write_text(text, encoding="utf-8")


def guidance(content, **meta):
    return json.dumps({"content": content, "metadata": {"doc_type": "legal_guidance", "domain": "general", **meta}})


class TestActMarkdown:
    def test_sections_become_documents(self, tmp_path):
        write(tmp_path, "example.md", ACT_HEADER + """
Preamble text is ignored.

## Section 316 — Criminal breach of trust
(1) Whoever, being entrusted with property ...

## Sec. 318A: Cheating
Text of 318A.

## धारा 2 — परिभाषाएँ
हिंदी पाठ।
""")
        docs = build_documents(tmp_path)
        assert [d["metadata"]["section"] for d in docs] == ["316", "318A", "2"]
        first = docs[0]
        assert first["content"].startswith("The Example Act, 2024 — Section 316: Criminal breach of trust\n")
        meta = first["metadata"]
        assert meta["doc_type"] == "statute_section" and meta["source_url"].startswith("https://")
        assert meta["topic"] == "Criminal breach of trust" and meta["date"] == "2024-01-01"
        assert all(d["id"].startswith("statute_section:") for d in docs)

    def test_long_section_split_with_overlap(self, tmp_path):
        paras = [f"Clause {i}. " + ("word " * 60) for i in range(12)]
        write(tmp_path, "long.md", ACT_HEADER + "## Section 5 — Long\n" + "\n\n".join(paras))
        docs = build_documents(tmp_path)
        assert len(docs) > 1
        assert all(len(d["content"]) <= MAX_CHUNK_CHARS + 120 for d in docs)     # + heading line
        assert "(part 1/" in docs[0]["content"]
        # consecutive parts share a paragraph (overlap) and every clause is covered
        body = "\n".join(d["content"] for d in docs)
        assert all(f"Clause {i}." in body for i in range(12))
        last_unit_of_first = docs[0]["content"].split("\n")[-1]
        assert last_unit_of_first in docs[1]["content"]

    def test_single_newline_clauses_and_long_lines_are_split(self, tmp_path):
        """Converted Acts put each clause on its own line (no blank lines); a clause may be huge."""
        clauses = "\n".join(f"({i}) " + "term " * 40 + ";" for i in range(30))
        giant = "Definitions: " + " ".join(f"({i}) item {i} means thing {i};" for i in range(200))
        write(tmp_path, "act.md", ACT_HEADER + "## Section 2 — Defs\n" + clauses + "\n" + giant)
        docs = build_documents(tmp_path)
        assert len(docs) > 10
        assert all(len(d["content"]) <= MAX_CHUNK_CHARS + 120 for d in docs)
        body = " ".join(d["content"] for d in docs)
        assert "(29) term" in body and "item 199 means thing 199" in body

    @pytest.mark.parametrize("text,error", [
        ("## Section 1 — X\ntext", "header block"),
        (ACT_HEADER + "no section headings here", "no '## Section"),
        (ACT_HEADER + "## Section 1 — Empty\n\n## Section 2 — X\ntext", "Section 1 has no text"),
        (ACT_HEADER.replace("source_url: https://www.indiacode.nic.in/handle/123/example\n", "")
         + "## Section 1 — X\ntext", "source_url"),
        (ACT_HEADER.replace("domain: criminal", "domain: tax") + "## Section 1 — X\ntext", "domain"),
    ])
    def test_invalid_markdown(self, tmp_path, text, error):
        write(tmp_path, "bad.md", text)
        with pytest.raises(CorpusError, match=error):
            build_documents(tmp_path)


class TestJsonl:
    def test_reads_and_ids_stable(self, tmp_path):
        write(tmp_path, "g.jsonl", guidance("Lok Adalat settles disputes.") + "\n\n" + guidance("FIR basics."))
        a, b = build_documents(tmp_path), build_documents(tmp_path)
        assert [d["id"] for d in a] == [d["id"] for d in b] and len(a) == 2

    def test_errors_report_file_and_line(self, tmp_path):
        write(tmp_path, "g.jsonl", guidance("ok") + "\n{not json\n" + guidance("x", doc_type="blog"))
        with pytest.raises(CorpusError) as e:
            build_documents(tmp_path)
        assert "g.jsonl:2" in str(e.value) and "g.jsonl:3" in str(e.value)

    def test_duplicate_text_rejected(self, tmp_path):
        write(tmp_path, "a.jsonl", guidance("Same text."))
        write(tmp_path, "b.jsonl", guidance("Same text."))
        with pytest.raises(CorpusError, match="duplicate of a.jsonl:1"):
            build_documents(tmp_path)

    def test_readme_ignored_and_empty_dir_rejected(self, tmp_path):
        write(tmp_path, "README.md", "# not a corpus file")
        with pytest.raises(CorpusError, match="no corpus files"):
            build_documents(tmp_path)


class TestDryRun:
    def test_dry_run_counts_without_indexing(self, tmp_path, monkeypatch):
        import sys
        write(tmp_path, "g.jsonl", guidance("a") + "\n" + guidance("b"))
        write(tmp_path, "act.md", ACT_HEADER + "## Section 1 — X\ntext")
        out = load_data(dry_run=True, corpus_dir=tmp_path)
        assert out == {"documents": 3, "per_file": {"act.md": 1, "g.jsonl": 2}}


class TestRealCorpus:
    def test_shipped_corpus_validates(self):
        docs = build_documents()
        assert len(docs) >= 26
