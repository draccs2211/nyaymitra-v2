"""
Corpus metadata tests (CLAUDE.md issue #3). No model load, no paid APIs.
Run: pytest tests/test_metadata.py -v
"""
import pytest
from pydantic import ValidationError

from backend.data_loader import build_documents

LEGAL_DATA = build_documents()      # the real corpus files in data/legal_corpus
from backend.rag.metadata import DocMetadata

REQUIRED_KEYS = {"act", "section", "court", "date", "source_url", "doc_type", "jurisdiction"}


class TestCorpusMetadata:
    @pytest.mark.parametrize("doc", LEGAL_DATA, ids=lambda d: d["content"][:30])
    def test_every_doc_validates(self, doc):
        DocMetadata.model_validate(doc["metadata"])

    def test_every_doc_has_identical_keys(self):
        keysets = {frozenset(DocMetadata.model_validate(d["metadata"]).to_chroma()) for d in LEGAL_DATA}
        assert len(keysets) == 1
        assert REQUIRED_KEYS <= set(next(iter(keysets)))

    def test_no_none_values(self):
        for d in LEGAL_DATA:
            assert None not in DocMetadata.model_validate(d["metadata"]).to_chroma().values()

    def test_curated_docs_not_presented_as_case_law(self):
        curated = [d for d in LEGAL_DATA if d["_file"] == "curated_summaries.jsonl"]
        assert curated
        for d in curated:
            m = DocMetadata.model_validate(d["metadata"])
            assert m.doc_type != "case_law"
            assert "curated" in m.source_name.lower()

    def test_ids_unique_and_stable(self):
        a, b = build_documents(), build_documents()
        assert [d["id"] for d in a] == [d["id"] for d in b]
        assert len({d["id"] for d in a}) == len(a)


class TestMetadataRules:
    def test_case_law_requires_source(self):
        with pytest.raises(ValidationError):
            DocMetadata(doc_type="case_law", domain="criminal", court="Supreme Court of India")

    def test_case_law_ok_with_court_date_url(self):
        DocMetadata(doc_type="case_law", domain="criminal", court="Supreme Court of India",
                    date="2014-07-02", source_url="https://indiankanoon.org/doc/1")

    @pytest.mark.parametrize("field,value", [
        ("domain", "tax"), ("date", "02/07/2014"), ("source_url", "indiankanoon.org"),
        ("jurisdiction", "UP"), ("doc_type", "blog"),
    ])
    def test_invalid_values_rejected(self, field, value):
        base = {"doc_type": "legal_guidance", "domain": "general"}
        with pytest.raises(ValidationError):
            DocMetadata(**{**base, field: value})

    def test_unknown_key_rejected(self):
        with pytest.raises(ValidationError):
            DocMetadata(doc_type="legal_guidance", domain="general", type="legacy")

    def test_ipc_section_requires_act_and_section(self):
        with pytest.raises(ValidationError):
            DocMetadata(doc_type="ipc_section", domain="criminal", act="The Indian Penal Code, 1860")
