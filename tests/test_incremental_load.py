"""
Incremental corpus load: only new/changed text is embedded; metadata-only
changes never load the embedding model. Uses an in-memory Chroma collection
(never the real index). Run: pytest tests/test_incremental_load.py -v
"""
import uuid

import chromadb
import pytest

from backend.rag.hybrid_search import hybrid_rag


def doc(i, text, **meta):
    return {"id": f"legal_guidance:{i}", "content": text,
            "metadata": {"doc_type": "legal_guidance", "domain": "general", **meta}}


@pytest.fixture
def rag(monkeypatch):
    coll = chromadb.EphemeralClient().create_collection(f"test_{uuid.uuid4().hex[:8]}")
    calls = []

    def fake_embed(texts, batch_size=64):
        calls.append(list(texts))
        return [[float(len(t)), 1.0, 0.0] for t in texts]

    monkeypatch.setattr(hybrid_rag, "collection", coll)
    monkeypatch.setattr(hybrid_rag, "embed_many", fake_embed)
    monkeypatch.setattr(hybrid_rag, "rebuild_bm25_from_chroma", lambda: None)   # don't touch the real BM25 pickle
    hybrid_rag._test_calls = calls
    yield hybrid_rag
    del hybrid_rag._test_calls


class TestIncrementalLoad:
    def test_first_load_embeds_everything(self, rag):
        out = rag.add_documents([doc(1, "alpha"), doc(2, "beta")])
        assert out == {"embedded": 2, "updated": 0, "reinserted": 0}
        assert rag._test_calls == [["alpha", "beta"]]
        assert rag.collection.count() == 2

    def test_rerun_embeds_nothing(self, rag):
        rag.add_documents([doc(1, "alpha"), doc(2, "beta")])
        out = rag.add_documents([doc(1, "alpha"), doc(2, "beta")])
        assert out == {"embedded": 0, "updated": 2, "reinserted": 0}
        assert len(rag._test_calls) == 1                     # no second embedding call

    def test_metadata_change_updates_in_place(self, rag):
        rag.add_documents([doc(1, "alpha", topic="old", source_url="https://a")])
        before = rag.collection.get(ids=["legal_guidance:1"], include=["embeddings"])["embeddings"][0]
        out = rag.add_documents([doc(1, "alpha", topic="new", source_url="https://b")])
        got = rag.collection.get(ids=["legal_guidance:1"], include=["metadatas", "embeddings"])
        assert out["updated"] == 1 and out["embedded"] == 0
        assert got["metadatas"][0]["source_url"] == "https://b" and got["metadatas"][0]["topic"] == "new"
        assert list(got["embeddings"][0]) == list(before)

    def test_removed_key_reinserted_with_stored_embedding(self, rag):
        rag.add_documents([doc(1, "alpha")])
        rag.collection.update(ids=["legal_guidance:1"], metadatas=[{"legacy_key": "x"}])   # simulate old schema
        before = rag.collection.get(ids=["legal_guidance:1"], include=["embeddings"])["embeddings"][0]
        out = rag.add_documents([doc(1, "alpha")])
        got = rag.collection.get(ids=["legal_guidance:1"], include=["metadatas", "embeddings"])
        assert out == {"embedded": 0, "updated": 0, "reinserted": 1}
        assert "legacy_key" not in got["metadatas"][0]
        assert list(got["embeddings"][0]) == list(before)
        assert len(rag._test_calls) == 1

    def test_mixed_batch_only_embeds_new(self, rag):
        rag.add_documents([doc(1, "alpha")])
        out = rag.add_documents([doc(1, "alpha"), doc(3, "gamma")])
        assert out == {"embedded": 1, "updated": 1, "reinserted": 0}
        assert rag._test_calls[-1] == ["gamma"]

    def test_duplicate_ids_in_batch_ignored(self, rag):
        out = rag.add_documents([doc(1, "alpha"), doc(1, "alpha")])
        assert out["embedded"] == 1 and rag.collection.count() == 1
