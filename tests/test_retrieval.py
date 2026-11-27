"""
Retrieval tests (CLAUDE.md issue #9): tokenizer, multi-query RRF, English
rewrite wiring, eval scoring, and a quality regression check on the real
local embedding model + corpus. No paid APIs (Cohere disabled by conftest).
Run: pytest tests/test_retrieval.py -v
"""
import json

import pytest

from backend.evaluation import retrieval_eval as ev
from backend.rag.hybrid_search import hybrid_rag, tokenize
from tests.test_agent_outputs import fake_openai
from tests.test_pipeline import make_state


class TestTokenizer:
    def test_punctuation_stripped(self):
        assert tokenize("Section 498A, IPC? flat.") == ["section", "498a", "ipc", "flat"]

    def test_devanagari_words_kept_whole(self):
        # matras (vowel signs) must not split words
        assert tokenize("मकान मालिक, किरायेदार!") == ["मकान", "मालिक", "किरायेदार"]

    def test_empty(self):
        assert tokenize("") == [] and tokenize("?!") == []


class TestFusion:
    def test_fuse_many_lists(self):
        a = [{"id": "x"}, {"id": "y"}]
        b = [{"id": "y"}, {"id": "z"}]
        c = [{"id": "y"}]
        fused = hybrid_rag.fuse([a, b, c])
        assert [d["id"] for d in fused][0] == "y"          # found by all three lists
        assert {d["id"] for d in fused} == {"x", "y", "z"}

    def test_rrf_two_lists_compat(self):
        out = hybrid_rag.reciprocal_rank_fusion([{"id": "a"}], [{"id": "a"}, {"id": "b"}])
        assert [d["id"] for d in out] == ["a", "b"]

    def test_retrieve_dedupes_queries(self, monkeypatch):
        seen = []
        monkeypatch.setattr(hybrid_rag, "dense_search", lambda q, k=10: seen.append(q) or [])
        monkeypatch.setattr(hybrid_rag, "sparse_search", lambda q, k=10: [])
        hybrid_rag.retrieve(["Deposit", " deposit ", "", None, "landlord"])
        assert seen == ["Deposit", "landlord"]


class TestRewriteWiring:
    def test_orchestrator_returns_normalized_rewrite(self, monkeypatch):
        from backend.agents import orchestrator
        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(json.dumps({
            "legal_domain": "family", "language": "hinglish",
            "search_query_en": "  wife claiming   monthly maintenance\nfrom husband "})))
        out = orchestrator.orchestrator_agent(make_state(user_query="patni kharcha chahti hai"))
        assert out["search_query_en"] == "wife claiming monthly maintenance from husband"

    def test_missing_rewrite_is_empty(self, monkeypatch):
        from backend.agents import orchestrator
        monkeypatch.setattr(orchestrator, "OpenAI", fake_openai(json.dumps({"legal_domain": "family"})))
        assert orchestrator.orchestrator_agent(make_state(user_query="q"))["search_query_en"] == ""

    @pytest.mark.parametrize("rewrite,expected", [("wife maintenance", ["wife maintenance"]), ("", None)])
    def test_research_passes_rewrite(self, monkeypatch, rewrite, expected):
        from backend.agents import research
        calls = []
        monkeypatch.setattr(research.hybrid_rag, "search",
                            lambda q, d=None, **kw: calls.append(kw.get("extra_queries")) or [])
        research.research_agent(make_state(user_query="patni kharcha", search_query_en=rewrite))
        assert calls == [expected]


class TestEvalScoring:
    def test_rules(self):
        meta = {"topic": "eviction", "act": "The Uttar Pradesh Urban Buildings ... Act, 1972", "section": "498A"}
        assert ev.is_relevant(meta, [{"topic": "eviction"}])
        assert ev.is_relevant(meta, [{"act": "uttar pradesh"}])          # contains, case-insensitive
        assert ev.is_relevant(meta, [{"section": "498a"}])
        assert not ev.is_relevant(meta, [{"topic": "eviction", "section": "420"}])   # AND within a rule

    def test_score(self):
        rel = [{"topic": "t"}]
        assert ev.score([{}, {"topic": "t"}], rel) == {"hit@1": 0.0, "hit@5": 1.0, "mrr@10": 0.5}
        assert ev.score([{}] * 10, rel)["mrr@10"] == 0.0


class TestRetrievalQuality:
    """Regression guard on the real model + corpus (local, free). Thresholds sit
    below current numbers; a drop means retrieval got worse."""

    @pytest.fixture(scope="class")
    def report(self):
        if hybrid_rag.collection.count() == 0:
            pytest.skip("corpus not loaded — run python -m backend.data_loader")
        methods = {k: v for k, v in ev.default_methods().items()
                   if k in ("hybrid (RRF)", "hybrid+en (oracle)")}
        return ev.evaluate(methods, ev.load_items())["results"]

    # Floors re-based 2026-11-26 on the full corpus (9 Acts, ~3.5k chunks); measured:
    # hybrid+en  en 0.884 / hi 0.935 / hinglish 0.815 MRR@10; hybrid en 0.884 / hi 0.675.
    def test_production_path_floor(self, report):
        """Production searches with the original query + the orchestrator's English rewrite."""
        r = report["hybrid+en (oracle)"]
        assert r["en"]["mrr@10"] >= 0.85
        assert r["hi"]["mrr@10"] >= 0.85
        assert r["hinglish"]["mrr@10"] >= 0.75
        # 2026-11-26, + 2 Labour Codes (4.6k chunks): "fired without notice" misses top-5 in en
        # (statute says "retrenchment"; curated summary ranks #8). Known issue in CLAUDE.md.
        # 2026-11-28, + OSH Code (5.2k chunks): the Hindi "wrongful termination" query ranks
        # its first relevant doc #4 locally and #6 on CI (CPU torch); same known issue.
        assert r["en"]["hit@5"] >= 0.94 and r["hi"]["hit@5"] >= 0.94

    def test_hybrid_floor_without_rewrite(self, report):
        r = report["hybrid (RRF)"]
        assert r["en"]["mrr@10"] >= 0.85
        assert r["hi"]["mrr@10"] >= 0.6

    def test_english_rewrite_helps_hinglish(self, report):
        assert report["hybrid+en (oracle)"]["hinglish"]["mrr@10"] >= report["hybrid (RRF)"]["hinglish"]["mrr@10"] + 0.1


class TestRerank:
    DOCS = [{"id": "a", "content": "A"}, {"id": "b", "content": "B"}, {"id": "c", "content": "C"}]

    class FakeCohere:
        def __init__(self, fail=False):
            self.calls, self.fail = [], fail

        def rerank(self, **kw):
            self.calls.append(kw)
            if self.fail:
                err = RuntimeError("model sunset")
                err.status_code = 404
                raise err
            from types import SimpleNamespace as NS
            return NS(results=[NS(index=2, relevance_score=0.9), NS(index=0, relevance_score=0.5)])

    @pytest.fixture
    def rag(self, monkeypatch):
        monkeypatch.setattr(hybrid_rag, "_rerank_disabled_until", 0.0)
        return hybrid_rag

    def test_uses_configured_model_and_reorders(self, rag, monkeypatch):
        from backend.config import COHERE_RERANK_MODEL
        fake = self.FakeCohere()
        monkeypatch.setattr(rag, "cohere_client", fake)
        out = rag.rerank("q", self.DOCS, top_k=2)
        assert [d["id"] for d in out] == ["c", "a"] and out[0]["rerank_score"] == 0.9
        assert fake.calls[0]["model"] == COHERE_RERANK_MODEL != "rerank-multilingual-v2.0"

    def test_failure_falls_back_and_pauses(self, rag, monkeypatch):
        fake = self.FakeCohere(fail=True)
        monkeypatch.setattr(rag, "cohere_client", fake)
        assert [d["id"] for d in rag.rerank("q", self.DOCS, top_k=2)] == ["a", "b"]   # RRF order
        rag.rerank("q", self.DOCS, top_k=2)
        assert len(fake.calls) == 1                     # paused: no second failing call

    def test_resumes_after_pause(self, rag, monkeypatch):
        fake = self.FakeCohere(fail=True)
        monkeypatch.setattr(rag, "cohere_client", fake)
        rag.rerank("q", self.DOCS)
        monkeypatch.setattr(rag, "_rerank_disabled_until", 0.0)       # window elapsed
        rag.rerank("q", self.DOCS)
        assert len(fake.calls) == 2
