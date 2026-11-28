"""
Hybrid RAG Pipeline - NyayMitra 2.0
Fixed: Canonical IDs — BM25 aur ChromaDB same IDs use karte hain.
Dense + Sparse search with RRF fusion + Cohere reranking.
"""
from typing import List, Dict, Any, Optional
import chromadb
from rank_bm25 import BM25Okapi
from backend.config import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    TOP_K_RESULTS, RERANK_TOP_K, COHERE_API_KEY, COHERE_RERANK_MODEL, RERANK_DISABLE_SECONDS
)
from backend.rag.metadata import DocMetadata
import cohere
import logging
import pickle
import os
import re
import threading
import time

logger = logging.getLogger("nyaymitra.rag")

# Bump when tokenize() changes: a cached BM25 index built with an older
# tokenizer is discarded and rebuilt from Chroma.
TOKENIZER_VERSION = 2
_TOKEN_RE = re.compile(r"[0-9a-z\u0900-\u097F]+")


def tokenize(text: str) -> List[str]:
    """
    BM25 tokens: lowercase runs of Latin letters/digits or Devanagari
    (incl. matras), so punctuation never sticks to words ("flat?" -> "flat",
    "498A," -> "498a").
    """
    return _TOKEN_RE.findall((text or "").lower())


def filter_by_jurisdiction(docs: List[Dict], jurisdiction: str) -> List[Dict]:
    """
    State known (IN-XX): keep central (IN) docs and that state's docs only —
    never present another state's law. State unknown (IN): keep everything;
    state-specific docs are labelled downstream instead.
    """
    if not jurisdiction or jurisdiction == "IN":
        return docs
    allowed = {"IN", jurisdiction}
    return [d for d in docs if (d.get("metadata") or {}).get("jurisdiction", "IN") in allowed]


class HybridRAG:
    def __init__(self):
        # Embedding model (~0.5 GB) loads lazily on first use; the API warms it
        # up at startup (main.lifespan) so CLIs/imports don't pay for it.
        self._embedding_model = None
        self._model_lock = threading.Lock()
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None
        self.rerank_model = COHERE_RERANK_MODEL
        self._rerank_disabled_until = 0.0   # set after a failure; avoids a failing call per query

        # ── CANONICAL ID FIX ──────────────────────────────────────────────────
        # BM25 aur ChromaDB same canonical IDs use karenge
        # bm25_id_map: {canonical_id -> index in bm25_docs list}
        # bm25_docs:   list of content strings (same order as IDs)
        self.bm25_index = None
        self.bm25_docs: List[str] = []
        self.bm25_id_map: Dict[str, int] = {}   # canonical_id → list index
        self.bm25_ids: List[str] = []            # list index → canonical_id
        self.bm25_metas: List[Dict] = []         # list index → metadata

        self.collection = None
        self._init_collection()
        self._load_bm25_index()
        # BM25 must mirror Chroma; rebuild if cache missing or stale
        if self.collection.count() != len(self.bm25_ids):
            self.rebuild_bm25_from_chroma()

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            with self._model_lock:
                if self._embedding_model is None:
                    # imported here: sentence_transformers pulls in torch (slow import)
                    from sentence_transformers import SentenceTransformer
                    logger.info("Loading embedding model %s", EMBEDDING_MODEL)
                    self._embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        return self._embedding_model

    @embedding_model.setter
    def embedding_model(self, model) -> None:
        self._embedding_model = model

    def warm_up(self) -> None:
        """Load the embedding model now (call at service startup)."""
        self.get_embedding("warm up")

    def _init_collection(self):
        self.collection = self.chroma_client.get_or_create_collection(
            name="legal_corpus",
            metadata={
                "hnsw:space": "cosine",
                "hnsw:construction_ef": 200,
                "hnsw:M": 32,
                "hnsw:search_ef": 100
            }
        )

    # ── BM25 Persistence ──────────────────────────────────────────────────────

    def _bm25_cache_path(self) -> str:
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        return os.path.join(CHROMA_PERSIST_DIR, "bm25_index.pkl")

    def _save_bm25_index(self):
        """Persist BM25 index to disk so it survives restarts."""
        try:
            cache = {
                "index": self.bm25_index,
                "docs": self.bm25_docs,
                "id_map": self.bm25_id_map,
                "ids": self.bm25_ids,
                "metas": self.bm25_metas,
                "tokenizer": TOKENIZER_VERSION,
            }
            with open(self._bm25_cache_path(), "wb") as f:
                pickle.dump(cache, f)
            logger.info(f"BM25 index saved — {len(self.bm25_docs)} docs")
        except Exception as e:
            logger.warning(f"BM25 save failed: {e}")

    def _load_bm25_index(self):
        """Load BM25 index from disk if available."""
        path = self._bm25_cache_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("tokenizer") != TOKENIZER_VERSION:
                logger.warning("Cached BM25 index uses an old tokenizer — will rebuild")
                return
            self.bm25_index = cache["index"]
            self.bm25_docs = cache["docs"]
            self.bm25_id_map = cache["id_map"]
            self.bm25_ids = cache["ids"]
            self.bm25_metas = cache.get("metas") or [{} for _ in self.bm25_ids]
            logger.info(f"BM25 index loaded from disk — {len(self.bm25_docs)} docs")
        except Exception as e:
            logger.warning(f"BM25 load failed: {e}")

    def build_bm25_index(self, docs: List[Dict]):
        """
        Build BM25 index using canonical IDs from document dicts.
        Each doc must have 'id' and 'content' fields (metadata optional).
        Replaces the whole index — pass the FULL corpus, not a batch.
        """
        self.bm25_docs = []
        self.bm25_id_map = {}
        self.bm25_ids = []
        self.bm25_metas = []

        for doc in docs:
            canonical_id = doc["id"]
            idx = len(self.bm25_docs)
            self.bm25_docs.append(doc["content"])
            self.bm25_metas.append(doc.get("metadata") or {})
            self.bm25_id_map[canonical_id] = idx
            self.bm25_ids.append(canonical_id)

        if self.bm25_docs:
            tokenized = [tokenize(d) for d in self.bm25_docs]
            self.bm25_index = BM25Okapi(tokenized)
        else:
            self.bm25_index = None
        self._save_bm25_index()
        logger.info(f"BM25 index built — {len(self.bm25_docs)} docs")

    def rebuild_bm25_from_chroma(self):
        """Rebuild BM25 from the full Chroma collection so both stay in sync."""
        data = self.collection.get(include=["documents", "metadatas"])
        metas = data.get("metadatas") or [{} for _ in data["ids"]]
        self.build_bm25_index([
            {"id": i, "content": c, "metadata": m or {}}
            for i, c, m in zip(data["ids"], data["documents"], metas)
        ])

    def corpus_docs(self) -> List[Dict]:
        """Every indexed chunk as {'id','content','metadata'} (in-memory BM25 copy, no DB call)."""
        return [{"id": i, "content": c, "metadata": m}
                for i, c, m in zip(self.bm25_ids, self.bm25_docs, self.bm25_metas)]

    # ── Embeddings ────────────────────────────────────────────────────────────

    def get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    # ── Dense Search ──────────────────────────────────────────────────────────

    def dense_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """Semantic search using ChromaDB — returns canonical IDs."""
        query_embedding = self.get_embedding(query)
        count = self.collection.count()
        if count == 0:
            return []

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, count)
        )

        docs = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                canonical_id = results["ids"][0][i]   # ChromaDB stores canonical ID
                docs.append({
                    "id": canonical_id,                # ← canonical ID
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0.5,
                    "source": "dense"
                })
        return docs

    # ── Sparse Search ─────────────────────────────────────────────────────────

    def sparse_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """
        BM25 keyword search — returns SAME canonical IDs as ChromaDB.
        Uses bm25_ids list to map index → canonical_id.
        """
        if not self.bm25_index or not self.bm25_docs:
            return []

        tokenized_query = tokenize(query)
        if not tokenized_query:
            return []
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        docs = []
        for idx in top_indices:
            if scores[idx] > 0:
                canonical_id = self.bm25_ids[idx]     # ← canonical ID (not numeric!)
                docs.append({
                    "id": canonical_id,                # ← SAME canonical ID as ChromaDB
                    "content": self.bm25_docs[idx],
                    "metadata": self.bm25_metas[idx] if idx < len(self.bm25_metas) else {},
                    "score": float(scores[idx]),
                    "source": "sparse"
                })
        return docs

    # ── RRF Fusion ────────────────────────────────────────────────────────────

    def fuse(self, result_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion over any number of ranked lists (dense/sparse,
        per query). Canonical IDs let the same doc accumulate score across lists.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}
        for results in result_lists:
            for rank, doc in enumerate(results):
                cid = doc["id"]
                scores[cid] = scores.get(cid, 0) + 1 / (rank + k)
                doc_map.setdefault(cid, doc)    # first occurrence (dense keeps distance score)

        fused = []
        for cid in sorted(scores, key=lambda x: scores[x], reverse=True):
            doc = doc_map[cid].copy()
            doc["rrf_score"] = round(scores[cid], 4)
            fused.append(doc)
        return fused

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """RRF of one dense + one sparse list (kept for callers/tests)."""
        return self.fuse([dense_results, sparse_results], k)

    def retrieve(self, queries: List[str], top_k: int = TOP_K_RESULTS) -> List[Dict]:
        """Dense + sparse for each distinct query, all fused with RRF (no rerank/filter)."""
        seen, distinct = set(), []
        for q in queries:
            key = (q or "").strip().lower()
            if key and key not in seen:
                seen.add(key)
                distinct.append(q.strip())
        lists = []
        for q in distinct:
            lists.append(self.dense_search(q, top_k))
            lists.append(self.sparse_search(q, top_k))
        logger.info(f"Hybrid retrieval: queries={len(distinct)} lists={[len(x) for x in lists]}")
        return self.fuse(lists)

    # ── Reranking ─────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        docs: List[Dict],
        top_k: int = RERANK_TOP_K
    ) -> List[Dict]:
        """
        Cohere reranking. On any failure (sunset model, quota, network) falls back
        to RRF order and skips Cohere for RERANK_DISABLE_SECONDS.
        """
        if not docs:
            return docs

        if self.cohere_client and time.monotonic() >= self._rerank_disabled_until:
            try:
                texts = [d["content"][:512] for d in docs]
                response = self.cohere_client.rerank(
                    query=query,
                    documents=texts,
                    top_n=min(top_k, len(docs)),
                    model=self.rerank_model,
                )
                reranked = []
                for r in response.results:
                    doc = docs[r.index].copy()
                    doc["rerank_score"] = round(r.relevance_score, 4)
                    reranked.append(doc)
                return reranked
            except Exception as e:
                self._rerank_disabled_until = time.monotonic() + RERANK_DISABLE_SECONDS
                status = getattr(e, "status_code", None)
                detail = getattr(e, "body", None) or str(e)
                logger.warning("Cohere rerank failed (model=%s, status=%s): %s — RRF order used, "
                               "rerank paused %.0fs", self.rerank_model, status,
                               str(detail)[:200], RERANK_DISABLE_SECONDS)

        return docs[:top_k]

    # ── Main Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        legal_domain: Optional[str] = None,
        top_k: int = TOP_K_RESULTS,
        extra_queries: Optional[List[str]] = None,
        jurisdiction: str = "IN",
    ) -> List[Dict]:
        """
        Full hybrid search pipeline:
        1. Dense (ChromaDB) + sparse (BM25) for the query and any extra queries
           (e.g. an English rewrite of a Hindi/Hinglish query)
        2. RRF fusion across all lists (canonical IDs)
        3. Reranking (Cohere) against the original query
        4. Domain filtering (falls back to unfiltered if it removes everything)
        """
        fused = self.retrieve([query] + list(extra_queries or []), top_k)
        fused = filter_by_jurisdiction(fused, jurisdiction)

        reranked = self.rerank(query, fused, top_k=RERANK_TOP_K)

        if legal_domain:
            filtered = [
                d for d in reranked
                if d.get("metadata", {}).get("domain", "") in [legal_domain, "general"]
            ]
            reranked = filtered if filtered else reranked

        return reranked

    # ── Add Documents ─────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict]) -> Dict[str, int]:
        """
        Incremental, idempotent load into ChromaDB, then BM25 rebuilt from the full
        collection (same canonical IDs in both).
        - new IDs (new or edited text): embedded (only case that loads the model)
        - existing IDs: metadata updated in place, no re-embedding
        - existing IDs whose stored metadata has keys the new metadata lacks: re-inserted
          with their stored embedding (Chroma merges metadata on update)
        Each document must have: id (canonical, deterministic), content, metadata.
        """
        if not documents:
            return {"embedded": 0, "updated": 0, "reinserted": 0}

        by_id: Dict[str, Dict] = {}
        for doc in documents:
            if doc["id"] in by_id:
                continue  # duplicate within batch — Chroma rejects these
            # Validate against canonical schema — raises on bad corpus data
            meta = DocMetadata.model_validate(doc.get("metadata") or {}).to_chroma()
            by_id[doc["id"]] = {"content": doc["content"], "metadata": meta}

        ids = list(by_id)
        existing = self.collection.get(ids=ids, include=["metadatas", "embeddings"])
        stored = {i: (m or {}, e) for i, m, e in zip(
            existing["ids"], existing["metadatas"] or [], existing["embeddings"]
            if existing.get("embeddings") is not None else [None] * len(existing["ids"]))}

        new_ids = [i for i in ids if i not in stored]
        reinsert = [i for i in ids if i in stored and set(stored[i][0]) - set(by_id[i]["metadata"])]
        update = [i for i in ids if i in stored and i not in reinsert]

        if new_ids:
            vectors = self.embed_many([by_id[i]["content"] for i in new_ids])
            self.collection.add(
                ids=new_ids, documents=[by_id[i]["content"] for i in new_ids],
                metadatas=[by_id[i]["metadata"] for i in new_ids], embeddings=vectors,
            )
        if update:
            self.collection.update(ids=update, metadatas=[by_id[i]["metadata"] for i in update])
        if reinsert:
            vectors = [list(stored[i][1]) for i in reinsert]
            self.collection.delete(ids=reinsert)
            self.collection.add(
                ids=reinsert, documents=[by_id[i]["content"] for i in reinsert],
                metadatas=[by_id[i]["metadata"] for i in reinsert], embeddings=vectors,
            )

        # BM25 mirrors the FULL collection, not just this batch
        self.rebuild_bm25_from_chroma()
        counts = {"embedded": len(new_ids), "updated": len(update), "reinserted": len(reinsert)}
        logger.info("Corpus load: %s — ChromaDB + BM25 in sync", counts)
        return counts

    def embed_many(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Batch embedding (much faster than one call per document)."""
        return self.embedding_model.encode(texts, batch_size=batch_size, show_progress_bar=False).tolist()

    def delete_ids_not_in(self, keep_ids: List[str]) -> int:
        """Remove docs whose IDs aren't in keep_ids (stale/legacy). Returns count removed."""
        keep = set(keep_ids)
        existing = self.collection.get(include=[])["ids"]
        stale = [i for i in existing if i not in keep]
        if stale:
            self.collection.delete(ids=stale)
            self.rebuild_bm25_from_chroma()
        return len(stale)

    def get_stats(self) -> Dict:
        """Get RAG statistics."""
        return {
            "chroma_count": self.collection.count(),
            "bm25_count": len(self.bm25_docs),
            "ids_in_sync": set(self.collection.get(include=[])["ids"]) == set(self.bm25_ids),
            "cohere_available": self.cohere_client is not None,
        }


# Singleton instance
hybrid_rag = HybridRAG()