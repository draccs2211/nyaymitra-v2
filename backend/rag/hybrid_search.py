"""
Hybrid RAG Pipeline - NyayMitra 2.0
Fixed: Canonical IDs — BM25 aur ChromaDB same IDs use karte hain.
Dense + Sparse search with RRF fusion + Cohere reranking.
"""
from typing import List, Dict, Any, Optional
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from backend.config import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    TOP_K_RESULTS, RERANK_TOP_K, COHERE_API_KEY
)
from backend.rag.metadata import DocMetadata
import cohere
import pickle
import os


class HybridRAG:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None

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
            }
            with open(self._bm25_cache_path(), "wb") as f:
                pickle.dump(cache, f)
            print(f"[BM25] Index saved — {len(self.bm25_docs)} docs")
        except Exception as e:
            print(f"[BM25] Save failed: {e}")

    def _load_bm25_index(self):
        """Load BM25 index from disk if available."""
        path = self._bm25_cache_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                cache = pickle.load(f)
            self.bm25_index = cache["index"]
            self.bm25_docs = cache["docs"]
            self.bm25_id_map = cache["id_map"]
            self.bm25_ids = cache["ids"]
            self.bm25_metas = cache.get("metas") or [{} for _ in self.bm25_ids]
            print(f"[BM25] Index loaded from disk — {len(self.bm25_docs)} docs")
        except Exception as e:
            print(f"[BM25] Load failed: {e}")

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
            tokenized = [d.lower().split() for d in self.bm25_docs]
            self.bm25_index = BM25Okapi(tokenized)
        else:
            self.bm25_index = None
        self._save_bm25_index()
        print(f"[BM25] Index built — {len(self.bm25_docs)} docs")

    def rebuild_bm25_from_chroma(self):
        """Rebuild BM25 from the full Chroma collection so both stay in sync."""
        data = self.collection.get(include=["documents", "metadatas"])
        metas = data.get("metadatas") or [{} for _ in data["ids"]]
        self.build_bm25_index([
            {"id": i, "content": c, "metadata": m or {}}
            for i, c, m in zip(data["ids"], data["documents"], metas)
        ])

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

        tokenized_query = query.lower().split()
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

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        RRF fusion using canonical IDs.
        Same doc found by both dense + sparse → score DOUBLES correctly.
        """
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        for rank, doc in enumerate(dense_results):
            canonical_id = doc["id"]
            scores[canonical_id] = scores.get(canonical_id, 0) + 1 / (rank + k)
            doc_map[canonical_id] = doc

        for rank, doc in enumerate(sparse_results):
            canonical_id = doc["id"]
            # ← Now correctly identifies same doc found by both searches!
            scores[canonical_id] = scores.get(canonical_id, 0) + 1 / (rank + k)
            if canonical_id not in doc_map:
                doc_map[canonical_id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = []
        for canonical_id in sorted_ids:
            doc = doc_map[canonical_id].copy()
            doc["rrf_score"] = round(scores[canonical_id], 4)
            fused.append(doc)
        return fused

    # ── Reranking ─────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        docs: List[Dict],
        top_k: int = RERANK_TOP_K
    ) -> List[Dict]:
        """Cohere reranking for better accuracy."""
        if not docs:
            return docs

        if self.cohere_client:
            try:
                texts = [d["content"][:512] for d in docs]
                response = self.cohere_client.rerank(
                    query=query,
                    documents=texts,
                    top_n=min(top_k, len(docs)),
                    model="rerank-multilingual-v2.0"
                )
                reranked = []
                for r in response.results:
                    doc = docs[r.index].copy()
                    doc["rerank_score"] = round(r.relevance_score, 4)
                    reranked.append(doc)
                return reranked
            except Exception as e:
                print(f"[Rerank] Cohere failed: {e} — using RRF order")

        return docs[:top_k]

    # ── Main Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        legal_domain: Optional[str] = None,
        top_k: int = TOP_K_RESULTS
    ) -> List[Dict]:
        """
        Full hybrid search pipeline:
        1. Dense search (ChromaDB)
        2. Sparse search (BM25) — same canonical IDs
        3. RRF fusion — correctly combines scores
        4. Reranking (Cohere)
        5. Domain filtering
        """
        dense_results = self.dense_search(query, top_k)
        sparse_results = self.sparse_search(query, top_k)

        # Log for debugging
        print(f"[RAG] Dense: {len(dense_results)} | Sparse: {len(sparse_results)}")

        # RRF fusion — canonical IDs ensure correct score combining
        fused = self.reciprocal_rank_fusion(dense_results, sparse_results)

        # Rerank
        reranked = self.rerank(query, fused, top_k=RERANK_TOP_K)

        # Domain filter — if no results after filter, return all
        if legal_domain:
            filtered = [
                d for d in reranked
                if d.get("metadata", {}).get("domain", "") in [legal_domain, "general"]
            ]
            reranked = filtered if filtered else reranked

        return reranked

    # ── Add Documents ─────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict]):
        """
        Idempotently upsert documents into ChromaDB, then rebuild BM25 from
        the full collection so both indexes hold the same canonical IDs.
        Each document must have: id (canonical, deterministic), content, metadata (optional).
        """
        if not documents:
            return

        ids, texts, metadatas, embeddings = [], [], [], []
        seen = set()

        for doc in documents:
            canonical_id = doc["id"]
            if canonical_id in seen:
                continue  # duplicate within batch — Chroma rejects these
            seen.add(canonical_id)
            ids.append(canonical_id)
            texts.append(doc["content"])
            # Validate against canonical schema — raises on bad corpus data
            metadatas.append(DocMetadata.model_validate(doc.get("metadata") or {}).to_chroma())
            embeddings.append(self.get_embedding(doc["content"]))

        # Chroma upsert MERGES metadata, so keys removed from the schema would
        # linger. Delete docs whose stored keys aren't a subset of the new ones.
        existing = self.collection.get(ids=ids, include=["metadatas"])
        new_keys = {i: set(m) for i, m in zip(ids, metadatas)}
        stale = [
            i for i, m in zip(existing["ids"], existing["metadatas"] or [])
            if set(m or {}) - new_keys[i]
        ]
        if stale:
            self.collection.delete(ids=stale)

        # Upsert: re-running with same IDs updates instead of duplicating
        self.collection.upsert(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )

        # BM25 mirrors the FULL collection, not just this batch
        self.rebuild_bm25_from_chroma()

        print(f"[RAG] Upserted {len(ids)} docs — ChromaDB + BM25 in sync ✅")

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