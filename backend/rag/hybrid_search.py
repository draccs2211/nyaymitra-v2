from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from backend.config import (
    EMBEDDING_MODEL, CHROMA_PERSIST_DIR,
    TOP_K_RESULTS, RERANK_TOP_K, COHERE_API_KEY
)
import cohere
import json
import hashlib
import pickle


class HybridRAG:
    def __init__(self):
        self.embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.cohere_client = cohere.Client(COHERE_API_KEY) if COHERE_API_KEY else None
        self.bm25_index = None
        self.bm25_docs = []
        self.collection = None
        self._init_collection()

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

    def build_bm25_index(self, documents: List[str]):
        self.bm25_docs = documents
        tokenized = [doc.lower().split() for doc in documents]
        self.bm25_index = BM25Okapi(tokenized)

    def get_embedding(self, text: str) -> List[float]:
        return self.embedding_model.encode(text).tolist()

    def dense_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        query_embedding = self.get_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1)
        )
        docs = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"][0]):
                docs.append({
                    "id": results["ids"][0][i],
                    "content": doc,
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "score": 1 - results["distances"][0][i] if results["distances"] else 0.5,
                    "source": "dense"
                })
        return docs

    def sparse_search(self, query: str, top_k: int = TOP_K_RESULTS) -> List[Dict]:
        if not self.bm25_index or not self.bm25_docs:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25_index.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        docs = []
        for idx in top_indices:
            if scores[idx] > 0:
                docs.append({
                    "id": str(idx),
                    "content": self.bm25_docs[idx],
                    "score": float(scores[idx]),
                    "source": "sparse"
                })
        return docs

    def reciprocal_rank_fusion(
        self,
        dense_results: List[Dict],
        sparse_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        scores: Dict[str, float] = {}
        doc_map: Dict[str, Dict] = {}

        for rank, doc in enumerate(dense_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
            doc_map[doc_id] = doc

        for rank, doc in enumerate(sparse_results):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (rank + k)
            if doc_id not in doc_map:
                doc_map[doc_id] = doc

        sorted_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        fused = []
        for doc_id in sorted_ids:
            doc = doc_map[doc_id].copy()
            doc["rrf_score"] = scores[doc_id]
            fused.append(doc)
        return fused

    def rerank(self, query: str, docs: List[Dict], top_k: int = RERANK_TOP_K) -> List[Dict]:
        if not docs:
            return docs
        if self.cohere_client:
            try:
                texts = [d["content"][:512] for d in docs]
                response = self.cohere_client.rerank(
                    query=query,
                    documents=texts,
                    top_n=top_k,
                    model="rerank-multilingual-v2.0"
                )
                reranked = []
                for r in response.results:
                    doc = docs[r.index].copy()
                    doc["rerank_score"] = r.relevance_score
                    reranked.append(doc)
                return reranked
            except Exception:
                pass
        return docs[:top_k]

    def search(self, query: str, legal_domain: str = None) -> List[Dict]:
        dense_results = self.dense_search(query)
        sparse_results = self.sparse_search(query)
        fused = self.reciprocal_rank_fusion(dense_results, sparse_results)
        reranked = self.rerank(query, fused)
        if legal_domain:
            reranked = [
                d for d in reranked
                if d.get("metadata", {}).get("domain", "") in [legal_domain, "general"]
            ] or reranked
        return reranked

    def add_documents(self, documents: List[Dict]):
        ids, texts, metadatas, embeddings = [], [], [], []
        for doc in documents:
            ids.append(doc["id"])
            texts.append(doc["content"])
            metadatas.append(doc.get("metadata", {}))
            embeddings.append(self.get_embedding(doc["content"]))
        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings
        )
        self.build_bm25_index([d["content"] for d in documents])


hybrid_rag = HybridRAG()
