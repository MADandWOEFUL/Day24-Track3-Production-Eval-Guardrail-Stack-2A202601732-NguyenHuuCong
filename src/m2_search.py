from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K,
                    OPENAI_API_KEY, OPENAI_BASE_URL)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize
        segmented = word_tokenize(text, format="text")
        return segmented.replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        if not chunks:
            return
        self.documents = chunks
        self.corpus_tokens = [
            segment_vietnamese(chunk.get("text", "")).lower().split()
            for chunk in chunks
        ]
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or not self.documents:
            return []
        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []
        scores = self.bm25.get_scores(tokenized_query)
        scored_indices = [
            (i, float(scores[i])) for i in range(len(scores)) if scores[i] > 0
        ]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        top_indices = scored_indices[:top_k]
        return [
            SearchResult(
                text=self.documents[idx]["text"],
                score=score,
                metadata=self.documents[idx].get("metadata", {}),
                method="bm25"
            )
            for idx, score in top_indices
        ]


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._openai_client = None
        self._encoder = None

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            kwargs = {}
            if OPENAI_API_KEY:
                kwargs["api_key"] = OPENAI_API_KEY
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            self._openai_client = OpenAI(**kwargs)
        return self._openai_client

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if EMBEDDING_MODEL.startswith("text-embedding") or EMBEDDING_MODEL.startswith("openai/"):
            client = self._get_openai_client()
            batch_size = 100
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = client.embeddings.create(input=batch, model=EMBEDDING_MODEL)
                embeddings.extend([item.embedding for item in response.data])
            return embeddings
        else:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            vectors = self._encoder.encode(texts, show_progress_bar=False)
            return [v.tolist() for v in vectors]

    def _encode_query(self, query: str) -> list[float]:
        if EMBEDDING_MODEL.startswith("text-embedding") or EMBEDDING_MODEL.startswith("openai/"):
            client = self._get_openai_client()
            response = client.embeddings.create(input=[query], model=EMBEDDING_MODEL)
            return response.data[0].embedding
        else:
            if self._encoder is None:
                from sentence_transformers import SentenceTransformer
                self._encoder = SentenceTransformer(EMBEDDING_MODEL)
            return self._encoder.encode(query).tolist()

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        if not chunks:
            return
        from qdrant_client.models import Distance, VectorParams, PointStruct
        self.client.recreate_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
        )
        texts = [c["text"] for c in chunks]
        vectors = self._encode(texts)
        points = [
            PointStruct(
                id=i,
                vector=v,
                payload={**c.get("metadata", {}), "text": c["text"]}
            )
            for i, (c, v) in enumerate(zip(chunks, vectors))
        ]
        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=collection,
                points=points[i:i + batch_size]
            )

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if not query:
            return []
        query_vector = self._encode_query(query)
        response = self.client.query_points(
            collection_name=collection,
            query=query_vector,
            limit=top_k
        )
        return [
            SearchResult(
                text=pt.payload.get("text", ""),
                score=pt.score,
                metadata=pt.payload,
                method="dense"
            )
            for pt in response.points
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank + 1)."""
    rrf_scores = {}  # text -> {"score": float, "metadata": dict, "text": str}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "metadata": result.metadata,
                    "text": result.text
                }
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["text"],
            score=item["score"],
            metadata=item["metadata"],
            method="hybrid"
        )
        for item in sorted_items
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
