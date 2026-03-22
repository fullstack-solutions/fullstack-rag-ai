# bm25_retriever.py
from typing import List
from vectorless_rag_ai.types import RetrievalResult
from vectorless_rag_ai.indexing.bm25_index import BM25Index

class BM25Retriever:
    """
    BM25-based retriever.
    Wraps BM25Index and returns top-k results.
    """
    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        results = self.index.search(query)
        # Sort descending by score
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        return [RetrievalResult(chunk=doc, score=score) for doc, score in ranked[:top_k]]