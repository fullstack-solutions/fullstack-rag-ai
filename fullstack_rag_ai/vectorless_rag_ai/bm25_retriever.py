# bm25_retriever.py

from typing import List
from .types import RetrievalResult
from .bm25_index import BM25Index

class BM25Retriever:
    """
    Retrieves top-k documents from a BM25 index.

    Args:
        index: BM25Index object.
    """

    def __init__(self, index: BM25Index):
        self.index = index

    def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for a query.

        Args:
            query: The search query.
            top_k: Number of documents to return.

        Returns:
            List of RetrievalResult objects sorted by score descending.
        """
        results = self.index.search(query)
        ranked = sorted(results, key=lambda x: x[1], reverse=True)
        return [RetrievalResult(chunk=doc, score=score) for doc, score in ranked[:top_k]]