# simple_ranker.py
from typing import List
from vectorless_rag_ai.types import RetrievalResult

class SimpleReranker:
    """
    Placeholder reranker that keeps original ranking.
    Can be extended by user.
    """
    def rerank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        return results