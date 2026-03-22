from typing import List
from vectorless_rag_ai.types import RetrievalResult


class SimpleReranker:
    def rerank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        # Placeholder: keep order (already ranked)
        return results