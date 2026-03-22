# simple_ranker.py

from typing import List
from .types import RetrievalResult

class SimpleReranker:
    """
    Default reranker that preserves original ranking order.
    Can be extended by users to implement custom ranking logic.
    """

    def rerank(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Return results without modification.

        Args:
            results: List of RetrievalResult objects.

        Returns:
            Same list of results in original order.
        """
        return results