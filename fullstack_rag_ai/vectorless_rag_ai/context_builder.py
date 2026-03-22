# context_builder.py

from typing import List
from .types import RetrievalResult

class ContextBuilder:
    """
    Concatenates retrieved document chunks to create context for downstream models.

    Stops concatenation once max_length is reached.
    """

    def build(self, results: List[RetrievalResult], max_length: int) -> str:
        """
        Build context from retrieved results.

        Args:
            results: List of RetrievalResult objects.
            max_length: Maximum length of the concatenated context (in characters).

        Returns:
            A single string containing concatenated document chunks.
        """
        context = ""
        for r in results:
            if len(context) + len(r.chunk.text) > max_length:
                break
            context += r.chunk.text + "\n\n"
        return context.strip()