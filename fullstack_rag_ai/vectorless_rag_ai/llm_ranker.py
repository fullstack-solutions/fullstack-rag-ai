# llm_ranker.py

from typing import List
from .types import RetrievalResult
from .config import DEFAULT_CONFIG

class LLMReranker:
    """
    LLM-based reranker to reorder retrieval results based on relevance.

    Args:
        llm_callable: Function that takes a prompt string and returns a string response.
        prompt_template: Optional string template with placeholders:
            {query} -> original query
            {documents} -> list of document texts
    """

    def __init__(self, llm_callable, prompt_template: str = None):
        self.llm = llm_callable
        self.prompt_template = prompt_template or DEFAULT_CONFIG["reranker_prompt"]

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """
        Rerank the retrieved results using the LLM.

        Args:
            query: Original user query.
            results: List of RetrievalResult objects.

        Returns:
            Reordered list of RetrievalResult objects.
        """
        if not results:
            return results

        # Build prompt for the LLM
        prompt = self.prompt_template.format(
            query=query,
            documents="\n\n".join(f"{i}. {r.chunk.text}" for i, r in enumerate(results))
        )

        # Call the LLM
        response = self.llm(prompt)

        # Parse returned indices
        try:
            indices = [int(x.strip()) for x in response.split(",")]
            return [results[i] for i in indices if 0 <= i < len(results)]
        except:
            # Fall back to original order if parsing fails
            return results