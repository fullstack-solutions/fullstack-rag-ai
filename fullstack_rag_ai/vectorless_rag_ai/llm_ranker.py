# llm_ranker.py
from typing import List
from .types import RetrievalResult

class LLMReranker:
    """
    LLM-based reranker.
    Users can plug in OpenAI, local model, etc.
    """
    def __init__(self, llm_callable):
        self.llm = llm_callable

    def rerank(self, query: str, results: List[RetrievalResult]) -> List[RetrievalResult]:
        if not results:
            return results
        prompt = self._build_prompt(query, results)
        response = self.llm(prompt)
        order = self._parse_response(response, len(results))
        return [results[i] for i in order if i < len(results)]

    def _build_prompt(self, query: str, results: List[RetrievalResult]) -> str:
        chunks_text = "\n\n".join([f"{i}. {r.chunk.text}" for i, r in enumerate(results)])
        return f"""
You are a ranking system.

Query:
{query}

Documents:
{chunks_text}

Return ONLY a list of indices (most relevant first).
Example: 2,0,1
"""

    def _parse_response(self, response: str, max_len: int):
        try:
            indices = [int(x.strip()) for x in response.split(",")]
            return [i for i in indices if 0 <= i < max_len]
        except:
            return list(range(max_len))