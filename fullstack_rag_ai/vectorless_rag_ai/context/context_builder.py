from typing import List
from vectorless_rag_ai.types import RetrievalResult


class ContextBuilder:
    def build(self, results: List[RetrievalResult], max_length: int) -> str:
        context = ""
        for r in results:
            if len(context) + len(r.chunk.text) > max_length:
                break
            context += r.chunk.text + "\n\n"

        return context.strip()