# context_builder.py
from typing import List
from .types import RetrievalResult

class ContextBuilder:
    """
    Build the context for the downstream model.
    Stops concatenation once max_length is reached.
    """
    def build(self, results: List[RetrievalResult], max_length: int) -> str:
        context = ""
        for r in results:
            if len(context) + len(r.chunk.text) > max_length:
                break
            context += r.chunk.text + "\n\n"
        return context.strip()