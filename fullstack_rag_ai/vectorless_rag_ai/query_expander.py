# query_expander.py

from typing import List
from .config import DEFAULT_CONFIG

class QueryExpander:
    """
    Optional query expansion using an LLM.

    Args:
        llm_callable: Function that takes a prompt string and returns a string response.
        prompt_template: Optional string template with placeholder {query}.
    """

    def __init__(self, llm_callable, prompt_template: str = None):
        self.llm = llm_callable
        self.prompt_template = prompt_template or DEFAULT_CONFIG["query_expander_prompt"]

    def expand(self, query: str) -> List[str]:
        """
        Expand the original query into multiple alternative queries.

        Args:
            query: Original user query.

        Returns:
            List of queries including the original.
        """
        prompt = self.prompt_template.format(query=query)
        response = self.llm(prompt)

        # Split by comma and remove duplicates
        queries = [q.strip() for q in response.split(",") if q.strip()]
        return list({query, *queries})