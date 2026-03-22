class QueryExpander:
    def __init__(self, llm_callable):
        self.llm = llm_callable

    def expand(self, query: str) -> list[str]:
        prompt = f"""
Expand the following query into 3 alternative search queries.

Query: {query}

Return as a comma-separated list.
"""

        response = self.llm(prompt)

        try:
            queries = [q.strip() for q in response.split(",") if q.strip()]
            return list(set([query] + queries))
        except:
            return [query]