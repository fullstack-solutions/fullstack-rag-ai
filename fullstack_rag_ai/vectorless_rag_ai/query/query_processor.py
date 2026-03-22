import re


class QueryProcessor:
    def process(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r"[^\w\s]", "", query)
        return query.strip()