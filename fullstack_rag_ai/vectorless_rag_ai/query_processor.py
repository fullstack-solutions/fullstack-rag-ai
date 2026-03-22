import re

class QueryProcessor:
    """
    Simple query normalization: lowercase + remove special characters.
    """
    def process(self, query: str) -> str:
        query = query.lower()
        query = re.sub(r"[^\w\s]", "", query)
        return query.strip()