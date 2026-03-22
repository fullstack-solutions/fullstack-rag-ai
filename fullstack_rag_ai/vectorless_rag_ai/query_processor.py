# query_processor.py

import re

class QueryProcessor:
    """
    Normalize queries by:
    - Converting to lowercase.
    - Removing special characters.
    """

    def process(self, query: str) -> str:
        """
        Process a query string.

        Args:
            query: Original query string.

        Returns:
            Normalized query string.
        """
        query = query.lower()
        return re.sub(r"[^\w\s]", "", query).strip()