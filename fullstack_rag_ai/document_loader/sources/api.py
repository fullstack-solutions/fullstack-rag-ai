from typing import List
from .base import BaseSource


class APISource(BaseSource):
    def __init__(self, fetch_fn):
        self.fetch_fn = fetch_fn

    def load(self) -> List[dict]:
        data = self.fetch_fn()

        docs = []
        for item in data:
            docs.append({
                "content": str(item),
                "metadata": {"source": "api"}
            })

        return docs