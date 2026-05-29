import os
from typing import List
from langchain_core.documents import Document

from ..loaders.universal_loader import UniversalLoader


class IngestionEngine:
    def __init__(self, source, loader=None):
        self.source = source
        self.loader = loader or UniversalLoader()

    def run(self) -> List[Document]:
        items = self.source.load()
        all_docs = []

        for item in items:

            # File path case
            if isinstance(item, str) and os.path.exists(item):
                docs = self.loader.load(item)
                all_docs.extend(docs)

            # In-memory case
            elif isinstance(item, dict):
                all_docs.append(
                    Document(
                        page_content=item["content"],
                        metadata=item.get("metadata", {})
                    )
                )

        return all_docs