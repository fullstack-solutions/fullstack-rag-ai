import os
from typing import List
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader
)

class UniversalLoader:

    def load(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext.lower() == ".pdf":
                docs = PyPDFLoader(file_path).load()
            for d in docs:
                d.metadata["source"] = file_path
                d.metadata["file_type"] = ext

            return docs

        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            return []