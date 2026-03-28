import os
from typing import List
from langchain_core.documents import Document

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredFileLoader,
    Docx2txtLoader
)


class UniversalLoader:

    def load(self, file_path: str) -> List[Document]:
        ext = os.path.splitext(file_path)[1].lower()

        try:
            if ext == ".pdf":
                loader = PyPDFLoader(file_path)

            elif ext == ".csv":
                loader = CSVLoader(file_path)

            elif ext in [".doc", ".docx"]:
                loader = Docx2txtLoader(file_path)

            elif ext in [
                ".txt", ".md", ".py", ".js",
                ".ts", ".html", ".css",
                ".json", ".yaml", ".yml"
            ]:
                loader = TextLoader(file_path, encoding="utf-8")

            else:
                loader = UnstructuredFileLoader(file_path)

            docs = loader.load()

            for d in docs:
                d.metadata["source"] = file_path
                d.metadata["file_type"] = ext

            return docs

        except Exception as e:
            print(f"[ERROR] {file_path}: {e}")
            return []