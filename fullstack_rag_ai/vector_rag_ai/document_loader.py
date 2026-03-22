import os
from typing import List, Optional
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader

from .text_loader import TextDocumentLoader


class BaseDocumentLoader:
    """Abstract base class for document loaders."""

    def load(self) -> List[Document]:
        raise NotImplementedError


class PDFDocumentLoader(BaseDocumentLoader):
    """Loads PDF documents from a directory."""

    def __init__(self, path: str):
        self.path = path

    def load(self) -> List[Document]:
        documents: List[Document] = []

        if not os.path.exists(self.path):
            print(f"[WARN] PDF directory does not exist: {self.path}")
            return documents

        for file in os.listdir(self.path):
            if file.lower().endswith(".pdf"):
                file_path = os.path.join(self.path, file)

                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()

                    # Add source filename to metadata
                    for d in docs:
                        d.metadata["source"] = file

                    documents.extend(docs)
                    print(f"[INFO] Loaded PDF: {file}")

                except FileNotFoundError:
                    print(f"[ERROR] File not found: {file_path}")
                except PermissionError:
                    print(f"[ERROR] Permission denied: {file_path}")
                except Exception as e:
                    print(f"[ERROR] Failed to load PDF {file}: {e}")

        return documents


class DocumentLoader:
    """
    Orchestrates loading from multiple sources (PDF + text).
    """

    def __init__(
        self,
        path: Optional[str] = None,
        texts: Optional[List[str]] = None,
    ):
        self.path = path
        self.texts = texts

    def load(self) -> List[Document]:
        documents: List[Document] = []

        # Load PDFs
        if self.path:
            pdf_loader = PDFDocumentLoader(self.path)
            documents.extend(pdf_loader.load())

        # Load text
        if self.texts:
            text_loader = TextDocumentLoader(self.texts)
            documents.extend(text_loader.load())

        return documents