from typing import List
from langchain.schema import Document


class TextListDocumentLoader:
    """
    Converts a list of text strings into LangChain Document objects.
    """

    def __init__(self, texts: List[str]):
        self.texts = texts

    def load(self) -> List[Document]:
        return [Document(page_content=t) for t in self.texts]


class TextDocumentLoader:
    """Loads documents from a list of text strings with logging and error handling."""

    def __init__(self, texts: List[str]):
        self.texts = texts

    def load(self) -> List[Document]:
        try:
            loader = TextListDocumentLoader(self.texts)
            docs = loader.load()
            print(f"[INFO] Loaded {len(docs)} text documents")
            return docs
        except Exception as e:
            print(f"[ERROR] Failed to load documents from text: {e}")
            return []