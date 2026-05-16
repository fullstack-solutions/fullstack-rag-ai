import os

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from langchain_experimental.text_splitter import (
    SemanticChunker
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

class TextSplitterFactory:
    """
    Factory class to create configured text splitters based on strategy.

    Supported strategies:
        - "recursive"
        - "token"
        - "semantic"
    """

    def __init__(
        self,
        strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        embeddings=None,
    ):
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embeddings = embeddings

    def get_splitter(self):
        """
        Return a configured text splitter instance.

        Raises:
            ValueError: If unsupported strategy is provided
            ImportError: If semantic dependencies are missing
        """

        if self.strategy == "recursive":

            return RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        elif self.strategy == "token":

            return TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        elif self.strategy == "semantic":

            if not self.embeddings:
                raise ValueError(
                    "Embeddings required for semantic chunking"
                )

            return SemanticChunker(
                self.embeddings
            )

        raise ValueError(
            f"Unsupported strategy: {self.strategy}"
        )