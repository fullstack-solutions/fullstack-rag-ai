from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
import os

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
        embeddings: HuggingFaceEmbeddings = None,
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
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        elif self.strategy == "token":
            splitter = TokenTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )

        elif self.strategy == "semantic":
            try:
                if not self.embeddings:
                    raise ValueError(
                        "Embeddings are required for semantic chunking"
                    )

                splitter = SemanticChunker(self.embeddings)

            except ImportError:
                raise ImportError(
                    "SemanticChunker requires 'langchain-experimental'. "
                    "Install it to use this option."
                )

        else:
            raise ValueError(
                f"Unsupported chunking strategy: {self.strategy}"
            )

        return splitter