# types.py

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DocumentChunk:
    """
    Represents a single chunk of a document.

    Attributes:
        id: Unique identifier of the chunk.
        text: The text content of the chunk.
        metadata: Optional metadata dictionary (e.g., source file, page number).
    """
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class RetrievalResult:
    """
    Represents a single retrieval result.

    Attributes:
        chunk: The DocumentChunk retrieved.
        score: The retrieval score (BM25 or any custom metric).
    """
    chunk: DocumentChunk
    score: float

@dataclass
class VectorlessConfig:
    """
    Configuration class for the Vectorless RAG pipeline.

    Attributes:
        top_k: Number of top documents to retrieve.
        max_context_length: Maximum length of concatenated context.
        k1: BM25 parameter (term frequency scaling).
        b: BM25 parameter (document length normalization).
    """
    top_k: int = 5
    max_context_length: int = 2000
    k1: float = 1.5
    b: float = 0.75