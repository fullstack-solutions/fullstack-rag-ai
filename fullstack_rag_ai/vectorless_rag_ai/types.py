# types.py
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DocumentChunk:
    """
    Represents a document chunk that can be retrieved.
    """
    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class RetrievalResult:
    """
    Represents a retrieval result: the document chunk + its score.
    """
    chunk: DocumentChunk
    score: float


@dataclass
class VectorlessConfig:
    """
    Configuration for the vectorless RAG pipeline.
    Users can modify BM25 parameters, context length, top_k, etc.
    """
    top_k: int = 5
    max_context_length: int = 2000
    k1: float = 1.5  # BM25 parameter
    b: float = 0.75  # BM25 parameter