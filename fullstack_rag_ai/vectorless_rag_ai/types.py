from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DocumentChunk:
    id: str
    text: str
    metadata: Dict[str, Any] = None


@dataclass
class RetrievalResult:
    chunk: DocumentChunk
    score: float


@dataclass
class VectorlessConfig:
    top_k: int = 5
    max_context_length: int = 2000