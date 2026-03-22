from typing import Optional
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings


def get_text_splitter(
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: Optional[str] = None
): 
    """
    Return a configured text splitter based on the selected chunking strategy.

    This factory function enables flexible chunking approaches for document
    preprocessing in RAG pipelines.

    Supported strategies:
        - "recursive": Character-based recursive splitting (default, general-purpose)
        - "token": Token-based splitting (better aligned with LLM token limits)
        - "semantic": Embedding-based semantic chunking (groups text by meaning)

    Args:
        strategy (str): Chunking strategy to use. One of:
                        ["recursive", "token", "semantic"]
        chunk_size (int): Maximum size of each chunk
                          (characters for recursive, tokens for token splitter)
        chunk_overlap (int): Overlap between consecutive chunks
        embedding_model (Optional[str]): Required only for "semantic" strategy.
                                         Name of HuggingFace embedding model.

    Returns:
        TextSplitter: A configured LangChain text splitter instance.

    Raises:
        ValueError: If an unsupported strategy is provided
        ImportError: If semantic chunking is used without required dependencies
    """

    if strategy == "recursive":
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    elif strategy == "token":
        splitter = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    elif strategy == "semantic":
        try:
            if not embedding_model:
                raise ValueError("embedding_model is required for semantic chunking")

            embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
            splitter = SemanticChunker(embeddings)

        except ImportError:
            raise ImportError(
                "SemanticChunker requires 'langchain-experimental'. "
                "Install it to use this option."
            )

    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    return splitter