import os
from typing import Any, Dict, List, Set, Tuple, Union

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import TextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.document_loaders import PyPDFLoader

from .config import DEFAULT_CONFIG
from .utils import get_file_hash, load_binary, save_binary
from .chunking import get_text_splitter


# -------------------
# State management
# -------------------
def load_state(
    metadata_file: str,
    cache_file: str,
    docs_file: str,
    first_run: bool = False
) -> Tuple[Dict[str, str], Dict[str, Any], List[Any]]:
    """
    Load metadata, QA cache, and document chunks from pickle files.
    """
    success, metadata, msg = load_binary(metadata_file)
    if not success and not first_run:
        print(f"[WARN] metadata: {msg}")
    metadata = metadata or {}

    success, qa_cache, msg = load_binary(cache_file)
    if not success and not first_run:
        print(f"[WARN] qa_cache: {msg}")
    qa_cache = qa_cache or {}

    success, all_docs, msg = load_binary(docs_file)
    if not success and not first_run:
        print(f"[WARN] all_docs: {msg}")
    all_docs = all_docs or []

    return metadata, qa_cache, all_docs


# -------------------
# File detection
# -------------------
def get_current_pdf_files(documents_path: str) -> Set[str]:
    """
    Scan a directory for PDF files.
    """
    if not os.path.exists(documents_path):
        print(f"[WARN] Documents path does not exist: {documents_path}")
        return set()
    return {f for f in os.listdir(documents_path) if f.lower().endswith(".pdf")}


def handle_removed_files(
    metadata: Dict[str, str],
    current_files: Set[str],
    all_docs: List[Any],
    qa_cache: Dict[str, Any],
) -> Tuple[List[Any], Dict[str, Any], Dict[str, str], bool]:
    """
    Remove documents that no longer exist in the source directory.
    """
    removed_files = [f for f in metadata if f not in current_files]
    rebuild_needed = bool(removed_files)

    if removed_files:
        all_docs = [d for d in all_docs if d.metadata.get("source") not in removed_files]

        keys_to_delete = [
            k for k, v in qa_cache.items()
            if any(src in removed_files for src in v.get("sources", []))
        ]
        for k in keys_to_delete:
            del qa_cache[k]

        for f in removed_files:
            del metadata[f]
            print(f"[INFO] Removed document: {f}")

    return all_docs, qa_cache, metadata, rebuild_needed


def detect_updated_files(
    documents_path: str,
    current_files: Set[str],
    metadata: Dict[str, str],
) -> Tuple[List[str], Dict[str, str], bool]:
    """
    Detect new or updated PDF files in the directory.
    """
    updated_files = []
    rebuild_needed = False

    for f in current_files:
        file_path = os.path.join(documents_path, f)
        success, new_hash, msg = get_file_hash(file_path)

        if not success:
            print(f"[ERROR] {f}: {msg}")
            continue

        if metadata.get(f) != new_hash:
            updated_files.append(f)
            metadata[f] = new_hash
            rebuild_needed = True
            print(f"[INFO] Updated document: {f}")

    return updated_files, metadata, rebuild_needed


# -------------------
# Document processing
# -------------------
def remove_old_chunks(all_docs: List[Any], updated_files: List[str]) -> List[Any]:
    """
    Remove document chunks that belong to updated files.
    """
    return [d for d in all_docs if d.metadata.get("source") not in updated_files]


def process_new_files(
    documents_path: str,
    updated_files: List[str],
    splitter: Union[TextSplitter, SemanticChunker],
) -> List[Any]:
    """
    Load new or updated PDFs and split them into chunks.
    """
    new_chunks = []

    for f in updated_files:
        try:
            loader = PyPDFLoader(os.path.join(documents_path, f))
            docs = loader.load()

            for d in docs:
                d.metadata["source"] = f

            new_chunks.extend(splitter.split_documents(docs))

        except Exception as e:
            print(f"[ERROR] Failed processing {f}: {e}")

    return new_chunks


# -------------------
# Vector DB rebuild
# -------------------
def rebuild_vector_db(
    all_docs: List[Any],
    embeddings: HuggingFaceEmbeddings,
    index_path: str
):
    """
    Rebuild or clear the FAISS vector database.
    """
    os.makedirs(index_path, exist_ok=True)

    try:
        if all_docs:
            vectordb = FAISS.from_documents(all_docs, embeddings)
            vectordb.save_local(index_path)
            print("[INFO] Vector DB rebuilt successfully")
        else:
            print("[INFO] No documents left, clearing vector DB")
            for f in ["index.faiss", "index.pkl"]:
                fp = os.path.join(index_path, f)
                if os.path.exists(fp):
                    os.remove(fp)

    except Exception as e:
        print(f"[ERROR] Failed to rebuild vector DB: {e}")


# -------------------
# State persistence
# -------------------
def persist_state(
    docs_file: str,
    metadata_file: str,
    cache_file: str,
    all_docs: List[Any],
    metadata: Dict[str, str],
    qa_cache: Dict[str, Any],
):
    """
    Save document chunks, metadata, and QA cache to disk safely.
    """
    for path, data, name in [
        (docs_file, all_docs, "documents"),
        (metadata_file, metadata, "metadata"),
        (cache_file, qa_cache, "QA cache"),
    ]:
        success, msg = save_binary(path, data)
        if not success:
            print(f"[WARN] Failed to save {name}: {msg}")


# -------------------
# Main orchestrator
# -------------------
def sync_vector_db(
    documents_path: str,
    index_path: str,
    embedding_model: str = DEFAULT_CONFIG["embedding_model"],
    chunk_size: int = DEFAULT_CONFIG["chunk_size"],
    chunk_overlap: int = DEFAULT_CONFIG["chunk_overlap"],
    chunking_strategy: str = DEFAULT_CONFIG.get("chunking_strategy", "recursive"),
) -> Tuple[List[Any], Dict[str, str], Dict[str, Any], str]:
    """
    Synchronize a FAISS vector database with a directory of PDF documents.

    This function performs incremental updates to the vector database by:
        - Detecting new or modified PDF files (via file hashing)
        - Removing deleted files and their associated chunks
        - Reprocessing only updated documents
        - Rebuilding the FAISS index when changes occur

    It also maintains:
        - Metadata (file -> hash mapping)
        - QA cache (question-answer pairs tied to document sources)
        - Persisted document chunks

    Supports configurable chunking strategies for document preprocessing.

    Args:
        documents_path (str): Path to the directory containing PDF documents.
        index_path (str): Directory where FAISS index and state files are stored.
        embedding_model (str): HuggingFace embedding model name.
        chunk_size (int): Maximum size of each chunk.
        chunk_overlap (int): Overlap between consecutive chunks.
        chunking_strategy (str): Chunking strategy to use.
            Options:
                - "recursive": Character-based splitting (default)
                - "token": Token-based splitting
                - "semantic": Embedding-based semantic chunking

    Returns:
        Tuple containing:
            - all_docs (List[Any]): List of all document chunks currently indexed
            - metadata (Dict[str, str]): Mapping of filename to file hash
            - qa_cache (Dict[str, Any]): Cached QA responses
            - message (str): Status message describing the operation result

    Raises:
        ValueError: If an unsupported chunking strategy is provided
        ImportError: If semantic chunking is used without required dependencies
    """
    first_run = not os.path.exists(index_path)
    os.makedirs(index_path, exist_ok=True)

    metadata_file = os.path.join(index_path, "metadata.bin")
    cache_file = os.path.join(index_path, "qa_cache.bin")
    docs_file = os.path.join(index_path, "documents.bin")

    metadata, qa_cache, all_docs = load_state(
        metadata_file, cache_file, docs_file, first_run=first_run
    )

    # Configurable splitter
    splitter = get_text_splitter(
        strategy=chunking_strategy,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model
    )
    print(f"[INFO] Using chunking strategy: {chunking_strategy}")

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    current_files = get_current_pdf_files(documents_path)

    # Handle removals & updates
    all_docs, qa_cache, metadata, removed_flag = handle_removed_files(
        metadata, current_files, all_docs, qa_cache
    )
    updated_files, metadata, updated_flag = detect_updated_files(
        documents_path, current_files, metadata
    )

    # First run
    if first_run and current_files:
        new_chunks = process_new_files(documents_path, list(current_files), splitter)
        all_docs.extend(new_chunks)
        rebuild_vector_db(all_docs, embeddings, index_path)
        persist_state(docs_file, metadata_file, cache_file, all_docs, metadata, qa_cache)
        message = "No previous DB found. Created new vector DB with available documents."
        print(f"[INFO] {message}")
        return all_docs, metadata, qa_cache, message

    # Nothing changed
    if not removed_flag and not updated_flag:
        message = "No changes detected: Vector DB remains unchanged."
        print(f"[INFO] {message}")
        return all_docs, metadata, qa_cache, message

    # Process updates
    all_docs = remove_old_chunks(all_docs, updated_files)
    new_chunks = process_new_files(documents_path, updated_files, splitter)
    all_docs.extend(new_chunks)

    rebuild_vector_db(all_docs, embeddings, index_path)
    persist_state(docs_file, metadata_file, cache_file, all_docs, metadata, qa_cache)

    message = "Vector DB synchronized successfully with new/updated documents."
    print(f"[INFO] {message}")
    return all_docs, metadata, qa_cache, message