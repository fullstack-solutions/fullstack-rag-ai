import os
import hashlib
import pickle
from typing import Any, List, Set, Dict
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from .config import DEFAULT_CONFIG
from .utils import load_binary, save_binary

# -----------------------------
# Binary Helpers
# -----------------------------
def load_binary(path: str) -> Dict[Any, Any]:
    """
    Safely load a Python object from a pickle file.

    Args:
        path (str): Path to the pickle file.

    Returns:
        dict: Loaded object if successful, else empty dict.
    """
    try:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
        else:
            print(f"[WARN] File not found: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to load {path}: {e}")
    return {}


def save_binary(path: str, data: Any) -> bool:
    """
    Safely save a Python object to a pickle file.

    Args:
        path (str): Path to save the pickle file.
        data (Any): Object to serialize.

    Returns:
        bool: True if save successful, False otherwise.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save {path}: {e}")
        return False

# -----------------------------
# Cache Key Helper
# -----------------------------
def compute_cache_key(question: str, docs: List[Any]) -> str:
    """
    Generate a deterministic cache key based on the question and document content.

    Args:
        question (str): User query.
        docs (List[Document]): List of document chunks retrieved from vector DB.

    Returns:
        str: MD5 hash representing the cache key.
    """
    context_text = "".join(d.page_content for d in docs)
    context_hash = hashlib.md5(context_text.encode()).hexdigest()
    return hashlib.md5((question + context_hash).encode()).hexdigest()

# -----------------------------
# QA Helper Functions
# -----------------------------
def retrieve_documents(
    index_path: str,
    question: str,
    embedding_model: str,
    k: int,
    debug: bool
) -> List[Any]:
    """
    Retrieve the top-k most relevant documents from a FAISS vector database.

    Args:
        index_path (str): Path to FAISS vector DB.
        question (str): User query.
        embedding_model (str): HuggingFace embedding model name.
        k (int): Number of top chunks to retrieve.
        debug (bool): If True, prints debug info including retrieved sources.

    Returns:
        List[Document]: Retrieved document chunks. Empty if index missing or error occurs.
    """
    if not os.path.exists(index_path):
        print(f"[WARN] FAISS index path does not exist: {index_path}")
        return []

    try:
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        vectordb = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        docs = vectordb.similarity_search(question, k=k)

        if debug:
            print("[DEBUG] Retrieved sources:", [d.metadata.get("source", "unknown") for d in docs])
            for d in docs:
                print(d.page_content[:200], "\n---")

        return docs
    except Exception as e:
        print(f"[ERROR] Failed to retrieve documents from FAISS: {e}")
        return []

def generate_answer(prompt: str, model: str) -> str:
    """
    Generate an answer from the LLM using the given prompt.

    Args:
        prompt (str): Context + question for the LLM.
        model (str): Ollama model name.

    Returns:
        str: Generated answer, or error message if LLM fails.
    """
    try:
        llm = ChatOllama(model=model)
        return llm.invoke(prompt)
    except Exception as e:
        print(f"[ERROR] LLM failed to generate answer: {e}")
        return "Failed to generate answer due to LLM error."

# -----------------------------
# Main QA Function
# -----------------------------
def ask_question(
    question: str,
    index_path: str,
    model: str = DEFAULT_CONFIG["llm_model"],
    embedding_model: str = DEFAULT_CONFIG["embedding_model"],
    k: int = DEFAULT_CONFIG["k"],
    debug: bool = False,
    prompt_template: str = DEFAULT_CONFIG["prompt_template"],
) -> str:
    """
    Ask a question using a FAISS vector database and optional QA cache.
    Returns cached answers if available and sources unchanged.

    Args:
        question (str): User query.
        index_path (str): Path to vector DB and QA cache.
        model (str): Ollama LLM model name.
        embedding_model (str): HuggingFace embedding model name.
        k (int): Number of top chunks to retrieve.
        debug (bool): If True, prints debug info.
        prompt_template (str): Template for constructing the prompt with context and question.

    Returns:
        str: Answer to the question or error message.
    """
    if not os.path.exists(index_path):
        print(f"[WARN] Vector DB path does not exist: {index_path}")
        return "No documents found. Please check your vector DB path."

    # Retrieve top documents
    docs = retrieve_documents(index_path, question, embedding_model, k, debug)
    if not docs:
        return "No relevant information found in the documents."

    # Load QA cache
    cache_file = os.path.join(index_path, "qa_cache.bin")
    qa_cache = load_binary(cache_file)

    # Compute cache key
    key = compute_cache_key(question, docs)
    source_files: Set[str] = {d.metadata.get("source") for d in docs}

    # Return cached answer if sources unchanged
    if key in qa_cache:
        cached_sources = set(qa_cache[key].get("sources", []))
        if cached_sources == source_files:
            if debug:
                print("[INFO] Returning cached answer")
            return qa_cache[key].get("answer", "Cached answer missing.")

    # Build context and prompt
    context = "\n\n".join(d.page_content for d in docs)
    prompt = prompt_template.format(context=context, question=question)

    # Generate answer from LLM
    answer = generate_answer(prompt, model)

    # Save answer in QA cache
    qa_cache[key] = {"answer": answer, "sources": list(source_files)}
    if not save_binary(cache_file, qa_cache):
        print(f"[WARN] Could not update QA cache at {cache_file}")

    return answer