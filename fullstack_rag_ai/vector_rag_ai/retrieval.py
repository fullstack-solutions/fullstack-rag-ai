import os
import hashlib
from typing import Any, List, Set

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from .config import DEFAULT_CONFIG
from .helpers import FileUtils


class QAService:
    """
    Handles document retrieval, caching, and LLM-based QA.
    """

    def __init__(
        self,
        index_path: str,
        model: str = DEFAULT_CONFIG["llm_model"],
        embedding_model: str = DEFAULT_CONFIG["embedding_model"],
        k: int = DEFAULT_CONFIG["k"],
        debug: bool = False,
        prompt_template: str = DEFAULT_CONFIG["prompt_template"],
    ):
        self.index_path = index_path
        self.model = model
        self.embedding_model = embedding_model
        self.k = k
        self.debug = debug
        self.prompt_template = prompt_template

        self.cache_file = os.path.join(index_path, "qa_cache.bin")

    # -----------------------------
    # Cache Key
    # -----------------------------
    @staticmethod
    def compute_cache_key(question: str, docs: List[Any]) -> str:
        context_text = "".join(d.page_content for d in docs)
        context_hash = hashlib.md5(context_text.encode()).hexdigest()
        return hashlib.md5((question + context_hash).encode()).hexdigest()

    # -----------------------------
    # Retrieval
    # -----------------------------
    def retrieve_documents(self, question: str) -> List[Any]:
        if not os.path.exists(self.index_path):
            print(f"[WARN] FAISS index path does not exist: {self.index_path}")
            return []

        try:
            embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
            vectordb = FAISS.load_local(
                self.index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            docs = vectordb.similarity_search(question, k=self.k)

            if self.debug:
                print("[DEBUG] Retrieved sources:",
                      [d.metadata.get("source", "unknown") for d in docs])
                for d in docs:
                    print(d.page_content[:200], "\n---")

            return docs

        except Exception as e:
            print(f"[ERROR] Failed to retrieve documents from FAISS: {e}")
            return []

    # -----------------------------
    # LLM
    # -----------------------------
    def generate_answer(self, prompt: str) -> str:
        try:
            llm = ChatOllama(model=self.model)
            return llm.invoke(prompt)
        except Exception as e:
            print(f"[ERROR] LLM failed to generate answer: {e}")
            return "Failed to generate answer due to LLM error."

    # -----------------------------
    # Main QA
    # -----------------------------
    def ask(self, question: str) -> str:
        if not os.path.exists(self.index_path):
            print(f"[WARN] Vector DB path does not exist: {self.index_path}")
            return "No documents found. Please check your vector DB path."

        # Retrieve documents
        docs = self.retrieve_documents(question)
        if not docs:
            return "No relevant information found in the documents."

        # Load QA cache using FileUtils
        success, qa_cache, msg = FileUtils.load_binary(self.cache_file)
        if not success:
            if self.debug:
                print(f"[WARN] Failed to load QA cache: {msg}")
            qa_cache = {}

        # Compute cache key
        key = self.compute_cache_key(question, docs)
        source_files: Set[str] = {d.metadata.get("source") for d in docs}

        # Return cached answer if sources unchanged
        if key in qa_cache:
            cached_sources = set(qa_cache[key].get("sources", []))
            if cached_sources == source_files:
                if self.debug:
                    print("[INFO] Returning cached answer")
                return qa_cache[key].get("answer", "Cached answer missing.")

        # Build context + prompt
        context = "\n\n".join(d.page_content for d in docs)
        prompt = self.prompt_template.format(context=context, question=question)

        # Generate answer
        answer = self.generate_answer(prompt)

        # Save cache using FileUtils
        qa_cache[key] = {"answer": answer, "sources": list(source_files)}
        success, msg = FileUtils.save_binary(self.cache_file, qa_cache)
        if not success and self.debug:
            print(f"[WARN] Could not update QA cache: {msg}")

        return answer