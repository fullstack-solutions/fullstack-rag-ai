# qa_service.py
import hashlib
from typing import List

from langchain_ollama import ChatOllama

from .helpers import FileUtils


class QAService:
    def __init__(
        self,
        vector_db=None,
        graph_db=None,
        model="llama3",
        k=10,
        cache_path="qa_cache.bin",
        debug=False,
        prompt_template=None,
    ):
        self.vector_db = vector_db
        self.graph_db = graph_db

        self.model = model
        self.k = k
        self.debug = debug
        self.cache_path = cache_path
        self.prompt_template = prompt_template

    # -------------------
    # CACHE KEY
    # -------------------
    def compute_cache_key(self, question: str, docs: List[str]) -> str:
        context_hash = hashlib.md5("".join(docs).encode()).hexdigest()
        return hashlib.md5((question + context_hash).encode()).hexdigest()

    # -------------------
    # RETRIEVE (HYBRID VECTOR + GRAPH)
    # -------------------
    def retrieve(self, question: str) -> List[str]:
        results: List[str] = []

        # Vector DB search
        if self.vector_db:
            try:
                vector_results = self.vector_db.search(question, self.k)
                if self.debug:
                    print(f"[DEBUG] Vector DB returned {len(vector_results)} docs")
                results.extend(vector_results)
            except Exception as e:
                if self.debug:
                    print(f"[ERROR] Vector DB search failed: {e}")

        # Graph DB search
        if self.graph_db:
            try:
                graph_results = self.graph_db.search(question, self.k)
                if self.debug:
                    print(f"[DEBUG] Graph DB returned {len(graph_results)} docs")
                results.extend(graph_results)
            except Exception as e:
                if self.debug:
                    print(f"[ERROR] Graph DB search failed: {e}")

        # Deduplicate while preserving order
        seen = set()
        unique_results = []
        for r in results:
            if r not in seen:
                seen.add(r)
                unique_results.append(r)

        # Limit to top-k results
        return unique_results[:self.k]

    # -------------------
    # LLM CALL
    # -------------------
    def generate(self, prompt: str) -> str:
        llm = ChatOllama(model=self.model)
        return llm.invoke(prompt)

    # -------------------
    # MAIN QA ENTRY
    # -------------------
    def ask(self, question: str) -> str:
        docs = self.retrieve(question)

        if not docs:
            return "No relevant information found."

        # Load cache
        success, cache, _ = FileUtils.load_binary(self.cache_path)
        cache = cache or {}

        # Compute key
        key = self.compute_cache_key(question, docs)

        # Return cached answer if available
        if key in cache:
            if self.debug:
                print("[DEBUG] Cache hit")
            return cache[key]["answer"]

        # Build prompt
        context = "\n\n".join(docs)
        prompt = self.prompt_template.format(
            context=context,
            question=question
        )

        # Generate answer
        answer = self.generate(prompt)

        # Save to cache
        cache[key] = {"answer": answer}
        FileUtils.save_binary(self.cache_path, cache)

        if self.debug:
            print("[DEBUG] Answer cached")
        return answer