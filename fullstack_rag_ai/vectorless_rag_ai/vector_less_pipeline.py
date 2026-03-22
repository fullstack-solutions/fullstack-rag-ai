# vectorless_pipeline.py

from typing import List, Optional
from .types import DocumentChunk, VectorlessConfig, RetrievalResult
from .bm25_index import BM25Index
from .bm25_retriever import BM25Retriever
from .query_processor import QueryProcessor
from .context_builder import ContextBuilder
from .config import DEFAULT_CONFIG

class VectorlessRAGPipeline:
    """
    Fully modular Vectorless RAG pipeline.

    Features:
        - BM25-based retrieval (default)
        - Optional query expansion via LLM
        - Optional LLM-based reranking
        - Configurable top_k and context length
        - Dynamic module swapping
        - Query caching for repeated queries
    """

    def __init__(
        self,
        config: Optional[VectorlessConfig] = None,
        retriever: Optional[object] = None,
        reranker: Optional[object] = None,
        query_expander: Optional[object] = None,
        context_builder: Optional[ContextBuilder] = None,
        query_processor: Optional[QueryProcessor] = None,
        prompt_template: Optional[str] = None,
        cache_enabled: bool = True
    ):
        """
        Initialize the pipeline.

        Args:
            config: Optional VectorlessConfig object to override defaults.
            retriever: Optional custom retriever module.
            reranker: Optional custom reranker module.
            query_expander: Optional query expander module.
            context_builder: Optional context builder module.
            query_processor: Optional query processor module.
            prompt_template: Optional default prompt template.
            cache_enabled: Whether to cache query results.
        """
        self.config = config or VectorlessConfig(
            top_k=DEFAULT_CONFIG["top_k"],
            max_context_length=DEFAULT_CONFIG["max_context_length"],
            k1=DEFAULT_CONFIG["k1"],
            b=DEFAULT_CONFIG["b"]
        )
        self.cache_enabled = cache_enabled
        self.cache = {}

        self.prompt_template = prompt_template or DEFAULT_CONFIG["default_prompt"]

        self.query_processor = query_processor or QueryProcessor()
        self.context_builder = context_builder or ContextBuilder()

        self.index = BM25Index(k1=self.config.k1, b=self.config.b)
        self.retriever = retriever or BM25Retriever(self.index)
        self.reranker = reranker
        self.query_expander = query_expander

    # -------------------------
    # Document management
    # -------------------------
    def add_documents(self, docs: List[DocumentChunk]):
        """Add documents to the retrieval index and clear cache."""
        self.index.add_documents(docs)
        if self.cache_enabled:
            self.cache.clear()

    def delete_documents(self, doc_ids: List[str]):
        """Delete documents by ID and clear cache."""
        self.index.delete_documents(doc_ids)
        if self.cache_enabled:
            self.cache.clear()

    # -------------------------
    # Query pipeline
    # -------------------------
    def run(self, query: str, use_prompt: Optional[str] = None):
        """
        Run a query through the full RAG pipeline.

        Steps:
            1. Process query
            2. Expand query (if query_expander is set)
            3. Retrieve top-k documents
            4. Rerank results (if reranker is set)
            5. Build context
            6. Format prompt
            7. Return full response dictionary

        Args:
            query: User input query string.
            use_prompt: Optional override prompt template.

        Returns:
            Dict with keys:
                - query: Original query
                - expanded_queries: List of expanded queries
                - context: Concatenated context string
                - results: List of RetrievalResult
                - prompt: Final formatted prompt
        """
        if self.cache_enabled and query in self.cache:
            return self.cache[query]

        # Normalize query
        processed_query = self.query_processor.process(query)

        # Query expansion
        queries = self.query_expander.expand(processed_query) if self.query_expander else [processed_query]

        # Retrieve documents
        all_results: List[RetrievalResult] = []
        for q in queries:
            all_results.extend(self.retriever.retrieve(q, self.config.top_k))

        # Deduplicate by chunk id, keeping highest score
        seen = {}
        for r in all_results:
            if r.chunk.id not in seen or r.score > seen[r.chunk.id].score:
                seen[r.chunk.id] = r
        merged_results = list(seen.values())

        # Rerank
        if self.reranker:
            merged_results = self.reranker.rerank(query, merged_results)
        else:
            merged_results.sort(key=lambda x: x.score, reverse=True)

        # Build context
        context = self.context_builder.build(merged_results, self.config.max_context_length)

        # Format final prompt
        final_prompt = (use_prompt or self.prompt_template).format(context=context, question=query)

        response = {
            "query": query,
            "expanded_queries": queries,
            "context": context,
            "results": merged_results,
            "prompt": final_prompt
        }

        if self.cache_enabled:
            self.cache[query] = response
        return response

    # -------------------------
    # Utilities for dynamic modules
    # -------------------------
    def clear_cache(self): 
        """Clear cached query results."""
        self.cache.clear()

    def set_reranker(self, reranker):
        """Swap reranker dynamically and clear cache."""
        self.reranker = reranker
        self.clear_cache()

    def set_query_expander(self, expander):
        """Swap query expander dynamically and clear cache."""
        self.query_expander = expander
        self.clear_cache()

    def set_retriever(self, retriever):
        """Swap retriever dynamically and clear cache."""
        self.retriever = retriever
        self.clear_cache()

    def set_prompt(self, prompt_template: str):
        """Update the default prompt template and clear cache."""
        self.prompt_template = prompt_template
        self.clear_cache()