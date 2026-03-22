from typing import List
from vectorless_rag_ai.types import DocumentChunk, VectorlessConfig
from vectorless_rag_ai.indexing.bm25_index import BM25Index
from vectorless_rag_ai.retrieval.bm25_retriever import BM25Retriever
from vectorless_rag_ai.query.query_processor import QueryProcessor
from vectorless_rag_ai.context.context_builder import ContextBuilder

class VectorlessRAGPipeline:
    """
    Modular vectorless RAG pipeline.
    Supports:
    - BM25 retrieval
    - Optional query expansion
    - Optional LLM reranking
    - Incremental document add/delete
    - Query caching
    - Configurable parameters
    """
    def __init__(self, config: VectorlessConfig = VectorlessConfig(), reranker=None, query_expander=None, cache_enabled=True):
        self.config = config

        # Core modules
        self.index = BM25Index(k1=config.k1, b=config.b)
        self.retriever = BM25Retriever(self.index)
        self.query_processor = QueryProcessor()
        self.context_builder = ContextBuilder()

        # Optional modules
        self.reranker = reranker
        self.query_expander = query_expander

        # Cache
        self.cache_enabled = cache_enabled
        self.cache = {}

    # -------------------------
    # Document Management
    # -------------------------
    def add_documents(self, docs: List[DocumentChunk]):
        self.index.add_documents(docs)
        if self.cache_enabled:
            self.cache.clear()

    def delete_documents(self, doc_ids: List[str]):
        self.index.delete_documents(doc_ids)
        if self.cache_enabled:
            self.cache.clear()

    # -------------------------
    # Run Query
    # -------------------------
    def run(self, query: str):
        if self.cache_enabled and query in self.cache:
            return self.cache[query]

        processed_query = self.query_processor.process(query)

        # Query expansion
        queries = self.query_expander.expand(processed_query) if self.query_expander else [processed_query]

        # Retrieve
        all_results = []
        for q in queries:
            results = self.retriever.retrieve(q, self.config.top_k)
            all_results.extend(results)

        # Deduplicate
        seen = {}
        for r in all_results:
            if r.chunk.id not in seen or r.score > seen[r.chunk.id].score:
                seen[r.chunk.id] = r
        merged_results = list(seen.values())

        # Rerank
        if self.reranker:
            merged_results = self.reranker.rerank(query, merged_results)
        else:
            merged_results = sorted(merged_results, key=lambda x: x.score, reverse=True)

        # Build context
        context = self.context_builder.build(merged_results, self.config.max_context_length)

        response = {
            "query": query,
            "expanded_queries": queries,
            "context": context,
            "results": merged_results,
        }

        # Cache
        if self.cache_enabled:
            self.cache[query] = response

        return response