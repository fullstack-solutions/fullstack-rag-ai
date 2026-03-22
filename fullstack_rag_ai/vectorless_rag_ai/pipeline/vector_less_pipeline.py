from vectorless_rag_ai.types import DocumentChunk, VectorlessConfig
from vectorless_rag_ai.indexing.bm25_index import BM25Index
from vectorless_rag_ai.retrieval.bm25_retriever import BM25Retriever
from vectorless_rag_ai.query.query_processor import QueryProcessor
from vectorless_rag_ai.context.context_builder import ContextBuilder


class VectorlessRAGPipeline:
    def __init__(
        self,
        config: VectorlessConfig = VectorlessConfig(),
        reranker=None,
        query_expander=None,
    ):
        self.config = config

        self.index = BM25Index()
        self.retriever = BM25Retriever(self.index)
        self.query_processor = QueryProcessor()
        self.context_builder = ContextBuilder()

        # Optional modules
        self.reranker = reranker
        self.query_expander = query_expander

    def add_documents(self, docs: list[DocumentChunk]):
        self.index.add_documents(docs)

    def run(self, query: str):
        processed_query = self.query_processor.process(query)

        # Query Expansion
        queries = (
            self.query_expander.expand(processed_query)
            if self.query_expander
            else [processed_query]
        )

        # Retrieve from all expanded queries
        all_results = []
        for q in queries:
            results = self.retriever.retrieve(q, self.config.top_k)
            all_results.extend(results)

        # Deduplicate by chunk id
        seen = {}
        for r in all_results:
            if r.chunk.id not in seen or r.score > seen[r.chunk.id].score:
                seen[r.chunk.id] = r

        merged_results = list(seen.values())

        # LLM Reranking
        if self.reranker:
            merged_results = self.reranker.rerank(query, merged_results)

        else:
            merged_results = sorted(
                merged_results, key=lambda x: x.score, reverse=True
            )

        context = self.context_builder.build(
            merged_results,
            self.config.max_context_length
        )

        return {
            "query": query,
            "expanded_queries": queries,
            "context": context,
            "results": merged_results,
        }