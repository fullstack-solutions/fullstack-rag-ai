# config.py

DEFAULT_CONFIG = {
    # -------------------------
    # BM25 / retrieval settings
    # -------------------------
    "top_k": 5,  # Number of top documents to retrieve
    "max_context_length": 2000,  # Maximum characters to include in context
    "k1": 1.5,  # BM25 parameter: term frequency scaling
    "b": 0.75,  # BM25 parameter: document length normalization

    # -------------------------
    # Prompts for optional LLM modules
    # -------------------------
    "reranker_prompt": """
        You are a ranking system.
        
        Query:
        {query}

        Documents:
        {documents}

        Return ONLY a list of indices (most relevant first).
        Example: 2,0,1
    """,

    "query_expander_prompt": """
        Expand the following query into 3 alternative search queries.

        Query: {query}

        Return as a comma-separated list.
    """,

    "default_prompt": """
        Question: {question}

        Context:
        {context}
    """
}