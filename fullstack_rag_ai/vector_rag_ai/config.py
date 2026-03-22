DEFAULT_CONFIG = {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model": "llama3",
    "chunk_size": 500,
    "chunk_overlap": 100,
    "k": 15,
    "chunking_strategy": "recursive",
    "prompt_template": """
                        You are a helpful assistant. Use the context below to answer the question. 
                        Combine information, reason, and infer if needed. 
                        If answer is not in context, say "I could not find the answer in the documents."

                        Context:
                        {context}

                        Question:
                        {question}
                        """
}