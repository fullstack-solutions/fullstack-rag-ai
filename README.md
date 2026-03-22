# Fullstack RAG AI

Lightweight Retrieval-Augmented Generation (RAG) library for building internal knowledge systems using local documents and LLMs.

Supports:

1. Vector-based retrieval using FAISS and embeddings
2. Vectorless retrieval using BM25

Optional LLM integration allows reranking, query expansion, and dynamic prompt responses.

# Features
- Load documents from PDFs or raw text
- Automatic chunking (per-page or whole PDF)
- Vector-based retrieval with FAISS + embeddings
- Vectorless retrieval with BM25
- Optional LLM reranking and query expansion
- Customizable prompts for LLMs
- Deterministic caching to avoid repeated LLM calls
- Automatic detection of document updates, additions, and deletions
- Fully modular: swap retriever, reranker, or query expander

Supported tools:

- Ollama LLMs
- HuggingFace embeddings
- FAISS vector database

---

# Installation

```bash
pip install fullstack-rag-ai
```

# Default Configuration

The library has two separate default configurations depending on the retrieval method.

1. Vector-Based (FAISS + Embeddings)
```Python
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
```
2. Vectorless BM25

```Python
DEFAULT_CONFIG_BM25 = {
    "top_k": 5,
    "max_context_length": 2000,
    "k1": 1.5,  # BM25 parameter
    "b": 0.75,  # BM25 parameter
    "reranker_prompt": """
You are a ranking system.
Query:
{query}
Documents:
{documents}
Return ONLY a list of indices (most relevant first)
""",
    "query_expander_prompt": """
Expand the following query into 3 alternative search queries.
Query: {query}
Return as a comma-separated list.
"""
}
```
- top_k: number of documents retrieved per query
- max_context_length: max context length for LLM
- k1 & b: BM25 scoring parameters
# Core Workflow

## Vector-Based (FAISS + Embeddings)
```
Documents → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer
```
Vectorless (BM25)
```
Documents → Chunking → BM25 Index → Retrieval → Optional LLM Rerank → Context → LLM Answer
```
# Vector-Based FAISS Pipeline


# Vector Database Synchronization

```Python
from fullstack_rag_lib.vector_rag_ai import VectorDBSynchronizer, QAService

# Step 1: Build or update vector database
syncer = VectorDBSynchronizer(
    documents_path="./data",
    index_path="./vector_db",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=600,
    chunk_overlap=120,
    chunking_strategy="semantic"
)
all_docs, metadata, qa_cache, message = syncer.sync()
print(message)

# Step 2: Ask questions
qa = QAService(index_path="./vector_db", model="llama3", k=15, debug=True)
answer = qa.ask("What is FAISS used for?")
print(answer)
```
- Supports custom prompt templates:

```Python
custom_prompt = """
You are a technical assistant. Use the context to answer concisely.
If answer is not in context, say 'Not found'.
Context:
{context}
Question:
{question}
"""
qa = QAService(index_path="./vector_db", prompt_template=custom_prompt)
```
# Vectorless BM25 Pipeline

```Python
from fullstack_rag_ai.vectorless_rag_ai import VectorlessRAGPipeline
from fullstack_rag_ai.vectorless_rag_ai import load_pdfs
from fullstack_rag_ai.vectorless_rag_ai import VectorlessConfig
from fullstack_rag_ai.vectorless_rag_ai import LLMReranker
from fullstack_rag_ai.vectorless_rag_ai import QueryExpander

# Load documents
documents = load_pdfs("./data", chunk_pages=True)

# Initialize pipeline
pipeline = VectorlessRAGPipeline(
    config=VectorlessConfig(top_k=5, max_context_length=2000)
)
pipeline.add_documents(documents)

# Optional LLM reranker
def my_llm(prompt: str) -> str:
    return "0,1,2"  # Replace with actual LLM call
pipeline.set_reranker(LLMReranker(my_llm))

# Optional query expansion
pipeline.set_query_expander(QueryExpander(my_llm))

# Run query
result = pipeline.run("tell me about ec2 instance m7a.medium cost")
print(result["prompt"])
print([r.chunk.id for r in result["results"]])
```
- Works without embeddings or FAISS
- Fully modular: swap retriever, reranker, or query expander dynamically
- Cache results with pipeline.clear_cache()

# Changing the Embedding Model

Embeddings:

```Python
syncer = VectorDBSynchronizer(
    documents_path="./data",
    index_path="./vector_db",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
all_docs, metadata, qa_cache, message = syncer.sync()
```
LLM: 

```Python
qa = QAService(
    index_path="./vector_db",
    model="llama3:8b",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```
Any Ollama or locally available LLM can be used
To install a model:
```bash
ollama pull llama3
ollama pull llama3:8b
ollama pull llama3:70b
ollama pull phi3
ollama pull mistral
```

# Rebuilding Vector Database
If chunking or embedding model changes, rebuild in a new path:

```Python
syncer = VectorDBSynchronizer(
    documents_path="./data",
    index_path="./vector_new_db",
    embedding_model="BAAI/bge-base-en",
    chunk_strategy="token",
    chunk_size=600,
    chunk_overlap=120
)
all_docs, metadata, qa_cache, message = syncer.sync()
```
# Deterministic Caching
Cache keys: question + retrieved context hash
If question and context are unchanged, cached answer is returned
Stored in qa_cache.bin

# Dependencies

- Python >= 3.9
- pypdf (PDF loading)
- faiss-cpu (vector-based retrieval, optional if using BM25)
- sentence-transformers (embedding models, optional if using BM25)
- langchain and related packages for LLM integration:
    - langchain
    - langchain-community
    - langchain-huggingface
    - langchain-ollama
    - langchain-experimental (optional)

Supports dynamic usage of:
- Any HuggingFace embedding model
- Any Ollama LLM model

# License
MIT License

# Author
Fullstack-Solutions