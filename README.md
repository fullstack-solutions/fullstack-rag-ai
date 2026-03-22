# Fullstack RAG AI

Lightweight **Retrieval-Augmented Generation (RAG)** library for building internal knowledge systems using local documents and LLMs.

This library allows you to:

- Load documents from PDFs or raw text
- Automatically chunk and embed documents (supports multiple chunking strategies)
- Store embeddings in a FAISS vector database
- Ask questions against the knowledge base
- Use deterministic caching to avoid repeated LLM calls
- Automatically detect document updates, additions, and deletions
- Customize prompts dynamically for the LLM

It is designed to use:

- **Ollama LLMs**
- **HuggingFace embeddings**
- **FAISS vector database**

---

# Installation

```bash
pip install fullstack-rag-ai
```

# Default Configuration

The library uses a default configuration, but all parameters can be overridden by the user.

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
# Core Workflow

Typical usage flow:

- Load documents
- Build or update vector database
- Ask questions against the database

```
Documents → Chunking → Embeddings → FAISS → Retrieval → LLM → Answer
```
# Loading Documents

Documents can be loaded either from PDF files or from a list of text strings.

## Load From PDFs
```Python
from fullstack_rag_lib.ingestion import load_documents

docs = load_documents(path="documents/")
```
## Load From Text List
```Python
from fullstack_rag_lib.ingestion import load_documents

texts = [
    "Python is a programming language",
    "FAISS is used for vector similarity search"
]

docs = load_documents(texts=texts)
```
## Function
```Python
load_documents(path=None, texts=None)
```
| Parameter | Type      | Description                 |
| --------- | --------- | --------------------------- |
| path      | str       | Folder containing PDF files |
| texts     | List[str] | List of raw text strings    |


Returns: List[Document]

# Updating the Vector Database

This function builds or updates the FAISS vector database.

It automatically handles:
- New documents
- Updated documents
- Deleted documents
- Custom chunking strategy (recursive, token, semantic)

Only the necessary parts of the database are rebuilt.

```Python
from fullstack_rag_lib.vector_db import sync_vector_db

sync_vector_db(
    documents_path="./data",
    index_path="./vector_db",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    chunk_size=600,
    chunk_overlap=120,
    chunk_strategy="semantic",
)
```

## Function

```Python
sync_vector_db(
    documents_path=documents_path,
    index_path=index_path,
    embedding_model=embedding_model,
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    chunk_strategy=chunk_strategy
)
```
| Parameter       | Type | Description                                         |
| --------------- | ---- | --------------------------------------------------- |
| documents_path  | str  | Directory containing PDF files                      |
| index_path      | str  | Directory where FAISS index is stored               |
| embedding_model | str  | HuggingFace embedding model                         |
| chunk_size      | int  | Size of one chunk                                   |
| chunk_overlap   | int  | Overlap between chunks                              |
| chunk_strategy  | str  | Chunking strategy: 'recursive', 'token', 'semantic' |


This function maintains the following internal files:
index.faiss
index.pkl
documents.bin
metadata.bin
qa_cache.bin

# Asking Questions

Once the vector database exists, you can query it using an LLM.
The system retrieves the most relevant chunks and sends them to the LLM as context.

You can now also pass a custom prompt template dynamically:

```Python
from fullstack_rag_lib.qa import ask_question

custom_prompt = """
You are a highly technical assistant. Use the context to answer concisely.
If the answer is not in the context, say 'Not found'.
Context:
{context}
Question:
{question}
"""

answer = ask_question(
    question="What is FAISS used for?",
    index_path="./vector_db",
    prompt_template=custom_prompt
)

print(answer)
```
## Function

```Python
ask_question(
    question,
    index_path,
    model=model,
    embedding_model=embedding_model,
    k=k,
    debug=False,
    prompt_template=None
)
```

| Parameter       | Type | Description                     |
| --------------- | ---- | ------------------------------- |
| question        | str  | User question                   |
| index_path      | str  | Path to vector database         |
| model           | str  | Ollama LLM model                |
| embedding_model | str  | HuggingFace embedding model     |
| k               | int  | Number of chunks retrieved      |
| debug           | bool | Print retrieved chunks          |
| prompt_template | str  | Optional custom prompt template |


Returns: str

# Deterministic QA Cache

The library includes a smart cache system that prevents unnecessary LLM calls.

Cache keys are generated using: question + retrieved context hash

If:
- the question is the same

- the retrieved documents have not changed

The answer is returned directly from cache.

## Cache File

qa_cache.bin

# Helper Functions

## load_binary

Loads a binary pickle file.
```Python
load_binary(path)
```

Returns empty dictionary if file does not exist.

## save_binary

Stores data as a binary pickle file.

```Python
save_binary(path, data)
```

## compute_cache_key

Creates a deterministic key for caching LLM responses.

```Python
compute_cache_key(question, docs)
```

Uses:
- Question text
- Retrieved document content

# Example Full Pipeline

```Python
from fullstack_rag_lib.vector_db import sync_vector_db
from fullstack_rag_lib.qa import ask_question

# Step 1: Build / Update Vector DB
sync_vector_db(
    documents_path="./data",
    index_path="./vector_db",
    chunk_strategy="semantic",
    chunk_size=600,
    chunk_overlap=120
)

# Step 2: Ask Questions
answer = ask_question(
    question="Summarize the company onboarding process",
    index_path="./vector_db",
)

print(answer)
```

# Changing the Embedding Model

Any HuggingFace embedding model supported by sentence-transformers can be used.

Example:
```Python
sync_vector_db(
    documents_path="./data",
    index_path="./vector_db",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
```
The library will automatically download the model through HuggingFace if it is not already installed.

# Changing the LLM Model

The LLM model can be changed to any model available in Ollama.

Example:
```Python
answer = ask_question(
    question="Explain the onboarding process",
    index_path="./vector_db",
    model="llama3:8b",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```
Other examples:
- model="mistral"
- model="phi3"
- model="llama3:70b"

The only requirement is that the model must be available locally in Ollama.

To install a model:
```bash
ollama pull llama3
ollama pull llama3:8b
ollama pull llama3:70b
ollama pull phi3
ollama pull mistral
```

# Important: Rebuilding the Vector Database

If the embedding model or chunking parameters change, the vector database must be rebuilt removing the older database or creating new database path.

This is because embeddings from different models are not compatible.

Example workflow:
```Python
sync_vector_db(
    documents_path="./data",
    index_path="./vector_new_db",
    embedding_model="BAAI/bge-base-en",
    chunk_strategy="token",
    chunk_size=600,
    chunk_overlap=120
)
```

# Dependencies

- langchain
- langchain-community
- langchain-huggingface
- langchain-ollama
- faiss-cpu
- sentence-transformers
- pypdf
- langchain-experimental

These libraries allow dynamic usage of:
- Any HuggingFace embedding model
- Any Ollama LLM model


# License
MIT License

# Author
Fullstack Solutions