# Fullstack RAG AI

Lightweight Retrieval-Augmented Generation (RAG) library for building internal knowledge systems using local documents and LLMs.

This library allows you to:

- Load documents from PDFs or raw text
- Automatically chunk and embed documents (supports multiple chunking strategies)
- Store embeddings in a FAISS vector database
- Ask questions against the knowledge base
- Use deterministic caching to avoid repeated LLM calls
- Automatically detect document updates, additions, and deletions
- Customize prompts dynamically for the LLM

It is designed to use:

- Ollama LLMs
- HuggingFace embeddings
- FAISS vector database

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
# Loading Documents (Class-Based)

Documents are loaded using the DocumentLoader class, which can combine multiple sources:

- PDFs from a folder
- Raw text strings

## Load From PDFs
```Python
from fullstack_rag_lib import DocumentLoader

# Load all PDFs from a folder
loader = DocumentLoader(path="./documents")
docs = loader.load()
print(f"Loaded {len(docs)} documents from PDFs")
```
## Load From Text List
```Python
from fullstack_rag_lib import DocumentLoader

texts = [
    "Python is a programming language",
    "FAISS is used for vector similarity search"
]

loader = DocumentLoader(texts=texts)
docs = loader.load()
print(f"Loaded {len(docs)} documents from text")
```
## Load From Both PDFs and Texts
```Python
loader = DocumentLoader(
    path="./documents",
    texts=["Quick guide to FAISS", "Another text document"]
)
docs = loader.load()
print(f"Loaded {len(docs)} documents in total")
```
## Notes:
- PDFDocumentLoader: Loads PDFs, adds the filename as source metadata for each document.
- TextDocumentLoader: Converts a list of text strings into LangChain Document objects.
- DocumentLoader: Combines both sources for a unified list of Document.


# Vector Database Synchronization

The VectorDBSynchronizer class handles building and updating the FAISS vector database.

- Detects new, updated, or removed files
- Handles chunking automatically
- Saves state and cache files (metadata.bin, qa_cache.bin, documents.bin)


```Python
from fullstack_rag_lib import VectorDBSynchronizer

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
```
Notes:

- documents_path: Folder containing PDFs
- index_path: Path to store FAISS index and internal files
- chunking_strategy: "recursive", "token", or "semantic"

Asking Questions:

Use the QAService class for retrieval and LLM-based QA:

```Python
from fullstack_rag_lib import QAService

qa = QAService(
    index_path="./vector_db",
    model="llama3",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    k=15,
    debug=True
)

answer = qa.ask("What is FAISS used for?")
print(answer)
```
You can also pass a custom prompt template:

```Python
custom_prompt = """
You are a highly technical assistant. Use the context to answer concisely.
If the answer is not in the context, say 'Not found'.
Context:
{context}
Question:
{question}
"""

qa = QAService(
    index_path="./vector_db",
    prompt_template=custom_prompt
)

answer = qa.ask("Explain the onboarding process")
print(answer)
```

# Deterministic QA Cache

The library includes a smart cache system that prevents unnecessary LLM calls.

Cache keys are generated using: question + retrieved context hash

If:
- the question is the same

- the retrieved documents have not changed

The answer is returned directly from cache.

# Deterministic QA Cache
- Cache keys: question + retrieved context hash
- If question and context are unchanged, returns cached answer
- Stored in qa_cache.bin

# Example Full Pipeline

```Python
from fullstack_rag_lib import VectorDBSynchronizer
from fullstack_rag_lib import QAService

# Step 1: Build/update vector DB
syncer = VectorDBSynchronizer(
    documents_path="./data",
    index_path="./vector_db",
    chunk_size=600,
    chunk_overlap=120,
    chunking_strategy="semantic"
)
all_docs, metadata, qa_cache, message = syncer.sync()
print(message)

# Step 2: Ask questions
qa = QAService(index_path="./vector_db", debug=True)
answer = qa.ask("Summarize the company onboarding process")
print(answer)
```

# Changing the Embedding Model

Any HuggingFace embedding model supported by sentence-transformers can be used.

Example:
```Python
syncer = VectorDBSynchronizer(
    documents_path="./data",
    index_path="./vector_db",
    embedding_model="sentence-transformers/all-mpnet-base-v2"
)
all_docs, metadata, qa_cache, message = syncer.sync()
```
The library will automatically download the model through HuggingFace if it is not already installed.

# Changing the LLM Model

The LLM model can be changed to any model available in Ollama.

Example:
```Python
qa = QAService(
    index_path="./vector_db",
    model="llama3:8b",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)

answer = qa.ask("Explain the onboarding process")
print(answer)
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

If chunking parameters or embedding model changes, rebuild the vector DB in a new path or remove the old one.

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

# Dependencies

- langchain
- langchain-community
- langchain-huggingface
- langchain-ollama
- faiss-cpu
- sentence-transformers
- pypdf
- langchain-experimental

Supports dynamic usage of:

- Any HuggingFace embedding model
- Any Ollama LLM model

# License
MIT License

# Author
Fullstack-Solutions