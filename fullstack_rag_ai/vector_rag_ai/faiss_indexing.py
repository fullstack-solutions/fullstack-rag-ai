import os
import hashlib
from typing import List, Optional, Dict, Any
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .config import DEFAULT_CONFIG
from ..document_loader import SUPPORTED_EXTENSIONS, IGNORED_DIRS
from .helpers import FileUtils
from .text_splitter import TextSplitterFactory
from ..document_loader import UniversalLoader


class VectorDBSynchronizer:
    """
    Production-ready Vector DB sync engine:
    - Supports file-based documents (local, repos, CSV, etc.)
    - Supports in-memory documents (APIs, GitHub)
    - Incremental updates using content hashes
    - Tracks removed, updated, and new documents
    """

    def __init__(
        self,
        index_path: str,
        documents_path: str = None,
        embedding_model: str = DEFAULT_CONFIG["embedding_model"],
        chunk_size: int = DEFAULT_CONFIG["chunk_size"],
        chunk_overlap: int = DEFAULT_CONFIG["chunk_overlap"],
        chunking_strategy: str = DEFAULT_CONFIG.get("chunking_strategy", "recursive"),
    ):
        self.documents_path = documents_path
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy

        self.loader = UniversalLoader()

        self.metadata_file = os.path.join(index_path, "metadata.bin")
        self.cache_file = os.path.join(index_path, "qa_cache.bin")
        self.docs_file = os.path.join(index_path, "documents.bin")

        os.makedirs(index_path, exist_ok=True)

    # -------------------
    # Helpers
    # -------------------
    def _hash_text(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _hash_file(self, path: str) -> str:
        """Compute hash of file content"""
        with open(path, "rb") as f:
            return hashlib.md5(f.read()).hexdigest()

    def _to_document(self, item: Any, source: Optional[str] = None) -> Document:
        """
        Normalize any input into a LangChain Document.
        Supports:
        - Document -> returned as-is
        - dict -> converted to string content with optional metadata
        - str -> plain text content
        - any other object -> converted to str
        """
        if isinstance(item, Document):
            return item
        elif isinstance(item, dict):
            content = item.get("content") or str(item)
            metadata = item.get("metadata", {})
            metadata["id"] = metadata.get("id") or self._hash_text(content)
            if source:
                metadata["source"] = source
            return Document(page_content=content, metadata=metadata)
        elif isinstance(item, str):
            return Document(page_content=item, metadata={"id": self._hash_text(item), "source": source})
        else:
            content = str(item)
            return Document(page_content=content, metadata={"id": self._hash_text(content), "source": source})

    # -------------------
    # State management
    # -------------------
    def load_state(self, first_run=False):
        success, metadata, msg = FileUtils.load_binary(self.metadata_file)
        if not success and not first_run:
            print(f"[WARN] Failed loading metadata: {msg}")
        metadata = metadata or {}

        success, qa_cache, msg = FileUtils.load_binary(self.cache_file)
        if not success and not first_run:
            print(f"[WARN] Failed loading QA cache: {msg}")
        qa_cache = qa_cache or {}

        success, all_docs, msg = FileUtils.load_binary(self.docs_file)
        if not success and not first_run:
            print(f"[WARN] Failed loading documents: {msg}")
        all_docs = all_docs or []

        return metadata, qa_cache, all_docs

    # -------------------
    # File scanning
    # -------------------
    def get_current_files(self) -> List[str]:
        """Get all supported files under documents_path"""
        if not self.documents_path or not os.path.exists(self.documents_path):
            return []

        files = []
        for root, dirs, filenames in os.walk(self.documents_path):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]
            for f in filenames:
                if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    full_path = os.path.join(root, f)
                    files.append(full_path)

        return files

    # -------------------
    # Metadata sync
    # -------------------
    def _sync_metadata(
        self,
        current_docs: List[Document],
        metadata: Dict[str, str],
        all_docs: List[Document],
        qa_cache: Dict
    ):
        current_ids = set()
        new_or_updated_docs = []
        removed_ids = []

        for doc in current_docs:
            doc_id = doc.metadata.get("id") or self._hash_text(doc.page_content)
            doc.metadata["id"] = doc_id
            doc_hash = self._hash_text(doc.page_content)
            current_ids.add(doc_id)

            if metadata.get(doc_id) != doc_hash:
                new_or_updated_docs.append(doc)
                metadata[doc_id] = doc_hash
                print(f"[INFO] New/updated document: {doc_id}")

        # Detect removed docs
        for doc_id in list(metadata.keys()):
            if doc_id not in current_ids:
                removed_ids.append(doc_id)
                print(f"[INFO] Removed document: {doc_id}")
                del metadata[doc_id]

        if removed_ids:
            all_docs = [d for d in all_docs if d.metadata.get("id") not in removed_ids]
            # Remove from QA cache
            keys_to_delete = [
                k for k, v in qa_cache.items()
                if any(src in removed_ids for src in v.get("sources", []))
            ]
            for k in keys_to_delete:
                del qa_cache[k]

        return all_docs, new_or_updated_docs, metadata, qa_cache

    # -------------------
    # File processing
    # -------------------
    def process_files(self, files: List[str], splitter) -> List[Document]:
        chunks = []
        for f in files:
            try:
                docs = self.loader.load(f)  # handles CSV, TXT, etc.
                for d in docs:
                    d.metadata["source"] = f
                chunks.extend(splitter.split_documents(docs))
            except Exception as e:
                print(f"[ERROR] Failed processing {f}: {e}")
        return chunks

    # -------------------
    # Vector DB
    # -------------------
    def rebuild_vector_db(self, all_docs, embeddings):
        try:
            if all_docs:
                vectordb = FAISS.from_documents(all_docs, embeddings)
                vectordb.save_local(self.index_path)
                print("[INFO] Vector DB rebuilt successfully")
            else:
                print("[INFO] No documents left, clearing DB")
                for f in ["index.faiss", "index.pkl"]:
                    fp = os.path.join(self.index_path, f)
                    if os.path.exists(fp):
                        os.remove(fp)
        except Exception as e:
            print(f"[ERROR] Failed to rebuild vector DB: {e}")

    # -------------------
    # Persistence
    # -------------------
    def persist_state(self, all_docs, metadata, qa_cache):
        for path, data, name in [
            (self.docs_file, all_docs, "documents"),
            (self.metadata_file, metadata, "metadata"),
            (self.cache_file, qa_cache, "QA cache"),
        ]:
            success, msg = FileUtils.save_binary(path, data)
            if not success:
                print(f"[WARN] Failed saving {name}: {msg}")

    # -------------------
    # Main sync
    # -------------------
    def sync(self, external_docs: Optional[List[Any]] = None):
        """
        Synchronize vector DB with local files + optional external docs
        external_docs can be:
        - Document
        - dict (with "content" or "metadata")
        - string
        """
        first_run = not os.path.exists(self.index_path)

        metadata, qa_cache, all_docs = self.load_state(first_run)

        splitter = TextSplitterFactory(
            strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model=self.embedding_model,
        ).get_splitter()

        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)

        # -------------------
        # Collect current documents from local files
        # -------------------
        local_files = self.get_current_files()
        local_docs = []
        if local_files:
            local_docs.extend(self.process_files(local_files, splitter))

        # -------------------
        # Normalize external_docs to Document
        # -------------------
        external_docs = external_docs or []
        normalized_docs = [self._to_document(doc) for doc in external_docs]

        all_current_docs = local_docs + normalized_docs

        # -------------------
        # Sync metadata (detect new/updated/removed)
        # -------------------
        all_docs, new_or_updated_docs, metadata, qa_cache = self._sync_metadata(
            all_current_docs, metadata, all_docs, qa_cache
        )

        if not new_or_updated_docs:
            print("[INFO] No changes detected.")
            return all_docs, metadata, qa_cache, "No changes detected."

        # -------------------
        # Chunk new/updated docs
        # -------------------
        new_chunks = splitter.split_documents(new_or_updated_docs)
        for d in new_chunks:
            d.metadata["source"] = d.metadata.get("id")  # ensure source is set

        all_docs.extend(new_chunks)

        # -------------------
        # Rebuild vector DB and persist
        # -------------------
        self.rebuild_vector_db(all_docs, embeddings)
        self.persist_state(all_docs, metadata, qa_cache)

        print("[INFO] Vector DB synchronized successfully.")
        return all_docs, metadata, qa_cache, "Synced"