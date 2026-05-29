import os
import hashlib
from typing import List, Optional, Dict, Any, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from .config import DEFAULT_CONFIG
from ..document_loader import SUPPORTED_EXTENSIONS, IGNORED_DIRS
from .helpers import FileUtils
from .text_splitter import TextSplitterFactory
from ..document_loader import UniversalLoader
from datetime import datetime



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
        data_source_ref: Optional[str] = None,  # path like s3://bucket/folder or repo url
    ):
        self.documents_path = documents_path
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_strategy = chunking_strategy
        self.data_source_ref = data_source_ref
        if not self.data_source_ref:
            self.data_source_ref = documents_path

        self.loader = UniversalLoader()

        self.metadata_file = os.path.join(index_path, "metadata.bin")
        self.cache_file = os.path.join(index_path, "qa_cache.bin")
        self.docs_file = os.path.join(index_path, "documents.bin")
        self.file_metadata = os.path.join(index_path, "file_metadata.bin")

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

        success, file_metadata, msg = FileUtils.load_binary(self.file_metadata)
        if not success and not first_run:
            print(f"[WARN] Failed loading file metadata: {msg}")
        file_metadata = file_metadata or {}

        return metadata, qa_cache, all_docs, file_metadata

    # -------------------
    # File scanning
    # -------------------
    def get_new_files(
        self,
        file_metadata: Dict[str, Any],
        manual_file_mode: bool = False,
        uploaded_files: Optional[List[str]] = None,
        data_source_ref: Optional[str] = None
    ) -> Tuple[List[str], Dict[str, Any], List[str]]:

        """
        Returns:
            files -> list of new/modified file paths
            metadata_files -> updated metadata
            removed_files -> deleted file paths
        """

        files = []
        removed_files = []

        metadata_files = file_metadata.copy()

        # =====================================================
        # MANUAL FILE MODE
        # =====================================================

        if manual_file_mode:
            if not data_source_ref:
                raise Exception("In manual file mode, data_source_ref must be provided to track file origins")
            self.data_source_ref = data_source_ref
            uploaded_files = uploaded_files or []

            for full_path in uploaded_files:

                if not os.path.exists(full_path):
                    continue

                if not any(
                    full_path.lower().endswith(ext)
                    for ext in SUPPORTED_EXTENSIONS
                ):
                    continue

                try:
                    modified_time = os.path.getmtime(full_path)
                    created_time = os.path.getctime(full_path)

                except Exception:
                    continue

                old_metadata = metadata_files.get(full_path)

                # =================================================
                # IF FILE ALREADY EXISTS
                # mark old parsed version for removal
                # =================================================

                if old_metadata:

                    removed_files.append(full_path)

                # =================================================
                # NEW OR MODIFIED FILE
                # =================================================

                files.append(full_path)
                metadata_files[full_path] = {
                    "modified_timestamp": modified_time,
                    "created_timestamp": created_time,
                    "modified_date": datetime.fromtimestamp(modified_time),
                    "created_date": datetime.fromtimestamp(created_time),
                    "data_source": metadata_files[full_path]["data_source"] if full_path in metadata_files else self.data_source_ref,
                }

            return files, metadata_files, removed_files
        
        # Default: Folder scanning mode
        if not self.documents_path or not os.path.exists(self.documents_path):
            return [], file_metadata, []

        files = []
        removed_files = []

        metadata_files = file_metadata.copy()

        # all files currently present in this datasource folder
        current_files_set = set()

        for root, dirs, filenames in os.walk(self.documents_path):

            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS]

            for f in filenames:

                if not any(f.lower().endswith(ext) for ext in SUPPORTED_EXTENSIONS):
                    continue

                full_path = os.path.join(root, f)

                current_files_set.add(full_path)

                try:
                    modified_time = os.path.getmtime(full_path)
                    created_time = os.path.getctime(full_path)

                except Exception:
                    continue

                old_metadata = metadata_files.get(full_path)

                old_modified_time = (
                    old_metadata.get("modified_timestamp")
                    if old_metadata
                    else None
                )

                # only add if new or modified
                if old_modified_time != modified_time:

                    files.append(full_path)

                    metadata_files[full_path] = {
                        "modified_timestamp": modified_time,
                        "created_timestamp": created_time,
                        "modified_date": datetime.fromtimestamp(modified_time),
                        "created_date": datetime.fromtimestamp(created_time),
                        "data_source": self.documents_path,
                    }

        # detect removed files only for THIS datasource
        for metadata_path in list(metadata_files.keys()):

            metadata_entry = metadata_files.get(metadata_path, {})

            same_data_source = (
                metadata_entry.get("data_source") == self.documents_path
            )

            file_missing = metadata_path not in current_files_set

            # remove only if:
            # - belongs to same datasource
            # - no longer exists in folder
            if same_data_source and file_missing:
                print(f"[INFO] Detected removed file: {metadata_path}")
                removed_files.append(metadata_path)

                del metadata_files[metadata_path]

        return files, metadata_files, removed_files


    # -------------------
    # Metadata synchronization
    # -------------------

    def _sync_metadata(
        self,
        current_docs: List[Document],
        metadata: Dict[str, str],
        all_docs: List[Document],
        qa_cache: Dict,
        removed_files: Dict[str, Any],
    ):
        current_ids = set()
        removed_ids = []
        new_added_ids = []
        new_added_docs = []
        # Only docs from this source
        source_docs = [
            d for d in all_docs
            if self.data_source_ref in d.metadata.get("source", "")
        ]
        removed_document_source = [d for d in source_docs if d.metadata.get("source") in removed_files]
        source_doc_ids = {
            d.metadata.get("id") or self._hash_text(d.page_content)
            for d in removed_document_source
        }

        # Process current docs for this source
        for doc in current_docs:
            doc_source = doc.metadata.get("source", "")
            if self.data_source_ref not in doc_source:
                continue

            doc_id = doc.metadata.get("id") or self._hash_text(doc.page_content)
            doc.metadata["id"] = doc_id
            doc_hash = self._hash_text(doc.page_content)

            current_ids.add(doc_id)

            # Add or update
            if metadata.get(doc_id) != doc_hash:
                # Remove old version if exists
                new_added_ids.append(doc_id)
                new_added_docs.append(doc)
                all_docs = [d for d in all_docs if d.metadata.get("id") != doc_id]

                all_docs.append(doc)
                metadata[doc_id] = doc_hash

        # Detect removed docs ONLY for this source
        for doc_id in source_doc_ids:
            if doc_id not in current_ids:
                removed_ids.append(doc_id)
                metadata.pop(doc_id, None)

        if removed_ids:
            all_docs = [
                d for d in all_docs
                if d.metadata.get("id") not in removed_ids
            ]

            # Remove from QA cache
            keys_to_delete = [
                k for k, v in qa_cache.items()
                if any(src in removed_ids for src in v.get("sources", []))
            ]

            for k in keys_to_delete:
                del qa_cache[k]

        return all_docs, metadata, qa_cache, removed_ids, new_added_docs


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
                file_chunks = splitter.split_documents(docs)
                chunks.extend(file_chunks)
                print(f"[INFO] Processed {f} into {len(file_chunks)} chunks")
            except Exception as e:
                print(f"[ERROR] Failed processing {f}: {e}")
            print(f"[INFO] Total chunks after processing {f}: {len(chunks)}")
        return chunks

    # -------------------
    # Vector DB management
    # -------------------

    def update_vector_db(self, ids_to_remove, docs_to_add, embeddings):
        try:
            faiss_file = os.path.join(self.index_path, "index.faiss")
            pkl_file = os.path.join(self.index_path, "index.pkl")

            db_exists = os.path.exists(faiss_file) and os.path.exists(pkl_file)

            if not db_exists:
                if docs_to_add:
                    vectordb = FAISS.from_documents(docs_to_add, embeddings)
                    vectordb.save_local(self.index_path)
                    print("[INFO] Created new vector DB")
                return

            vectordb = FAISS.load_local(
                self.index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )

            # Remove documents
            if ids_to_remove:
                self._remove_from_faiss(vectordb, ids_to_remove)

            # Add new documents
            if docs_to_add:
                vectordb.add_documents(docs_to_add)
                print(f"[INFO] Added {len(docs_to_add)} documents")

            vectordb.save_local(self.index_path)
            print("[INFO] Vector DB updated successfully")

        except Exception as e:
            print(f"[ERROR] Failed to update vector DB: {e}")

    def _remove_from_faiss(self, vectordb, ids_to_remove):
        try:
            ids_to_remove = set(ids_to_remove)

            # Keep only docs not being removed
            remaining_docs = []
            for doc_id, doc in vectordb.docstore._dict.items():
                if doc.metadata.get("id") not in ids_to_remove:
                    remaining_docs.append(doc)

            if remaining_docs:
                new_db = FAISS.from_documents(remaining_docs, vectordb.embedding_function)
                vectordb.index = new_db.index
                vectordb.docstore = new_db.docstore
                vectordb.index_to_docstore_id = new_db.index_to_docstore_id
            else:
                vectordb.index.reset()
                vectordb.docstore._dict.clear()
                vectordb.index_to_docstore_id.clear()

            print(f"[INFO] Removed {len(ids_to_remove)} documents")

        except Exception as e:
            print(f"[ERROR] Failed to remove docs: {e}")

    # -------------------
    # Persistence
    # -------------------
    def persist_state(self, all_docs, metadata, qa_cache, file_metadata):
        for path, data, name in [
            (self.docs_file, all_docs, "documents"),
            (self.metadata_file, metadata, "metadata"),
            (self.cache_file, qa_cache, "QA cache"),
            (self.file_metadata, file_metadata, "file metadata"),
        ]:
            success, msg = FileUtils.save_binary(path, data)
            if not success:
                print(f"[WARN] Failed saving {name}: {msg}")

    # -------------------
    # Main sync
    # -------------------
    def sync(self, manual_file_mode=False, uploaded_files=None, data_source_ref=None):
        """
        Synchronize vector DB with local files
        """
        first_run = not os.path.exists(self.index_path)

        metadata, qa_cache, all_docs, file_metadata = self.load_state(first_run)
        embeddings = HuggingFaceEmbeddings(model_name=self.embedding_model)
        splitter = TextSplitterFactory(
            strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embeddings=embeddings,

        ).get_splitter()

       

        # -------------------
        # Collect current documents from local files
        # -------------------
        try:
            collected_files, file_metadata , removed_files = self.get_new_files(file_metadata, manual_file_mode, uploaded_files, data_source_ref)
        except Exception as e:
            raise Exception(f"Failed to get new files: {e}")
        collected_docs = []
        if not collected_files and not removed_files:
            print("[INFO] No new files to process")
            return all_docs, metadata, qa_cache, file_metadata, "No new files to process"
        if collected_files:
            collected_docs.extend(self.process_files(collected_files, splitter))
            if not collected_docs:
                return all_docs, metadata, qa_cache, file_metadata, "No documents found in the specified path"

        # -------------------
        # Sync metadata (detect new/updated/removed)
        # -------------------
        
        all_docs, metadata, qa_cache, removed_doc_ids, new_added_docs = self._sync_metadata(
            collected_docs, metadata, all_docs, qa_cache, removed_files
        )

        # -------------------
        # Rebuild vector DB and persist
        # -------------------

        if not removed_doc_ids and not new_added_docs:
            return all_docs, metadata, qa_cache, file_metadata, "No changes detected to sync"
            
        self.update_vector_db(removed_doc_ids, new_added_docs, embeddings)
        self.persist_state(all_docs, metadata, qa_cache, file_metadata)

        print("[INFO] Vector DB synchronized successfully.")
        return all_docs, metadata, qa_cache, file_metadata, "Synced"