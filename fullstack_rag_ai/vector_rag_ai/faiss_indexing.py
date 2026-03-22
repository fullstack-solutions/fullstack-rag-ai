import os
from typing import Set

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader

from .config import DEFAULT_CONFIG
from .helpers import FileUtils
from .text_splitter import TextSplitterFactory


class VectorDBSynchronizer:
    """
    Handles synchronization of FAISS vector DB with document directory.
    """

    def __init__(
        self,
        documents_path: str,
        index_path: str,
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

        self.metadata_file = os.path.join(index_path, "metadata.bin")
        self.cache_file = os.path.join(index_path, "qa_cache.bin")
        self.docs_file = os.path.join(index_path, "documents.bin")

        os.makedirs(index_path, exist_ok=True)

    # -------------------
    # State management
    # -------------------
    def load_state(self, first_run=False):
        success, metadata, msg = FileUtils.load_binary(self.metadata_file)
        if not success and not first_run:
            print(f"[WARN] metadata: {msg}")
        metadata = metadata or {}

        success, qa_cache, msg = FileUtils.load_binary(self.cache_file)
        if not success and not first_run:
            print(f"[WARN] qa_cache: {msg}")
        qa_cache = qa_cache or {}

        success, all_docs, msg = FileUtils.load_binary(self.docs_file)
        if not success and not first_run:
            print(f"[WARN] all_docs: {msg}")
        all_docs = all_docs or []

        return metadata, qa_cache, all_docs

    # -------------------
    # File detection
    # -------------------
    def get_current_pdf_files(self) -> Set[str]:
        if not os.path.exists(self.documents_path):
            print(f"[WARN] Documents path does not exist: {self.documents_path}")
            return set()

        return {
            f for f in os.listdir(self.documents_path)
            if f.lower().endswith(".pdf")
        }

    def handle_removed_files(
        self,
        metadata,
        current_files,
        all_docs,
        qa_cache,
    ):
        removed_files = [f for f in metadata if f not in current_files]
        rebuild_needed = bool(removed_files)

        if removed_files:
            all_docs = [
                d for d in all_docs
                if d.metadata.get("source") not in removed_files
            ]

            keys_to_delete = [
                k for k, v in qa_cache.items()
                if any(src in removed_files for src in v.get("sources", []))
            ]
            for k in keys_to_delete:
                del qa_cache[k]

            for f in removed_files:
                del metadata[f]
                print(f"[INFO] Removed document: {f}")

        return all_docs, qa_cache, metadata, rebuild_needed

    def detect_updated_files(self, current_files, metadata):
        updated_files = []
        rebuild_needed = False

        for f in current_files:
            file_path = os.path.join(self.documents_path, f)

            success, new_hash, msg = FileUtils.get_file_hash(file_path)

            if not success:
                print(f"[ERROR] {f}: {msg}")
                continue

            if metadata.get(f) != new_hash:
                updated_files.append(f)
                metadata[f] = new_hash
                rebuild_needed = True
                print(f"[INFO] Updated document: {f}")

        return updated_files, metadata, rebuild_needed

    # -------------------
    # Processing
    # -------------------
    def remove_old_chunks(self, all_docs, updated_files):
        return [
            d for d in all_docs
            if d.metadata.get("source") not in updated_files
        ]

    def process_new_files(self, updated_files, splitter):
        new_chunks = []

        for f in updated_files:
            try:
                loader = PyPDFLoader(os.path.join(self.documents_path, f))
                docs = loader.load()

                for d in docs:
                    d.metadata["source"] = f

                new_chunks.extend(splitter.split_documents(docs))

            except Exception as e:
                print(f"[ERROR] Failed processing {f}: {e}")

        return new_chunks

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
                print("[INFO] No documents left, clearing vector DB")
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
                print(f"[WARN] Failed to save {name}: {msg}")

    # -------------------
    # Main
    # -------------------
    def sync(self):
        first_run = not os.path.exists(self.index_path)

        metadata, qa_cache, all_docs = self.load_state(first_run)

        splitter = TextSplitterFactory(
            strategy=self.chunking_strategy,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            embedding_model=self.embedding_model,
        ).get_splitter()

        print(f"[INFO] Using chunking strategy: {self.chunking_strategy}")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model
        )

        current_files = self.get_current_pdf_files()

        all_docs, qa_cache, metadata, removed_flag = self.handle_removed_files(
            metadata, current_files, all_docs, qa_cache
        )

        updated_files, metadata, updated_flag = self.detect_updated_files(
            current_files, metadata
        )

        # First run
        if first_run and current_files:
            new_chunks = self.process_new_files(list(current_files), splitter)
            all_docs.extend(new_chunks)

            self.rebuild_vector_db(all_docs, embeddings)
            self.persist_state(all_docs, metadata, qa_cache)

            message = "No previous DB found. Created new vector DB."
            print(f"[INFO] {message}")
            return all_docs, metadata, qa_cache, message

        # No changes
        if not removed_flag and not updated_flag:
            message = "No changes detected."
            print(f"[INFO] {message}")
            return all_docs, metadata, qa_cache, message

        # Updates
        all_docs = self.remove_old_chunks(all_docs, updated_files)
        new_chunks = self.process_new_files(updated_files, splitter)
        all_docs.extend(new_chunks)

        self.rebuild_vector_db(all_docs, embeddings)
        self.persist_state(all_docs, metadata, qa_cache)

        message = "Vector DB synchronized successfully."
        print(f"[INFO] {message}")

        return all_docs, metadata, qa_cache, message