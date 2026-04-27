# vector_db.py

import os
import hashlib
from typing import List, Any
from datetime import datetime

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from ..vector_rag_ai.helpers import FileUtils


class VectorDB:
    def __init__(self, index_path: str, embedding_model: str):
        self.index_path = index_path

        self.metadata_file = os.path.join(index_path, "metadata.bin")
        self.docs_file = os.path.join(index_path, "documents.bin")

        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

        os.makedirs(index_path, exist_ok=True)

        self.metadata, self.all_docs = self._load_state()

    # -------------------
    # STATE
    # -------------------
    def _load_state(self):
        _, metadata, _ = FileUtils.load_binary(self.metadata_file)
        _, docs, _ = FileUtils.load_binary(self.docs_file)

        return metadata or {}, docs or []

    def _persist(self):
        FileUtils.save_binary(self.metadata_file, self.metadata)
        FileUtils.save_binary(self.docs_file, self.all_docs)

    # -------------------
    # HASH
    # -------------------
    def _hash(self, text: str):
        return hashlib.md5(text.encode()).hexdigest()

    # -------------------
    # SYNC
    # -------------------
    def sync(self, external_docs: List[Any]):
        docs_to_add = []

        for d in external_docs:
            if isinstance(d, str):
                content = d
                doc_id = self._hash(content)

            elif hasattr(d, "page_content"):
                content = d.page_content
                doc_id = self._hash(content)

            else:
                continue

            content_hash = self._hash(content)

            if self.metadata.get(doc_id, {}).get("hash") == content_hash:
                continue  # unchanged

            self.metadata[doc_id] = {
                "hash": content_hash,
                "updated_at": datetime.utcnow().isoformat(),
            }

            self.all_docs = [x for x in self.all_docs if x["id"] != doc_id]
            self.all_docs.append({"id": doc_id, "content": content})

            docs_to_add.append(
                Document(page_content=content, metadata={"id": doc_id})
            )

        if not docs_to_add:
            return

        # FAISS update
        if os.path.exists(os.path.join(self.index_path, "index.faiss")):
            vectordb = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            vectordb.add_documents(docs_to_add)
        else:
            vectordb = FAISS.from_documents(docs_to_add, self.embeddings)

        vectordb.save_local(self.index_path)

        self._persist()

    # -------------------
    # SEARCH
    # -------------------
    def search(self, question: str, k: int):
        try:
            vectordb = FAISS.load_local(
                self.index_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            docs = vectordb.similarity_search(question, k=k)
            return [d.page_content for d in docs]
        except:
            return []