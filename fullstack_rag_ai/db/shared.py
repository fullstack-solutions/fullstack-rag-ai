import hashlib
from typing import Any, Dict, List
from langchain_core.documents import Document

# ------------------------
# Shared Utilities
# ------------------------
def hash_text(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()

def to_document(item: Any, source: str = None) -> Document:
    """Normalize input into LangChain Document"""
    if isinstance(item, Document):
        return item
    elif isinstance(item, dict):
        content = item.get("content") or str(item)
        metadata = item.get("metadata", {})
        metadata["id"] = metadata.get("id") or hash_text(content)
        if source:
            metadata["source"] = source
        return Document(page_content=content, metadata=metadata)
    elif isinstance(item, str):
        return Document(page_content=item, metadata={"id": hash_text(item), "source": source})
    else:
        content = str(item)
        return Document(page_content=content, metadata={"id": hash_text(content), "source": source})

def sync_metadata(
    current_docs: List[Document],
    metadata: Dict[str, str],
    all_docs: List[Document],
    qa_cache: Dict
) -> tuple:
    """
    Compare current docs with existing metadata
    Returns: updated all_docs, new/updated docs, updated metadata, updated QA cache
    """
    current_ids = set()
    new_or_updated_docs = []
    removed_ids = []

    for doc in current_docs:
        doc_id = doc.metadata.get("id") or hash_text(doc.page_content)
        doc.metadata["id"] = doc_id
        doc_hash = hash_text(doc.page_content)
        current_ids.add(doc_id)

        if metadata.get(doc_id) != doc_hash:
            new_or_updated_docs.append(doc)
            metadata[doc_id] = doc_hash

    for doc_id in list(metadata.keys()):
        if doc_id not in current_ids:
            removed_ids.append(doc_id)
            del metadata[doc_id]

    if removed_ids:
        all_docs = [d for d in all_docs if d.metadata.get("id") not in removed_ids]
        keys_to_delete = [
            k for k, v in qa_cache.items()
            if any(src in removed_ids for src in v.get("sources", []))
        ]
        for k in keys_to_delete:
            del qa_cache[k]

    return all_docs, new_or_updated_docs, metadata, qa_cache