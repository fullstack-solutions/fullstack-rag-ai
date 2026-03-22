# bm25_index.py
import math
from collections import defaultdict
from typing import List
from .types import DocumentChunk

class BM25Index:
    """
    A BM25 index for efficient keyword-based retrieval.
    Supports incremental updates and deletion.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents: List[DocumentChunk] = []
        self.doc_freq = defaultdict(int)
        self.term_freqs = {}
        self.avg_doc_length = 0

    # -------------------------
    # Tokenization
    # -------------------------
    def tokenize(self, text: str):
        """Simple whitespace tokenizer, converts text to lowercase."""
        return [t for t in text.lower().split() if t]

    # -------------------------
    # Add Documents
    # -------------------------
    def add_documents(self, docs: List[DocumentChunk]):
        """Add documents incrementally."""
        self.documents.extend(docs)
        total_length = sum(sum(self.term_freqs[d.id].values()) for d in self.documents) if self.term_freqs else 0
        total_docs = len(self.documents)

        for doc in docs:
            tokens = self.tokenize(doc.text)
            total_length += len(tokens)

            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1
            self.term_freqs[doc.id] = tf

            for term in set(tokens):
                self.doc_freq[term] += 1

        self.avg_doc_length = total_length / total_docs if total_docs > 0 else 0

    # -------------------------
    # Delete Documents
    # -------------------------
    def delete_documents(self, doc_ids: List[str]):
        """Delete documents and update BM25 statistics incrementally."""
        remaining_docs = []
        for doc in self.documents:
            if doc.id in doc_ids:
                tf = self.term_freqs.pop(doc.id, {})
                for term in tf:
                    self.doc_freq[term] -= 1
                    if self.doc_freq[term] <= 0:
                        del self.doc_freq[term]
            else:
                remaining_docs.append(doc)
        self.documents = remaining_docs

        total_length = sum(sum(self.term_freqs[d.id].values()) for d in self.documents)
        self.avg_doc_length = total_length / len(self.documents) if self.documents else 0

    # -------------------------
    # Scoring
    # -------------------------
    def score(self, query_tokens, doc: DocumentChunk):
        """Compute BM25 score for a query and a single document."""
        tf = self.term_freqs.get(doc.id, {})
        doc_length = sum(tf.values())

        score = 0.0
        for term in query_tokens:
            if term not in self.doc_freq:
                continue
            df = self.doc_freq[term]
            idf = math.log((len(self.documents) - df + 0.5) / (df + 0.5) + 1)
            freq = tf.get(term, 0)
            numerator = freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)
        return score

    # -------------------------
    # Search
    # -------------------------
    def search(self, query: str):
        """Retrieve documents for a query."""
        tokens = self.tokenize(query)
        results = [(doc, self.score(tokens, doc)) for doc in self.documents]
        return results