# bm25_index.py

import math
from collections import defaultdict
from typing import List
from .types import DocumentChunk

class BM25Index:
    """
    BM25 index for keyword-based retrieval.

    Features:
    - Incremental addition and deletion of documents.
    - BM25 scoring for single documents.
    - Simple whitespace tokenizer (lowercases text).
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Args:
            k1: BM25 term frequency scaling parameter.
            b: BM25 document length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.documents: List[DocumentChunk] = []
        self.doc_freq = defaultdict(int)
        self.term_freqs = {}
        self.avg_doc_length = 0

    def tokenize(self, text: str):
        """Tokenizes text into lowercase words using whitespace splitting."""
        return [t for t in text.lower().split() if t]

    def add_documents(self, docs: List[DocumentChunk]):
        """
        Add documents incrementally and update BM25 statistics.

        Args:
            docs: List of DocumentChunk objects to add.
        """
        total_length = sum(sum(self.term_freqs[d.id].values()) for d in self.documents) if self.term_freqs else 0
        self.documents.extend(docs)
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

        self.avg_doc_length = total_length / total_docs if total_docs else 0

    def delete_documents(self, doc_ids: List[str]):
        """
        Delete documents by IDs and update BM25 statistics.

        Args:
            doc_ids: List of document IDs to remove.
        """
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

    def score(self, query_tokens, doc: DocumentChunk):
        """
        Compute BM25 score between query and a document.

        Args:
            query_tokens: List of tokens from the query.
            doc: DocumentChunk to score.

        Returns:
            BM25 score as float.
        """
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

    def search(self, query: str):
        """
        Retrieve documents matching the query.

        Args:
            query: Query string.

        Returns:
            List of tuples (DocumentChunk, score).
        """
        tokens = self.tokenize(query)
        return [(doc, self.score(tokens, doc)) for doc in self.documents]