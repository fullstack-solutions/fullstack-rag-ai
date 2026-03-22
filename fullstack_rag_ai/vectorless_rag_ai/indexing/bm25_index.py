import math
from collections import defaultdict
from typing import List
from vectorless_rag_ai.types import DocumentChunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self.documents: List[DocumentChunk] = []
        self.doc_freq = defaultdict(int)
        self.term_freqs = {}
        self.avg_doc_length = 0

    def tokenize(self, text: str):
        return [t for t in text.lower().split() if t]

    def add_documents(self, docs: List[DocumentChunk]):
        self.documents = docs
        total_length = 0

        for doc in docs:
            tokens = self.tokenize(doc.text)
            total_length += len(tokens)

            tf = defaultdict(int)
            for t in tokens:
                tf[t] += 1

            self.term_freqs[doc.id] = tf

            for term in set(tokens):
                self.doc_freq[term] += 1

        self.avg_doc_length = total_length / len(docs)

    def score(self, query_tokens, doc: DocumentChunk):
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
            denominator = freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )

            score += idf * (numerator / denominator)

        return score

    def search(self, query: str):
        tokens = self.tokenize(query)

        results = []
        for doc in self.documents:
            score = self.score(tokens, doc)
            results.append((doc, score))

        return results