"""Decision models that predict category / priority / intent from a ticket.

Three systems are compared in src/evaluate.py:
  A) TfidfLR   - classic sparse baseline (char n-grams + Logistic Regression)
  B) EmbLR     - multilingual embeddings + Logistic Regression  (used by the final pipeline)
  C) KNNVote   - similarity-weighted vote of the nearest historical tickets (retrieval as classifier)
"""
from collections import defaultdict
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from .embedder import Embedder, VectorIndex

TARGETS = ["category", "priority", "intent"]


class TfidfLR:
    name = "TF-IDF (char n-gram) + LR"

    def fit(self, texts: List[str], labels: Dict[str, np.ndarray]):
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)
        X = self.vec.fit_transform(texts)
        self.clf = {
            t: LogisticRegression(max_iter=2000, C=10, class_weight="balanced").fit(X, labels[t])
            for t in TARGETS
        }
        return self

    def predict(self, texts: List[str]) -> Dict[str, np.ndarray]:
        X = self.vec.transform(texts)
        return {t: self.clf[t].predict(X) for t in TARGETS}


class EmbLR:
    name = "Embeddings + LR"

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    def fit(self, texts: List[str], labels: Dict[str, np.ndarray], vecs: np.ndarray = None):
        X = vecs if vecs is not None else self.embedder.encode(texts)
        self.clf = {
            t: LogisticRegression(max_iter=2000, C=10, class_weight="balanced").fit(X, labels[t])
            for t in TARGETS
        }
        return self

    def predict_vecs(self, X: np.ndarray):
        preds, conf = {}, {}
        for t in TARGETS:
            proba = self.clf[t].predict_proba(X)
            preds[t] = self.clf[t].classes_[proba.argmax(axis=1)]
            conf[t] = proba.max(axis=1)
        return preds, conf

    def predict_with_conf(self, texts: List[str]):
        return self.predict_vecs(self.embedder.encode(texts))

    def predict(self, texts: List[str]) -> Dict[str, np.ndarray]:
        return self.predict_with_conf(texts)[0]


class KNNVote:
    def __init__(self, embedder: Embedder, k: int = 5):
        self.embedder, self.k = embedder, k
        self.name = f"kNN retrieval vote (k={k})"

    def fit(self, texts: List[str], labels: Dict[str, np.ndarray], vecs: np.ndarray = None):
        X = vecs if vecs is not None else self.embedder.encode(texts)
        self.index = VectorIndex(X.shape[1])
        self.index.add(X)
        self.labels = {t: np.asarray(labels[t]) for t in TARGETS}
        return self

    def predict(self, texts: List[str]) -> Dict[str, np.ndarray]:
        scores, idx = self.index.search(self.embedder.encode(texts), self.k)
        out = {t: [] for t in TARGETS}
        for row_scores, row_idx in zip(scores, idx):
            for t in TARGETS:
                votes = defaultdict(float)
                for s, i in zip(row_scores, row_idx):
                    votes[self.labels[t][i]] += float(s)
                out[t].append(max(votes, key=votes.get))
        return {t: np.array(v) for t, v in out.items()}
