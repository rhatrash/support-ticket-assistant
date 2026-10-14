"""Text -> normalized dense vectors, plus a small vector index (FAISS with a NumPy fallback)."""
from typing import List

import numpy as np

from .config import EMBED_BACKEND, EMBED_MODEL, SEED


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.clip(norms, 1e-12, None)).astype("float32")


class Embedder:
    """backend='st'    -> Sentence-Transformers multilingual model (handles Arabic + English)
    backend='tfidf' -> offline fallback: char n-gram TF-IDF reduced with SVD."""

    def __init__(self, backend: str = None, model_name: str = None):
        self.backend = backend or EMBED_BACKEND
        self.model_name = model_name or EMBED_MODEL
        self._model = None
        self._vec = None
        self._svd = None

    def fit(self, texts: List[str]) -> "Embedder":
        if self.backend == "st":
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        elif self.backend == "tfidf":
            from sklearn.decomposition import TruncatedSVD
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
            X = self._vec.fit_transform(texts)
            n_comp = max(2, min(128, X.shape[1] - 1, X.shape[0] - 1))
            self._svd = TruncatedSVD(n_components=n_comp, random_state=SEED).fit(X)
        else:
            raise ValueError(f"Unknown EMBED_BACKEND: {self.backend}")
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        texts = list(texts)
        if self.backend == "st":
            vecs = self._model.encode(
                texts, batch_size=64, convert_to_numpy=True, show_progress_bar=False
            )
        else:
            vecs = self._svd.transform(self._vec.transform(texts))
        return _l2_normalize(np.asarray(vecs))


class VectorIndex:
    """Cosine-similarity index (inner product on normalized vectors)."""

    def __init__(self, dim: int):
        self.dim = dim
        try:
            import faiss

            self._faiss = faiss
            self._index = faiss.IndexFlatIP(dim)
            self._matrix = None
        except ImportError:  # NumPy fallback keeps the project runnable without FAISS
            self._faiss = None
            self._index = None
            self._matrix = np.zeros((0, dim), dtype="float32")

    def add(self, vecs: np.ndarray) -> None:
        vecs = np.ascontiguousarray(vecs, dtype="float32")
        if self._faiss is not None:
            self._index.add(vecs)
        else:
            self._matrix = np.vstack([self._matrix, vecs])

    @property
    def size(self) -> int:
        return self._index.ntotal if self._faiss is not None else len(self._matrix)

    def search(self, queries: np.ndarray, k: int):
        k = min(k, self.size)
        queries = np.ascontiguousarray(queries, dtype="float32")
        if self._faiss is not None:
            return self._index.search(queries, k)
        sims = queries @ self._matrix.T
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores, idx
