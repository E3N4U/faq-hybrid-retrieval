import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)  # ⭐ 推荐用 IP（cosine）
        self.ids = []

    # =========================
    # add only vectors
    # =========================
    def add(self, vectors):

        vectors = np.asarray(vectors, dtype="float32")

        if vectors.ndim == 1:
            vectors = vectors.reshape(1, -1)

        faiss.normalize_L2(vectors)

        self.index.add(vectors)

    # =========================
    # search
    # =========================
    def search(self, query_vec, top_k=5):

        query_vec = np.asarray(query_vec, dtype="float32")

        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        faiss.normalize_L2(query_vec)

        D, I = self.index.search(query_vec, top_k)

        return D[0], I[0]


faq_index = FaissIndex(dim=512)