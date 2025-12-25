import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer
import os
import pickle

class VectorIndexer:
    """
    Vector-based semantic search using FAISS for multimodal content.
    """

    def __init__(self, embedding_model="all-MiniLM-L6-v2", dim = 384):
        self.model = SentenceTransformer(embedding_model)
        self.index = faiss.IndexFlatL2(dim)
        self.docs = []

    
    def add_documents(self, docs):
        """
        docs: list of {"file": ..., "content": ...}
        """

        embeddings = self.model.encode([d["content"] for d in docs], convert_to_numpy=True)
        self.index.add(embeddings.astype("float32"))
        self.docs.extend(docs)

    def search(self, query, top_k = 10):
        q_vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        D, I = self.index.search(q_vec, top_k)
        if self.index.ntotal == 0:
            return []
        results = []
        for i in I[0]:
            if i <len(self.docs):
                results.append(self.docs[i])
        return results
    
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        faiss.write_index(self.index, os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "docs.pkl"), "wb") as f:
            pickle.dump(self.docs, f)

    def load(self, path):
        self.index = faiss.read_index(os.path.join(path, "faiss.index"))
        with open(os.path.join(path, "docs.pkl"), "rb") as f:
            self.docs = pickle.load(f)