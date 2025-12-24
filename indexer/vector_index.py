import faiss 
import numpy as np
from sentence_transformers import SentenceTransformer

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
        results = []
        for i in I[0]:
            if i <len(self.docs):
                results.append(self.docs[i])
        return results