class HybridIndexer:
    """
    Combines the BM25 + Vector search for multimodal documents (PDF text + images)
    """

    def __init__(self, bm25_indexer, vector_indexer, alpha = 0.5):
        self.bm25 = bm25_indexer
        self.vector = vector_indexer
        self.alpha = alpha
    

    def search(self, query, top_k = 10):
        bm25_results = self.bm25.search(query, top_k*2)
        vector_results = self.vector.search(query, top_k*2)

        scored_docs = {}
        for i, doc in enumerate(bm25_results):
            scored_docs[doc["file"]] = scored_docs.get(doc["file"],0) + self.alpha*(len(bm25_results)-i)/len(bm25_results)
        for i, doc in enumerate(vector_results):
            scored_docs [doc["file"]] = scored_docs.get(doc["file"], 0) + (1-self.alpha)*(len(vector_results)-i)/len(vector_results)
        

        # Sort descending
        sorted_files = sorted(scored_docs.item(), key = lambda x:x[1], reverse=True)
        final_results = []
        seen = set()
        for file, score in sorted_files:
            if file not in seen:
                seen.add(file)
                doc = next((d for d in bm25_results + vector_results if d["file"]==file), None)
                if doc:
                    final_results.append(doc)
            if len(final_results) >= top_k:
                break
            return final_results