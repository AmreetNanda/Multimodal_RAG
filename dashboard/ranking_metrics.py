class RankingMetrics:
    """
    Computes basic metrics like Precision@K and Recall@K for multimodal retrieval 
    """

    @staticmethod
    def precision_at_k(retrieved, relevant, k=10):
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        correct = sum(1 for doc in retrieved_k if doc["file"] in relevant_set)
        return correct/k
    
    @staticmethod
    def recall_at_k(retrieved, relevant, k=10):
        retrieved_k = retrieved[:k]
        relevant_set = set(relevant)
        correct = sum(1 for doc in retrieved_k if doc["file"] in relevant_set)
        return correct/max(1, len(relevant))
    