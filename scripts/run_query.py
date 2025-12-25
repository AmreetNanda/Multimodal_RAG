import argparse
import pickle
from reranker.multimodal_reranker import MultimodalReranker
from generator.answer_generator import AnswerGenerator
from dashboard.latency_tracker import LatencyTracker
from dashboard.ranking_metrics import RankingMetrics
from utils.logger import get_logger

logger = get_logger("RunQuery")

# Ground truth for evaluation (extend this as needed)
GROUND_TRUTH = {
    "What does the system architecture diagram describe?": [
        "data/images/block_diagram.png"
    ],
    "Explain the design document": [
        "data/pdfs/design_doc.pdf"
    ]
}

def main(args):
    # Load hybrid index
    with open(args.hybrid_index_file,"rb") as f:
        hybrid_indexer = pickle.load(f)
    
    reranker = MultimodalReranker()
    answer_gen = AnswerGenerator()
    latency = LatencyTracker()

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        latency.records.clear()  # reset per query

        # 1. Retrieve top candidates
        # candidates = hybrid_indexer.search(query, top_k=args.top_k)
        candidates = latency.track(
            "Hybrid Retrieval",
            hybrid_indexer.search,
            query,
            args.top_k
        )
        if not candidates:
            print("No documents retrieved.")
            continue

        # 2. Rerank
        # reranked = reranker.rerank(query, candidates)
        reranked = latency.track(
            "Reranking",
            reranker.rerank,
            query,
            candidates[: args.rerank_k]
        )

        # 3. Evaluation (if ground truth exists)
        relevant_files = GROUND_TRUTH.get(query, [])

        if relevant_files:
            precision = RankingMetrics.precision_at_k(
                reranked,
                relevant_files,
                k=args.top_k
            )
            recall = RankingMetrics.recall_at_k(
                reranked,
                relevant_files,
                k=args.top_k
            )

            print(f"Precision@{args.top_k}: {precision:.2f}")
            print(f"Recall@{args.top_k}: {recall:.2f}")
        else:
            print("No ground truth available for this query.")

        # 4. Generate answer
        # answer = answer_gen.generate_answer(query, reranked)
        answer = latency.track(
            "Answer generation",
            answer_gen.generate_answer,
            query,
            reranked
        )
        print("\nAnswer:")
        print(answer)
        print("\nLatency summary:")
        for step, t in latency.summary().items():
            print(f"  {step}: {t:.2f}s")
        print("-" * 60)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Run multimodal RAG queries")
    parser.add_argument("--hybrid_index_file", 
                        type=str, 
                        default="index/hybrid/hybrid_index.pkl",
                        help="Path to saved hybrid index")
    parser.add_argument("--top_k", 
                        type=int, 
                        default=5,
                        help="Number of documents to retrieve")
    parser.add_argument("--rerank_k",
                        type=int,
                        default=5,
                        help="Number of documents to rerank (keep small)")
    args = parser.parse_args()
    main(args)