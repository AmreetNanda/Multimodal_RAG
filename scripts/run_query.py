import argparse
import pickle
from reranker.multimodal_reranker import MultimodalReranker
from generator.answer_generator import AnswerGenerator
from utils.logger import get_logger

logger = get_logger("RunQuery")


def main(args):
    # Load hybrid index
    with open(args.hybrid_index_file,"rb") as f:
        hybrid_indexer = pickle.load(f)
    
    reranker = MultimodalReranker()
    answer_gen = AnswerGenerator()

    while True:
        query = input("Enter your query (or 'exit' to quit): ")
        if query.lower() == "exit":
            break

        # 1. Retrieve top candidates
        candidates = hybrid_indexer.search(query, top_k=args.top_k)

        # Rerank
        reranked = reranker.rerank(query, candidates)

        # 3. Generate answer
        answer = answer_gen.generate_answer(query, reranked)
        print(f"\nAnswer:\n{answer}\n{'-'*50}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hybrid_index_file", type=str, default="index/hybrid/hybrid_index.pkl")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()
    main(args)