from langchain_ollama import OllamaLLM
from utils.logger import get_logger
import re

logger = get_logger("MultimodalReranker")

class MultimodalReranker:
    """
    Reranks the top-k multimodal search results using Ollama LLaVA.
    Combines the PDF text + extracted image objects / captions.
    """

    def __init__(self, model_name="llava-phi3"):
        self.model_name = model_name
        self.ollama = OllamaLLM(model=model_name)

    def rerank(self, query, candidates):
        """
        query: str
        candidates: list of {"file": ..., "content": ...}
        Returns candidates sorted by relevance
        """
        reranked = []
        for doc in candidates:
            prompt = f"""
            You are a multimodal assistant. Given the query:
            "{query}"

            And the document content (text + extracted image objects):
            {doc['content']}

            Score the relevance of this document to the query from 0 to 100.
            Only return the numeric score.
            """
            try:
                # Pass prompt as list
                response = self.ollama.generate(prompts=[prompt])
                # Extract numeric score from first generation
                score_str = response.generations[0][0].text.strip()
                # Extract first number found
                match = re.search(r'\d+(\.\d+)?', score_str)
                score = float(match.group()) if match else 0
                # score = float(score_str)
            except Exception as e:
                logger.warning(f"Failed to score {doc['file']}: {e}")
                score = 0
            reranked.append((doc, score))

        # Sort by score descending
        reranked.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in reranked]
