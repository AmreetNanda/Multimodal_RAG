from ollama import Ollama
from utils.logger import get_logger

logger = get_logger("MultimodalReranker")

class MultimodalReranker:
    """
    Reranks the top-k multimodal search results using llava/ bakllava. 
    Combines the PDF text + extracted image objects / captions
    """

    def __init__(self, model_name="llava-phi3"):
        self.model_name = model_name
        self.ollama = Ollama(model_name)
    
    def rerank(self, query, candidates):
        """
        query:str
        candidates: list of {"file":..., "content":...}
        Return candidates sorted by relevance
        """

        reranked = []
        for doc in candidates:
            prompt = f"""
            You are a multimodal assistant. Given the query:"{query}" 
            And the document content (text + extracted image objects): "{doc['content']}"
            Score the relevance of this document to the query from 0 to 100.
            Only return the numeric score
            """

            try:
                response = self.ollama.generate(prompt=prompt, max_tokens = 5)
                score = float(response.text.strip())
            except Exception as e:
                logger.warning(f"Failed to score {doc['file']}:{e}")
                score = 0
            reranked.append((doc, score))

        reranked.sort(key=lambda x:x[1], reverse = True)
        return [doc for doc, _ in reranked]