from ollama import Ollama
from utils.logger import get_logger

logger = get_logger("AnswerGenerator")

class AnswerGenerator:
    """
    Generates final answers from the multimodal top-k candidates (PDF text + image objects) using LLaVA / BakLLaVA in Ollama
    """

    def __init__(self, model_name = "llava-phi3"):
        self.model_name = model_name
        self.ollama = Ollama(model_name)

    def generate_answer(self, query, candidates, max_tokens=300):
        """
        query: str
        candidates: list of {"file":..., "content":...}
        Returns a concise, accurate answer combining text + images
        """

        combined_content = "\n".join([f"Document:{c['file']}\nContent:{c['content']}" for c in candidates])

        prompt = f"""
        You are a multimodal AI assistant. Aswer the following query:
        "{query}"
        Using the content from these documents (text + extracted images/diagrams):
        {combined_content}
        Provide a clear, concise, and accurate answer. Mention the source files if relevant.
        """

        try:
            response = self.ollama.generate(prompt=prompt, max_tokens = max_tokens)
            answer = response.text.strip()
            return answer
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer."



