from langchain_ollama import OllamaLLM
from utils.logger import get_logger
from utils.text_clean import clean_text

logger = get_logger("AnswerGenerator")


class AnswerGenerator:
    """
    Generates final answers from the multimodal top-k candidates
    (PDF text + image objects) using LLaVA / BakLLaVA in Ollama.
    """

    def __init__(self, model_name="llava-phi3"):
        self.model_name = model_name
        self.ollama = OllamaLLM(model=model_name)

    def generate_answer(self, query, candidates, max_input_chars=3000):
        """
        query: str
        candidates: list of {"file": ..., "content": ...}
        Returns a concise, accurate answer combining text + images
        """
        # Clean and truncate content to avoid overwhelming the model
        combined_content = "\n".join(
            [
                f"Document: {c['file']}\nContent: {clean_text(c['content'])[:max_input_chars]}"
                for c in candidates
            ]
        )

        prompt = f"""
You are a multimodal AI assistant. Answer the following query clearly and concisely:
"{query}"

Using the content from these documents (text + extracted images/diagrams):
{combined_content}

Instructions:
- Summarize the content in plain English.
- Ignore raw codes, long numeric sequences, or unformatted tables unless they are meaningful.
- Mention source files if relevant.
- Keep the answer concise and readable.
"""

        try:
            # Generate using OllamaLLM; prompts must be a list
            response = self.ollama.generate(prompts=[prompt])
            # Extract text from the first generation
            answer = response.generations[0][0].text.strip()
            return answer
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer."
