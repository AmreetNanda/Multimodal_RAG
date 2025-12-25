from PIL import Image
from langchain_ollama import OllamaLLM
from utils.logger import get_logger

logger = get_logger("ImageObjectExtractor")

class ImageObjectExtractor:
    """
    Uses LLaVA/BakLLaVA via OllamaLLM to extract objects, captions, or text from images.
    """
    def __init__(self, model_name="llava-phi-3"):
        self.model_name = model_name
        self.ollama = OllamaLLM(model=model_name)

    def extract(self, image_path):
        """
        Returns structured text describing image objects or captions.
        """
        prompt = f"""
        Analyze the image at {image_path} and describe all objects, tables, diagrams, or text in it.
        Output concise descriptive text.
        """
        try:
            response = self.ollama.generate(prompt=prompt)
            description = response.text.strip() if response else ""
            logger.info(f"Extracted objects from {image_path} ({len(description)} chars)")
            return {"file": image_path, "content": description, "type": "image"}
        except Exception as e:
            logger.error(f"Failed to extract objects from {image_path}: {e}")
            return {"file": image_path, "content": "", "type": "image"}

