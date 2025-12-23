from PIL import Image
from ollama import Ollama
from utils.logger import get_logger

logger = get_logger("ImageObjectExtractor")

class ImageObjectExtractor:
    """
    Uses LLaVa / BakLLaVa to extract objects, captions or text from images
    """

    def __init__(self, model_name = "llava-phi3"):
        self.ollama = Ollama(model_name)
        self.model_name = model_name

    def extract(self, image_path):
        """
        Returns structured text describing image objects or captions
        """

        prompt = f""" 
        Analyze the image at {image_path} and descibe all objects, tables, diagrams, or text in it. Output in concise descriptive text.
        """

        try:
            response = self.ollama.generate(prompt=prompt, max_tokens = 200)
            description = response.text.strip()
            logger.info(f"Extracted objects from {image_path} ({len(description)} chars)")
            return {"file":image_path, "content": description}
        except Exception as e:
            logger.error(f"Failed to extract objects from {image_path}: {e}")
            return {"file":image_path, "content":""}