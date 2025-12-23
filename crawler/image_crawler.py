import os
from PIL import Image
from utils.logger import get_logger

logger = get_logger("ImageCrawler")

class ImageCrawler:
    """
    Crawls local directories for images (PNG, JPG, JPEG)
    """

    def __init__(self, directories):
        self.directories = directories if isinstance(directories, list) else [directories]
    
    def crawl(self):
        image_files = []
        for directory in self.directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith((".png", ".jpg", ".jpeg")):
                        full_path = os.path.join(root, file)
                        image_files.append(full_path)
                        logger.info(f"Found Image: {full_path}")
        return image_files