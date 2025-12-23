import os
from utils.logger import get_logger

logger = get_logger("PDFCrawler")

class PDFCrawler:
    """
    Crawls local directoriesor given URLs to collect PDFs
    """

    def __init__(self, directories):
        self.directories = directories if isinstance(directories, list) else [directories]

    def crawl(self):
        pdf_files = []
        for directory in self.directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(".pdf"):
                        full_path = os.path.join(root, file)
                        pdf_files.append(full_path)
                        logger.info(f"Found PDF : {full_path}")
        return pdf_files
