import pdfplumber
from utils.logger import get_logger
from utils.text_clean import clean_text

logger = get_logger("PDFTextExtractor")

class PDFTextExtractor:
    """
    Extracts text (including tables) from PDFs.
    """
    def __init__(self):
        pass

    def extract_text(self, pdf_file):
        """
        Extracts text from a PDF file and returns a structured dictionary.
        """
        text_content = ""  # Ensure variable exists even if extraction fails
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text = page.extract_text() or ""  # ✅ fixed typo
                    text_content += text + "\n"
            text_content = clean_text(text_content)
            logger.info(f"Extracted text from {pdf_file} ({len(text_content)} chars)")
        except Exception as e:
            logger.error(f"Failed to extract {pdf_file}: {e}")
            text_content = ""  # fallback in case of error

        return {"file": pdf_file, "content": text_content, "type": "pdf"}
