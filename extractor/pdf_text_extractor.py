import pdfplumber
from utils.logger import get_logger
from utils.text_clean import clean_text

logger = get_logger("PDFTextExtractor")

class PDFTextExtractor:
    """
    Extract text (including tables) from PDFs
    """

    def __init__ (self):
        pass

    def extract_text(self, pdf_file):
        text_context = ""
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    text = page.extact_text() or ""
                    text_content += text + "\n"
            text_context = clean_text(text_content)
            logger.info(f"Extracted text from {pdf_file} ({len(text_content)} chars)")
        except Exception as e:
            logger.error(f"Failed to extract {pdf_file} : {e}")
        return {"file": pdf_file, "content":text_content}
    