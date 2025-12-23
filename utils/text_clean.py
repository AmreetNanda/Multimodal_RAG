import re

def clean_text(text):
    """
    Removes extra whitespaces, HTML tags, and non-printable characters.
    """

    # Remove HTML tags
    text = re.sub(r'<[^>]+>','',text)

    # Remove multiple spaces/newlines
    text = re.sub(r'\s+','',text)

    # Strip leading/trailing spaces
    return text.strip()