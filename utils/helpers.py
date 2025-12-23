from urllib.parse import urlparse
import os

def is_valid_url(url):
    """
    Checks if a URL is valid or not
    """

    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False
    

def file_exists(path):
    """
    Checks if a local file exists
    """
    return os.path.exists(path)