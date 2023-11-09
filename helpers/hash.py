import hashlib


def get_model_name_from_content(content):
    """
    Create a hash of the content to use as a filename
    """
    return hashlib.sha256(content.encode('utf-8')).hexdigest()
