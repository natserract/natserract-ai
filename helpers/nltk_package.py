import nltk


def download_nltk_packages(package):
    try:
        # Check if package is already downloaded
        nltk.data.find(package)
    except LookupError:
        # If not present, download it
        nltk.download(package.split('/')[1])