import os

from markdown import markdown
import html
import glob
import spacy
from gensim.utils import simple_preprocess


def get_base_path(path: str):
    base = os.getcwd()
    return os.path.join(
        base,
        path
    )


def load_markdowns() -> str:
    path = get_base_path(os.path.join('datasets'))
    pattern = os.path.join(path, '*.md')

    markdown_contents = []
    for markdown_file_path in glob.glob(pattern):
        with open(markdown_file_path, 'r', encoding="utf-8") as f:
            file_name = os.path.basename(markdown_file_path)
            content = f.read()
            markdown_contents.append((content, file_name))

    return markdown_contents


def get_markdown(name: str) -> str:
    path = get_base_path(os.path.join('datasets', f"{name}.md"))
    with open(path, 'r', encoding="utf-8") as f:
        content = f.read()

    return content


def markdown_to_text(markdown_string):
    """Convert a markdown string to plaintext"""
    html_string = markdown(markdown_string)
    text_string = html.unescape(html_string)
    return text_string



def get_vector(model, words):
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if word_vectors:
        return sum(word_vectors) / len(word_vectors)
    else:
        return None
