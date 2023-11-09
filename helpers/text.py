import html

from markdown import markdown


def markdown_to_text(markdown_string):
    """Convert a markdown string to plaintext"""
    html_string = markdown(markdown_string)
    text_string = html.unescape(html_string)
    return text_string
