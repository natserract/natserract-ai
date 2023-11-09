from pathlib import Path
from typing import List, Union

from langchain.document_loaders import TextLoader
from langchain.schema import Document

from data_loader.loader.markdown import MarkdownLoader


class FileExtractor:
    @classmethod
    def load_from_file(
            cls,
            file_path: str,
            return_text: bool = False,
    ) -> Union[List[Document] | str]:
        input_file = Path(file_path)
        delimiter = '\n'
        file_extension = input_file.suffix.lower()

        if file_extension in ['.md', '.markdown']:
            loader = MarkdownLoader(file_path, autodetect_encoding=True)
        else:
            loader = TextLoader(file_path, autodetect_encoding=True)

        return delimiter.join([
            document.page_content for document in loader.load()]
        ) if return_text else loader.load()
