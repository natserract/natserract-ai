import glob
import logging
import os
import re
import string

from gensim.models.doc2vec import TaggedDocument
from nltk import word_tokenize

from data_loaders.file_extractor import FileExtractor
from database import get_session
from helpers.file import get_base_path
from helpers.vectorize import preprocess_text
from models import Documents


async def create(path='_datasets'):
    try:
        path = get_base_path(os.path.join(path))
        pattern = os.path.join(path, '*.md')

        logging.info('Try to save documents')
        async with get_session() as session:
            documents = []
            for markdown_file_path in glob.glob(pattern):
                file_name = os.path.basename(markdown_file_path).replace('.md', '')
                file_content = FileExtractor.load_from_file(
                    markdown_file_path,
                    return_text=True
                )

                # Tokenize
                lowercase_sentence = file_content.lower()
                lowercase_sentence = lowercase_sentence.translate(
                    str.maketrans("", "", string.punctuation)
                )
                lowercase_sentence = lowercase_sentence.strip()
                lowercase_sentence = re.sub('\s+',' ',lowercase_sentence)

                tokens = word_tokenize(lowercase_sentence)
                tokens_str = ' '.join(tokens)

                document = Documents(
                    title=file_name,
                    content=tokens_str
                )
                session.add(document)
                await session.flush()

                documents.append({
                    "id": document.id,
                    "title": file_name,
                    "content": tokens_str
                })

            await session.commit()
            logging.info('All documents successfully created')

        return documents

    except Exception as e:
        raise ValueError(str(e))


def create_tagged_documents(nlp, documents: list):
    """
    Preprocess and tokenize each document
    """
    tagged_documents = [
        TaggedDocument(words=preprocess_text(nlp(document['content'])), tags=[str(document['id'])])
        for i, document in enumerate(documents)
    ]

    return tagged_documents
