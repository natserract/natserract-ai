import logging
from typing import Any

import spacy
from langchain.tools import BaseTool

from coordinators.documents.read import get as get_document, filter_by_similarity_score

nlp = spacy.load('en_core_web_sm')


class DocumentQARetrieverTool(BaseTool):
    name: str = "document_qa_retriever"
    description = """
    A tool used searches and returns documents regarding conversational tech. 
    Input: send the user input.
    """

    def _run(self) -> Any:
        raise NotImplementedError("DocumentQARetrieverTool does not support sync")

    async def _arun(
            self,
            query: str
    ):
        logging.info(f'Find document similarity from query: {query}')
        similarities = await filter_by_similarity_score(nlp, query, 3)

        document_titles = []
        document_contents = []
        for document_id, keywords, similarity_score in similarities:
            document = await get_document(int(document_id))
            document_titles.append((document['title'], similarity_score))
            document_contents.append(document['content'])

        logging.info(f'Document has similarity found: {", ".join(document_titles)}')
        return document_contents[:1]

