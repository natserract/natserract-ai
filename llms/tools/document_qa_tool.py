import logging
from typing import Any

import openai
import spacy
from langchain.tools import BaseTool

import config
from coordinators.documents.read import get as get_document, filter_by_similarity_score
from llms.chain.translation_text_chain import translation_text_chain
from llms.prompt.document_qa_prompt import document_qa_prompt

nlp = spacy.load('en_core_web_sm')


class DocumentQATool(BaseTool):
    name: str = "document_qa"
    description = """
    A tool when you want to answer questions about similarity search. 
    Input should be similarity search query based on the provided context.
    """

    def _run(self) -> Any:
        raise NotImplementedError("DocumentQATool does not support sync")

    async def _arun(
            self,
            query: str
    ):
        normalized_query = translation_text_chain(
            query
        )

        logging.info('Find document similarity')
        similarities = await filter_by_similarity_score(nlp, normalized_query)

        document_titles = []
        document_contents = []
        for document_id, keywords, similarity_score in similarities:
            document = await get_document(int(document_id))
            document_titles.append(document['title'])
            document_contents.append(document['content'])

        logging.info(f'Document has similarity found: {", ".join(document_titles)}')

        prompt = (document_qa_prompt
                  .format_prompt(context="".join(document_contents))
                  .to_string()
                  )

        response = await openai.ChatCompletion.acreate(
            model=config.Config.OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response
