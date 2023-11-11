import logging
from typing import Any

import openai
import spacy
from langchain.tools import BaseTool

import config
from coordinators.documents.read import get as get_document, filter_by_similarity_score
from llms.prompt.document_qa_prompt import document_qa_prompt

nlp = spacy.load('en_core_web_sm')


class DocumentQARetrieverTool(BaseTool):
    name: str = "document_qa_retriever"
    description = """
    A tool used searches and returns documents regarding conversational tech. 
    Input: send the user input.
    """
    return_direct = True

    def _run(self) -> Any:
        raise NotImplementedError("DocumentQARetrieverTool does not support sync")

    async def _arun(
            self,
            query: str
    ):
        logging.info(f'Find document similarities from the query: {query}')
        similarities = await filter_by_similarity_score(nlp, query, 3)

        document_titles = []
        document_contents = []
        for doc_id, keywords, similarity_score in similarities:
            document = await get_document(int(doc_id))
            document_titles.append(f"({document['title']}, {similarity_score})")
            document_contents.append(document['content'])

        logging.info(f'Document similarities found: {", ".join(document_titles)}')

        prompt = (document_qa_prompt
                  .format_prompt(
                    context="".join(document_contents),
                    input=query
                  )
                  .to_string()
                  )

        response = await openai.ChatCompletion.acreate(
            model=config.Config.OPENAI_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message['content']

