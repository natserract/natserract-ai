import logging

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import config
from llms.prompt.document_search_query import document_search_query_prompt
from llms.settings import settings


def document_search_query_chain(query: str):
    llm = ChatOpenAI(
        model_name=settings.MODEL,
        verbose=settings.VERBOSE,
        temperature=settings.TEMPERATURE,
        openai_api_key=config.Config.OPENAI_API_KEY,
    )

    lang_chain = LLMChain(llm=llm, prompt=document_search_query_prompt)
    result = lang_chain.run(query)

    logging.info(f'Transformed search query: {result}')
    return result
