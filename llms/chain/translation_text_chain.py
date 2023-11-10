import logging

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

import config
from llms.settings import settings

from llms.prompt.translation_text_prompt import translation_qa_prompt


def translation_text_chain(query: str):
    llm = ChatOpenAI(
        model_name=settings.MODEL,
        verbose=settings.VERBOSE,
        temperature=settings.TEMPERATURE,
        openai_api_key=config.Config.OPENAI_API_KEY,
    )

    lang_chain = LLMChain(llm=llm, prompt=translation_qa_prompt)
    result = lang_chain.run(query)

    logging.info(f'Translated question success: {result}')
    return result
