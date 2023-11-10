import asyncio
import time

import spacy

from coordinators.documents.read import count_documents
from coordinators.models.create import init_doc2_vec_models
from coordinators.models.read import count_models
from llms.agent.agent_executor import AgentExecutor
from logger import init_logger
from models import init_models

nlp = spacy.load('en_core_web_sm')


async def ainit():
    return init_doc2_vec_models(nlp)


async def arunning():
    start_time = time.time()

    query = input('Question: ')
    agent = AgentExecutor()
    response = await agent.run(query=query)
    print('\n', response.output)

    end_time = time.time()
    print("\nTotal time searched: ", end_time - start_time)


def main():
    init_logger()
    init_models()

    count_items_task = asyncio.gather(*[count_models(), count_documents()])
    count_items = asyncio.get_event_loop().run_until_complete(count_items_task)

    if len(count_items) > 1:
        asyncio.get_event_loop().run_until_complete(arunning())
    else:
        asyncio.get_event_loop().run_until_complete(ainit())
        asyncio.get_event_loop().run_until_complete(arunning())

    input()


main()
