import asyncio
import time

import spacy

from coordinators.documents.read import count_documents
from coordinators.models.create import init_doc2_vec_models
from coordinators.models.read import count_models
from llms.agent.agent_executor import AgentExecutor
from logger import init_logger
from models import init_models
import functools

nlp = spacy.load('en_core_web_sm')

async def ainit():
    try:
        models = await init_doc2_vec_models(nlp)
        response = await arunning()
        return response
    except Exception as e:
        raise ValueError(str(e))


async def arunning():
    start_time = time.time()

    query = input('\nQuestion: ')
    agent = AgentExecutor()
    response = await agent.run(query=query)
    print('\n', response.output)

    end_time = time.time()
    print("\nTotal time searched: ", end_time - start_time)

    await arunning()


if __name__ == "__main__":
    init_logger()
    init_models()

    count_items_task = asyncio.gather(*[count_models(), count_documents()])
    count_items = asyncio.get_event_loop().run_until_complete(count_items_task)
    count_initialized = functools.reduce(lambda a, b: a+b, count_items)

    if count_initialized > 3:
        asyncio.run(arunning())
    else:
        asyncio.run(ainit())
