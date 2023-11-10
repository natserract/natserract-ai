import asyncio
import time
from typing import List

import spacy
from langchain.schema import Document

from coordinators.documents.create import create_tagged_documents
from coordinators.documents.read import count_documents
from coordinators.models.create import init_doc2_vec_models
from coordinators.models.read import count_models, filter_by_similarity_score
from llms.agent.agent_executor import AgentExecutor
# from llms.completion import complete_stream

from logger import init_logger
from models import init_models


def main():
    init_logger()
    init_models()

    count_items_task = asyncio.gather(*[count_models(), count_documents()])
    count_items = asyncio.get_event_loop().run_until_complete(count_items_task)

    nlp = spacy.load('en_core_web_sm')
    if len(count_items) > 1:
        start_time = time.time()

        # query = 'Mulai versi berapa Angular fitur Renderer telah tersedia?'
        query = 'Tanggal berapa hari ini?'


        # similarities = asyncio.get_event_loop().run_until_complete(
        #     filter_by_similarity_score(nlp, query)
        # )
        #
        # documents = []
        # for keywords, similarity_score in similarities:
        #     value = " ".join(keywords)
        #     document = Document(page_content=value)
        #     documents.append(document)

        # agent_configuration = AgentConfiguration(
        #     strategy=planning_strategy,
        #     model_instance=agent_model_instance,
        #     tools=tools,
        #     summary_model_instance=summary_model_instance,
        #     memory=memory,
        #     callbacks=[chain_callback, agent_callback],
        #     max_iterations=10,
        #     max_execution_time=400.0,
        #     early_stopping_method="generate"
        # )
        agent = AgentExecutor()
        response = agent.run(query=query)
        #
        # response = complete_stream(query, documents)
        print('response', response.output)

        end_time = time.time()
        print("Total time searched: ", end_time - start_time)
    else:
        asyncio.get_event_loop().run_until_complete(init_doc2_vec_models(nlp))

    input()


main()
