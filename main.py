import asyncio
import time

import spacy

from coordinators.documents.read import count_documents
from coordinators.models.create import init_doc2_vec_models
from coordinators.models.read import count_models, retrieve_models, filter_by_similarity_score

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

        query = 'haskell'
        similarities = asyncio.get_event_loop().run_until_complete(
            filter_by_similarity_score(nlp, query)
        )

        end_time = time.time()
        print("Total time searched: ", end_time - start_time)
    else:
        asyncio.get_event_loop().run_until_complete(init_doc2_vec_models(nlp))

    input()


main()
