import asyncio

import spacy

from coordinators.models.create import init_doc2_vec_models

from logger import init_logger
from models import init_models


def main():
    init_logger()
    init_models()

    # Run
    nlp = spacy.load('en_core_web_sm')
    asyncio.get_event_loop().run_until_complete(init_doc2_vec_models(nlp))
    input()


main()
