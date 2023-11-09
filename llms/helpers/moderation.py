import logging
import random

import openai

from llms.error import LLMBadRequestError


def check_moderation(text: str) -> bool:
    # 2000 text per chunk
    length = 2000
    text_chunks = [text[i:i + length] for i in range(0, len(text), length)]

    if len(text_chunks) == 0:
        return True

    text_chunk = random.choice(text_chunks)

    try:
        moderation_result = openai.Moderation.create(
            input=text_chunk,
        )
    except Exception as ex:
        logging.exception(ex)
        raise LLMBadRequestError('Rate limit exceeded, please try again later.')

    for result in moderation_result.results:
        if result['flagged'] is True:
            return False

    return True
