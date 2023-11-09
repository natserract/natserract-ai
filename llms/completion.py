import datetime
import json

from langchain.chains import LLMChain

import config
import openai

from llms.llm.llm_creator import LLMCreator
from llms.llm.openai import OpenAILLM
from llms.prompt import question_prompt
from llms.prompt.chat_combine_prompt import chat_combine_prompt
from llms.settings import settings
from transformers import GPT2TokenizerFast


def count_tokens(string):
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    return len(tokenizer(string)['input_ids'])


def complete_stream(question: str, documents: str):
    try:

        # p_chat_combine = chat_combine_prompt(summaries=documents)
        messages_combine = [{"role": "system", "content": question_prompt}]

        # if len(chat_history) > 1:
        #     tokens_current_history = 0
        #     # count tokens in history
        #     chat_history.reverse()
        #     for i in chat_history:
        #         if "prompt" in i and "response" in i:
        #             tokens_batch = count_tokens(i["prompt"]) + count_tokens(i["response"])
        #             if tokens_current_history + tokens_batch < settings.TOKENS_MAX_HISTORY:
        #                 tokens_current_history += tokens_batch
        #                 messages_combine.append({"role": "user", "content": i["prompt"]})
        #                 messages_combine.append({"role": "system", "content": i["response"]})
        # messages_combine.append({"role": "user", "content": question})

        llm = openai.ChatCompletion.create(
            model=config.Config.OPENAI_MODEL,
            messages=messages_combine
        )
        chain = LLMChain(llm=llm, prompt=chat_combine_prompt)

        completion = chain.run(
            question=question,
            context=documents
        )

        # completion = llm.gen(
        #     model=config.Config.OPENAI_MODEL,
        #     messages=messages_combine
        # )

        # for line in completion:
        #     data = json.dumps({"answer": str(line)})
        #     response_full += str(line)
        #     yield f"data: {data}\n\n"

        # generate summary
        result = {"answer": completion}
        #
        # messages_summary = [
        #     {"role": "assistant", "content": "Summarise following conversation in no more than 3 words, "
        #                                      "respond ONLY with the summary, use the same language as the system \n\n"
        #                                      "User: " + question + "\n\n" + "AI: " + result["answer"]},
        #     {"role": "user", "content": "Summarise following conversation in no more than 3 words, "
        #                                 "respond ONLY with the summary, use the same language as the system"}
        # ]

        # completion = llm.gen(
        #     model=config.Config.OPENAI_MODEL,
        #     messages=messages_summary,
        #     max_tokens=30
        # )

        # data = json.dumps({
        #     "type": "id",
        #     "data": {
        #         "user": "local",
        #         "date": datetime.datetime.utcnow(),
        #         "name": completion,
        #     }
        # })
        # yield f"data: {data}\n\n"
        # data = json.dumps({"type": "end"})
        # yield f"data: {data}\n\n"
        return result
    except Exception as e:
        print(str(e))
