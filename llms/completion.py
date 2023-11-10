import datetime
import json

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms.openai import OpenAI
from langchain.schema import Document

import config
import openai

from llms.llm.llm_creator import LLMCreator
from llms.llm.openai import OpenAILLM
from llms.prompt import question_prompt
from llms.prompt.chat_combine_prompt import chat_combine_prompt
from llms.prompt.load_qa_prompt import load_qa_prompt
from llms.settings import settings
from transformers import GPT2TokenizerFast
from langchain.chains.question_answering import load_qa_chain


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

        llm = ChatOpenAI(model_name=config.Config.OPENAI_MODEL,
                         openai_api_key=config.Config.OPENAI_API_KEY,
                         temperature=0,
                         max_tokens=1000)
        chain = load_qa_chain(llm, chain_type="stuff", prompt=load_qa_prompt)

        resp = chain.run(input_documents=documents,
            question=question, tone="Sad"
        )

        # completion = chain.run(
        #     question=question,
        #     context=documents
        # )

        # completion = llm.gen(
        #     model=config.Config.OPENAI_MODEL,
        #     messages=messages_combine
        # )

        # for line in completion:
        #     data = json.dumps({"answer": str(line)})
        #     response_full += str(line)
        #     yield f"data: {data}\n\n"

        # generate summary
        result = {"answer": resp}
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
