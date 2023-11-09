from langchain.llms import BaseLLM

import config


class OpenAILLM(BaseLLM):
    def __init__(self, api_key):
        super().__init__()
        self.api_key = api_key
        self.openai = self._get_openai()

    def _llm_type(self):
        # This method should return the type of language model
        return "openai"

    def _generate(self, prompt, **kwargs):
        response = self.openai.Completion.create(prompt=prompt, **kwargs)
        return response.choices[0].text

    def _get_openai(self):
        # Import openai when needed
        import openai
        # Set the API key every time you import openai
        openai.api_key = self.api_key
        return openai

    def gen(self, model, messages, **kwargs):
        response = self.openai.ChatCompletion.create(
            model=model,
            messages=messages,
            **kwargs
        )
        return response["choices"][0]["message"]["content"]

    def gen_stream(self, model, messages, **kwargs):
        response = self.openai.ChatCompletion.create(
            model=model,
            messages=messages,
            **kwargs
        )
        for line in response:
            if "content" in line["choices"][0]["delta"]:
                yield line["choices"][0]["delta"]["content"]