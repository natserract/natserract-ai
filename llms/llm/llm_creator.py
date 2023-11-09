from llms.llm.openai import OpenAILLM


class LLMCreator:
    llms = {
        'openai': OpenAILLM,
    }

    @classmethod
    def create_llm(cls, type, *args, **kwargs):
        llm_class = cls.llms.get(type.lower())
        if not llm_class:
            raise ValueError(f"No LLM class found for type {type}")
        return llm_class(*args, **kwargs)