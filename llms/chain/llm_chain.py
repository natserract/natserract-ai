from typing import List, Dict, Any, Optional

from langchain import LLMChain as LCLLMChain
from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.llms import BaseLLM
from langchain.schema import LLMResult, Generation
from langchain.schema.language_model import BaseLanguageModel

from llms.agent.helpers.message import to_prompt_messages
from llms.llm.openai import OpenAILLM


class LLMChain(LCLLMChain):
    model_instance: BaseLLM
    """The language model instance to use."""
    llm: BaseLanguageModel = OpenAILLM()

    def generate(
            self,
            input_list: List[Dict[str, Any]],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> LLMResult:
        """Generate LLM result from inputs."""
        prompts, stop = self.prep_prompts(input_list, run_manager=run_manager)
        messages = prompts[0].to_messages()
        prompt_messages = to_prompt_messages(messages)
        result = self.model_instance.run(
            messages=prompt_messages,
            stop=stop
        )

        generations = [
            [Generation(text=result.content)]
        ]

        return LLMResult(generations=generations)
