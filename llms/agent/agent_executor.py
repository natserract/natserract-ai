import logging
from typing import Union, Optional

from langchain.agents import BaseSingleActionAgent, BaseMultiActionAgent
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.prompts import MessagesPlaceholder
from pydantic import BaseModel

from config import Config
from llms.agent.agent.openai_function_call import AutoSummarizingOpenAIFunctionCallAgent
from langchain.agents import AgentExecutor as LCAgentExecutor

from llms.error import LLMError
from llms.helpers import moderation
from llms.settings import settings


class AgentExecuteResult(BaseModel):
    output: Optional[str]


class AgentExecutor:
    def __init__(self):
        self.agent = self._init_agent()

    def _init_agent(self) -> Union[BaseSingleActionAgent | BaseMultiActionAgent]:
        tools = []
        agent = AutoSummarizingOpenAIFunctionCallAgent.from_llm_and_tools(
            llm=ChatOpenAI(
                model_name=settings.MODEL,
                verbose=settings.VERBOSE,
                temperature=settings.TEMPERATURE,
                openai_api_key=Config.OPENAI_API_KEY,
            ),
            tools=tools,
            extra_prompt_messages=[
                MessagesPlaceholder(variable_name="chat_history"),
            ],
            verbose=True
        )

        logging.info("Agent initialized.")
        return agent

    def should_use_agent(self, query: str) -> bool:
        return self.agent.should_use_agent(query)

    def run(self, query: str) -> AgentExecuteResult:
        moderation_result = moderation.check_moderation(
            query
        )

        if not moderation_result:
            return AgentExecuteResult(
                output="I apologize for any confusion, but I'm an AI assistant to be helpful, harmless, and honest.",
            )

        tools = []
        callbacks = []
        agent_executor = LCAgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=tools,
            verbose=settings.VERBOSE,
            callbacks=callbacks,
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                chat_memory=ChatMessageHistory(),
            )
        )

        try:
            output = agent_executor.run(query)
        except LLMError as ex:
            raise ex
        except Exception as ex:
            logging.exception("agent_executor run failed")
            output = None

        return AgentExecuteResult(
            output=output,
        )
