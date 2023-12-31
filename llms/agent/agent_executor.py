import logging
from typing import Union, Optional, Sequence, Any

from langchain.agents import AgentExecutor as LCAgentExecutor, OpenAIFunctionsAgent
from langchain.agents import BaseSingleActionAgent, BaseMultiActionAgent
from langchain.callbacks.base import Callbacks
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.memory.chat_memory import BaseChatMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain.tools import BaseTool
from pydantic.v1 import Extra, BaseModel

from config import Config
from llms.error import LLMError
from llms.helpers import moderation
from llms.settings import settings
from llms.tools.current_datetime_tool import DatetimeTool
from llms.tools.document_qa_tool import DocumentQARetrieverTool


class AgentConfiguration(BaseModel):
    llm: ChatOpenAI
    tools: list[BaseTool]
    memory: Optional[BaseChatMemory] = None
    callbacks: Callbacks = None
    max_iterations: int = 6
    max_execution_time: Optional[float] = None
    early_stopping_method: str = "generate"

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid.value
        arbitrary_types_allowed = True


class AgentExecuteResult(BaseModel):
    output: Optional[str]


class AgentExecutor:
    def __init__(self):
        self.configuration = AgentConfiguration(
            llm=ChatOpenAI(
                model_name=settings.MODEL,
                verbose=settings.VERBOSE,
                temperature=settings.TEMPERATURE,
                openai_api_key=Config.OPENAI_API_KEY,
            ),
            tools=self._get_tools(),
            memory=ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True,
                chat_memory=ChatMessageHistory(),
            ),
            callbacks=[]
        )
        self.agent = self._init_agent()

    def _get_tools(self) -> Sequence[BaseTool]:
        return [
            DatetimeTool(),
            DocumentQARetrieverTool()
        ]

    def _init_agent(self) -> Union[BaseSingleActionAgent | BaseMultiActionAgent]:
        logging.info("Initializing agent...")

        agent = OpenAIFunctionsAgent.from_llm_and_tools(
            llm=self.configuration.llm,
            tools=self.configuration.tools,
            system_message=SystemMessage(
                content="You are a NatserractAI, friendly and helpful AI assistant by Natserract that provides help "
                        "with documents."
            ),
            extra_prompt_messages=[
                MessagesPlaceholder(variable_name="chat_history"),
            ],
        )

        logging.info("Running agent...")
        return agent

    def should_use_agent(self, query: str) -> bool:
        return self.agent.should_use_agent(query)

    async def run(self, query: str) -> AgentExecuteResult:
        moderation_result = moderation.check_moderation(
            query
        )

        if not moderation_result:
            return AgentExecuteResult(
                output="I apologize for any confusion, but I'm an AI assistant to be helpful, harmless, and honest.",
                configuration=self.configuration
            )

        agent_executor = LCAgentExecutor.from_agent_and_tools(
            agent=self.agent,
            tools=self.configuration.tools,
            memory=self.configuration.memory,
            callbacks=self.configuration.callbacks,
            max_iterations=self.configuration.max_iterations,
            max_execution_time=self.configuration.max_execution_time,
            early_stopping_method=self.configuration.early_stopping_method,
        )

        try:
            output = await agent_executor.arun({
                'input': query
            })
        except LLMError as ex:
            raise ex
        except Exception as ex:
            logging.exception("agent_executor run failed")
            output = None

        return AgentExecuteResult(
            output=output,
            configuration=self.configuration
        )

