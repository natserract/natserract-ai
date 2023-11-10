import logging
from typing import (
    Union,
)

from langchain.agents import AgentOutputParser
from langchain.output_parsers.json import parse_json_markdown
from langchain.schema import (
    AgentAction,
    AgentFinish, OutputParserException, )


class OpenAIFunctionsAgentOutputParser(AgentOutputParser):
    """Parses a message into agent action/finish.

    Is meant to be used with OpenAI models, as it relies on the specific
    function_call parameter from OpenAI to convey what tools to use.

    If a function_call parameter is passed, then that is used to get
    the tool and tool input.

    If one is not passed, then the AIMessage is assumed to be the final output.
    """

    @property
    def _type(self) -> str:
        return "openai-functions-agent"

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        """Parse an AI message."""
        response = parse_json_markdown(text)

        if "action" in response and "action_input" in response:
            action, action_input = response["action"], response["action_input"]

            # If the action indicates a final answer, return an AgentFinish
            if action == "Final Answer":
                return AgentFinish({"output": action_input}, text)
            else:
                content_msg = f"responded: {action_input}\n" if action_input else "\n"
                log = f"\nInvoking: `{action}` \n{content_msg}\n"
                logging.info(log)

                return AgentAction(action, action_input, log)
        else:
            # If the necessary keys aren't present in the response, raise an
            # exception
            raise OutputParserException(
                f"Missing 'action' or 'action_input' in LLM output: {text}"
            )


