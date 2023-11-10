from datetime import datetime
from typing import Type

from langchain.tools import BaseTool
from pydantic import Field, BaseModel


class DatetimeTool(BaseTool):
    """Tool for querying current datetime."""
    name: str = "current_datetime"
    description: str = \
        "A tool when you want to get the current date, time, week, month or year, " \
        "and the time zone is UTC. Result is \"<date> <time> <timezone> <week>\"."

    def _run(self) -> str:
        # get current time
        current_time = datetime.utcnow()
        return current_time.strftime("%Y-%m-%d %H:%M:%S UTC+0000 %A")

    async def _arun(self, tool_input: str = '') -> str:
        # get current time
        current_time = datetime.utcnow()
        return current_time.strftime("%Y-%m-%d %H:%M:%S UTC+0000 %A")
