from typing import Any

from langchain.tools import BaseTool


# Searches and returns documents regarding conversational business. Input: send the user input.
class DocumentQATool(BaseTool):
    name: str = "document_qa"
    description = """
    A tool when you want to answer questions about similarity search. 
    Input should be search terms based on the provided context.
    """
    return_direct = True

    def _run(self) -> Any:
        raise NotImplementedError("DocumentQATool does not support sync")

    async def _arun(
        self,
        query: str
    ):
        print('Query', query)

        return query