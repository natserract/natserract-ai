from langchain.tools import BaseTool


# Searches and returns documents regarding conversational business. Input: send the user input.
class DocumentQATool(BaseTool):
    name: str = "document_qa"
    description = ''
