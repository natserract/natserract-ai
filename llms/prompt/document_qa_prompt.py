from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

document_qa_prompt_template = """
Use the following pieces of context to help answer the users question. If its not relevant to the question, provide friendly responses.
You have access to chat history, and can use it to help answer the question.
When using code examples, use the following format:
```(language)
(code)
```
----------------
{context}
"""


document_qa_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(document_qa_prompt_template)
    ],
    input_variables=["context"],
)
