from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

document_qa_prompt_template = """
Use the following pieces of context to help answer the users question. 
If the answer is not contained within the context below, say 'Maaf, saya tidak tau :('

You have access to chat history, and can use it to help answer the question.
When using code examples, use the following format:
```(language)
(code)
```
----------------

Context: {context}
"""


document_qa_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(document_qa_prompt_template)
    ],
    input_variables=["context"],
)
