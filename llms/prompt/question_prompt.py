from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

question_prompt_template = """
Use the following portion of a long document to see if any of the text is relevant to answer the question.
{{ context }}
Question: {{ question }}
Provide all relevant text to the question verbatim. Summarize if needed. If nothing relevant return "-".
"""

question_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(question_prompt_template)
    ],
    input_variables=["context", "question"],
)
