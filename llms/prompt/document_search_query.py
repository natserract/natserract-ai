from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

document_search_query_prompt_template = """
Transform input: '{input}' to similarity search query

You are only allowed to return the search term result and nothing else. 
IMPORTANT: ONLY RETURN SEARCH TERM RESULT AND NOTHING ELSE.
"""


document_search_query_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(document_search_query_prompt_template)
    ],
    input_variables=["context", "input"],
)
