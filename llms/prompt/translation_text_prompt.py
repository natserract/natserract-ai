from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

translation_qa_prompt_template = """
Translate the input text to English
        
Input: {input}

If the input is in English, return the input text.
        
You are only allowed to return the translated text and nothing else. 

IMPORTANT: ONLY RETURN TRANSLATED TEXT AND NOTHING ELSE.
"""

translation_qa_prompt = ChatPromptTemplate(
    messages=[
        HumanMessagePromptTemplate.from_template(translation_qa_prompt_template)
    ],
    input_variables=["input"],
)
