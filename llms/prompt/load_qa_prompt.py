from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate

load_qa_prompt_template = "Answer the user question based on provided context. Ensure to answer in the provided tone. "
"For happy tone use a smiley. For other tones use an appropriate emoji"
"\n\nContext: {context}\n\n Tone: {tone}\n\nQuestion: {question}\n\nAnswer:"

load_qa_prompt = PromptTemplate(
    template="Answer the user question based on provided context. Ensure to answer in the provided tone. "
             "For happy tone use a smiley. For other tones use an appropriate emoji"
             "\n\nContext: {context}\n\n Tone: {tone}\n\nQuestion: {question}\n\nAnswer:",
    input_variables=["question","context","tone"]
)