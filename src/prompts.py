from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate

def generate_js_code_prompt(query: str, context: str):
    return HumanMessage(
        content=f"""\n 
        You are an assistant for code generation tasks for user questions.\n\n
        Here is the user question:\n{query}\n\n
        Here is the context:\n{context}\n\n

        Generate a code snippet in javascript which includes a function to achieve the task described in the user question based on the context.
        If you don't know how to generate code for the task, say that you don't know. 
        If you have a generated code, do not explain the code neither give examples 
        nor suggestions to imporve it. And do not use markdown formatting.
        And resulting code should always end with call to console.log function which takes the output of the call to the previous function as an argument.
        For example: if the generated function to achieve user's task is as follows;\n
        function foo(a, b) '{' return a + b; '}'\n
        Then the resulting code should be like this:\n
        function foo(a, b) '{' return a + b; '}'
        console.log(foo(1, 2));
        \n\n
        """,
    )

def generate_reply_prompt(query: str, context: str):
    return HumanMessage(
        content=f"""\n 
        You are an assistant for a question-answering task.\n\n
        Here is the user question:\n{query}\n\n
        Here is the context:\n{context}\n\n

        Generate a reply to achieve the task described in the user question based on the context.
        First detect the language of the context.
        Then, If the context is in Korean, translate it to English.
        And, If it is in English, just use the context as it is.
        If there is no context related to the user question, say that you don't know.
        And your reply should be in English.
        Also, include the context in your reply.
        \n\n
        """,
    )

def translate_to_korean_prompt(query: str):
    return HumanMessage(
        content=f"""\n 
        You are an assistant for Korean-to-English translation tasks.\n\n
        Here is the user query:\n{query}\n\n

        First detect the language of the user query.
        Then, If the user query is in English, translate it to Korean.
        And, If it is in Korean, just return the user query as it is.
        \n\n
        """,
    )

def improve_question_prompt(query: str):
    return HumanMessage(
        content=f"""\n 
        Look at the input and try to reason about the underlying semantic intent / meaning.\n 
        Here is the initial question:\n\n{query}\n\n
        Formulate an improved question: """,
    )

def grade_relevance_prompt(query: str, context: str):
    return SystemMessage(
        content=f"""\n
        You are a grader assessing relevance of a retrieved context to a user question. \n 
        Retrieved context: \n\n {context} \n\n
        User question: \n\n {query} \n\n
        If the context contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the context is relevant to the question.""",
    )

def grade_relevance_prompt_template():
    return PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Retrieved document: \n\n {context} \n\n
        User question: \n\n {question} \n\n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
