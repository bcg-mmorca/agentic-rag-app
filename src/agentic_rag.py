import os

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from langgraph.graph import START, END, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

from typing import Any, Literal

from pydantic import BaseModel, Field

from langchain_google_vertexai import VertexAIEmbeddings

from src.document_processors.pdf_processor import PDFProcessor
from src.document_processors.javacript_code_processor import JSCodeDocumentProcessor
from src.vector_store.chromadb import ChromaDB
from src.vector_store.vertexai_vector_search import VertexAIVectorStore
from src.tools.javascript_executor.tool import JSCodeExecutor
from src.prompts import (
    generate_js_code_prompt,
    generate_reply_prompt,
    grade_relevance_prompt_template,
    improve_question_prompt,
    translate_to_korean_prompt,
)


PY_ENV = os.environ.get("PY_ENV")
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT")
REGION = os.environ.get("INDEX_REGION")
BUCKET_URI = os.environ.get("BUCKET_URI")
INDEX_ID = os.environ.get("INDEX_ID")
INDEX_ENDPOINT_ID = os.environ.get("INDEX_ENDPOINT_ID")
AGENT_MODE = os.environ.get("AGENT_MODE")
MEMORY_ENABLED = os.environ.get("MEMORY_ENABLED").lower() == "true"

embeddings_model = VertexAIEmbeddings(model="text-embedding-005")
llm = init_chat_model(
    "gemini-2.0-flash-001",
    model_provider="google_vertexai",
)

vector_store = VertexAIVectorStore(
    project_id=PROJECT_ID,
    region=REGION,
    bucket_uri=BUCKET_URI,
    index_id=INDEX_ID,
    index_endpoint_id=INDEX_ENDPOINT_ID,
    embeddings=embeddings_model,
) if PY_ENV == "prod" else ChromaDB(
    embeddings=embeddings_model,
    collection_name="js_code_collection",
)

document_processor = PDFProcessor(embeddings_model) if AGENT_MODE == "text" else JSCodeDocumentProcessor()

RECURSION_LIMIT = 5

@tool(response_format="content_and_artifact")
def retriever(query: str):
    """Retrieves context related to the given query"""

    retrieved_docs = vector_store.search(query, k=20)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
        for doc in retrieved_docs
    )

    return serialized, retrieved_docs

def grade_documents(state: MessagesState) -> Literal["generate", "rewrite"]:
    """Determines whether the retrieved documents are relevant to the question."""

    # Data model
    class Grade(BaseModel):
        """Binary score for relevance check."""

        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM with tool and validation
    llm_with_structured_output = llm.with_structured_output(Grade)

    grading_prompt = grade_relevance_prompt_template()
    grade = (grading_prompt | llm_with_structured_output).invoke({
        "question": state["messages"][0].content,
        "context": state["messages"][-1].content,
    })

    return "generate"if grade.binary_score == "yes" else "rewrite"

def translate(state: MessagesState):
    """Translates user query to the other language."""

    msg = translate_to_korean_prompt(
        query=state["messages"][0].content,
    )

    response = llm.invoke([msg])

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

def agent(state: MessagesState):
    """Generate tool call for retrieval or respond."""

    llm_with_tools = llm.bind_tools([retriever])
    response = llm_with_tools.invoke(state["messages"][0].content)

    tool_calls =[
        {
            "name": "retriever",
            "args": {"query": state["messages"][0].content},
            "id": "tool_call_id_1234abced",
            "type": "tool_call",
        }
    ]
    response.tool_calls = tool_calls

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

def generate_text(state: MessagesState):
    msg = generate_reply_prompt(
        query=state["messages"][0].content,
        context=state["messages"][-1].content,
    )

    response = llm.invoke([msg])

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

def generate_js_code(state: MessagesState):
    msg = generate_js_code_prompt(
        query=state["messages"][0].content,
        context=state["messages"][-1].content,
    )

    response = llm.invoke([msg])

    # MessagesState appends messages to state instead of overwriting
    return {"messages": [response]}

def rewrite(state: MessagesState):
    """Transform the query to produce a better question."""

    msg = improve_question_prompt(
        query=state["messages"][0].content,
    )

    response = llm.invoke([msg])
    return {"messages": [response]}

# TODO: Include this in the workflow
def generate_with_conversation(state: MessagesState):
    """Generate answer."""

    # Get previously generated ToolMessages
    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break
    tool_messages = recent_tool_messages[::-1]

    prev_tool_context = "\n\n".join(doc.content for doc in tool_messages)
    msg_with_prev_tool_context = generate_js_code_prompt(
        query=state["messages"][0].content,
        context=prev_tool_context,
    )

    # Get previously generated messages except tool messages
    conversation_messages = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    # Feed llm with the combined messages from memory
    response = llm.invoke([msg_with_prev_tool_context] + conversation_messages)

    return {"messages": [response]}

def execute(state: MessagesState):
    """Execute generated code."""

    code = state["messages"][-1].content
    cleaned_code = code.replace("```javascript", "").replace("```", "").strip()

    try:
        result = JSCodeExecutor.execute(cleaned_code)
        return {"messages": [result]}
    except Exception as e:
        error_message = f"Error executing code: {str(e)}"
        print(error_message)
        return {"messages": [HumanMessage(content=error_message)]}

def build_js_code_agent():
    """Create langgraph workflow for js code agent."""

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", ToolNode([retriever]))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate_js_code)
    workflow.add_node("execute", execute)
    workflow.add_node("grade_documents", grade_documents)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {END: "grade_documents", "tools": "retrieve"},
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", END)

    return workflow

def build_text_agent():
    """Create langgraph workflow for text agent."""

    workflow = StateGraph(MessagesState)
    # workflow.add_node("translate", translate)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", ToolNode([retriever]))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate_text)
    workflow.add_node("execute", execute)
    workflow.add_node("grade_documents", grade_documents)

    workflow.add_edge(START, "agent")
    # workflow.add_edge("translate", "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {END: "grade_documents", "tools": "retrieve"},
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", END)

    return workflow

def build_agent():
    """Build agent based on environment."""

    workflow = build_js_code_agent() if AGENT_MODE == "js_code" else build_text_agent()

    if MEMORY_ENABLED:
        print("Memory is enabled.")
        return workflow.compile(checkpointer=MemorySaver())
    else:
        print("Memory is disabled.")
        return workflow.compile()

def ask_agent(
    agent: CompiledStateGraph,
    query: str,
    thread_id: str,
    user_id: str,
) -> (dict[str, Any] | Any):
    # TODO: Check whether conversation history is empty,
    # and fetch conversation history if it is so.

    steps = agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
        config= {
            "recursion_limit": RECURSION_LIMIT,
            "configurable": {"thread_id": thread_id},
        },
    )

    res = []
    for step in steps:
        step["messages"][-1].pretty_print()
        # Get the last message from the final state
        res = step

    # res = agent.invoke(
    #     {"messages": [{"role": "user", "content": query}]},
    #     config= {"configurable": {"thread_id": thread_id}},
    # )

    # TODO: save conversation history to database

    return {"answer": res["messages"][-1].content, "context": []}

async def process_repository(path: str) -> int:
    docs = await document_processor.process(path)
    return vector_store.add(docs)
