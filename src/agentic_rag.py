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
CHROMADB_PERSIST_DIRECTORY = os.environ.get("CHROMADB_PERSIST_DIRECTORY", "./chroma_langchain_db")

print(f"CHROMADB_PERSIST_DIRECTORY: {CHROMADB_PERSIST_DIRECTORY}")
contents = os.listdir(CHROMADB_PERSIST_DIRECTORY)
print(contents)

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
    persist_directory=CHROMADB_PERSIST_DIRECTORY,
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
def generate_with_history(state: MessagesState):
    """Generate answer with chat history."""

    tools_history = _tools_recent_history(state)
    chat_history = _chat_history(state)

    msg = generate_reply_prompt(
        query=state["messages"][0].content,
        context=tools_history,
    )

    # Feed llm with the combined messages from memory
    response = llm.invoke([msg] + chat_history)

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

def dummy_state(state: MessagesState):
    """Empty state for the workflow."""

    return state

def _tools_recent_history(state: MessagesState):
    """Get recent ToolMessages."""

    recent_tool_messages = []
    for message in reversed(state["messages"]):
        if message.type == "tool":
            recent_tool_messages.append(message)
        else:
            break

    return "\n\n".join(doc.content for doc in recent_tool_messages[::-1])

def _chat_history(state: MessagesState):
    """Get chat history."""

    chat_history = [
        message
        for message in state["messages"]
        if message.type in ("human", "system")
        or (message.type == "ai" and not message.tool_calls)
    ]

    return chat_history

def build_js_code_agent():
    """Create langgraph workflow for js code agent."""

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", ToolNode([retriever]))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate_js_code)
    workflow.add_node("execute", execute)
    workflow.add_node("dummy_state", dummy_state)

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
    workflow.add_conditional_edges(
        "dummy_state",
        grade_documents,
    )
    workflow.add_edge("rewrite", "agent")
    workflow.add_edge("generate", "execute")
    workflow.add_edge("execute", END)

    return workflow

def build_text_agent():
    """Create langgraph workflow for text agent."""

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", agent)
    workflow.add_node("retrieve", ToolNode([retriever]))
    workflow.add_node("rewrite", rewrite)
    workflow.add_node("generate", generate_text)
    workflow.add_node("dummy_state", dummy_state)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges(
        "agent",
        tools_condition,
        {END: "dummy_state", "tools": "retrieve"},
    )
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
    )
    workflow.add_conditional_edges(
        "dummy_state",
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

    try:
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
    except RecursionError as e:
        print(f"Recursion error: {str(e)}")
        return {"answer": "Please try again with a simpler and concise question.", "context": []}
    except Exception as e:
        print(f"Error: {str(e)}")
        return {"answer": "An error occurred while processing your request.", "context": []}

async def process_repository(path: str) -> int:
    docs = await document_processor.process(path)
    return vector_store.add(docs)
