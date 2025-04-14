from typing import Any, Optional, Type
from langchain_core.callbacks import (
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from langchain_google_vertexai import VertexAIEmbeddings
from pydantic import BaseModel, Field

from src.vector_store.chromadb import ChromaDB


vector_store = ChromaDB(
    embeddings=VertexAIEmbeddings(model="text-embedding-005"),
    collection_name="js_code_collection",
)

class RetrieverInput(BaseModel):
    """Input for the Retreiver tool."""

    query: str = Field(description="Query corresponds to information to be retreived")

class Retriever(BaseTool):  # type: ignore[override]
    """Tool that retrieves information related to a query."""

    name: str = "retreiver"
    description: str = (
        """Retrieves information related to a query.

        Args:
            query (str): Query corresponds to information to be retreived.

        Returns:
            str: Serialized retrieved docs
            List[Document]: retrieved docs
        """
    )
    args_schema: Optional[Type[BaseModel]] = RetrieverInput

    def __init__(self) -> None:
        super().__init__()
        self.metadata = {
            "name": self.name,
            "description": self.description,
            "response_format": "content_and_artifact",
        }

    def _run(
        self,
        query: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> str:
        """Use the tool."""
        try:
            retrieved_docs = vector_store.search(query, k=4)
            serialized = "\n\n".join(
                (f"Source: {doc.metadata}\n" f"Content: {doc.page_content}")
                for doc in retrieved_docs
            )
            return serialized, retrieved_docs

        except Exception as e:
            return f"The following errors occurred during tool execution: `{e.args[0]}`", []
