from typing_extensions import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_vertexai import VectorSearchVectorStore

from google.cloud import aiplatform


class VertexAIVectorStore:
    """
    Vector store implementation for ChromaDB that uses the Chroma library to store and retrieve
    vector embeddings. It allows for efficient similarity search and retrieval of
    documents based on their vector representations.
    Attributes:
        embeddings (Embeddings): Embedding function to convert documents into vector representations.
        persist_directory (str): Directory to persist the vector store data.
        collection_name (str): Name of the collection in the vector store.
    """
    def __init__(
        self,
        project_id: str,
        region: str,
        bucket_uri: str,
        index_id: str,
        index_endpoint_id: str,
        embeddings: Embeddings,
    ):
        aiplatform.init(project=project_id, location=region, staging_bucket=bucket_uri)

        index = aiplatform.MatchingEngineIndex(index_id)
        index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_id)

        self.vector_store = VectorSearchVectorStore.from_components(
            project_id=project_id,
            region=region,
            gcs_bucket_name=bucket_uri,
            index_id=index.name,
            endpoint_id=index_endpoint.name,
            embedding=embeddings,
            stream_update=False,
        )

    def add(self, documents: List[Document]) -> int:
        doc_ids = self.vector_store.add_documents(documents=documents)
        return len(doc_ids)

    def search(self, query: str, k: int) -> List[Document]:
        return self.vector_store.similarity_search(query=query, k=k)
