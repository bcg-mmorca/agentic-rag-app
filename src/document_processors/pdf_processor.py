from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from typing_extensions import List
from langchain_core.embeddings import Embeddings

class PDFProcessor:
    def __init__(self, embeddings: Embeddings):
        self.splitter = SemanticChunker(
            embeddings=embeddings, breakpoint_threshold_type="gradient"
        )

    async def process(self, path) -> List[Document]:
        loader = PyPDFLoader(path)
        pages: List[Document] = []
        async for page in loader.alazy_load():
            pages.append(page)

        print(f"{pages[0].metadata}\n")
        print(pages[0].page_content)

        return self.splitter.split_documents(pages)
