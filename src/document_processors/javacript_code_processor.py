from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_core.documents import Document
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
from typing_extensions import List

class JSCodeDocumentProcessor:
    def __init__(self):
        self.js_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.JS, chunk_size=80, chunk_overlap=0
        )

    def process(self, path) -> List[Document]:
        document_loader = GenericLoader.from_filesystem(
            path,
            glob="**/[!.]*",
            suffixes=[".js", ".ts"],
            exclude=["**/node_modules/**"],
            parser=LanguageParser(language=Language.JS),
        )

        docs = document_loader.load()
        return self.js_splitter.split_documents(docs)
