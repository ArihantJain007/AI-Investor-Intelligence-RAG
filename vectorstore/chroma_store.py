import os
import uuid

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

load_dotenv()


class ChromaVectorStore:
    """Local persistent Chroma vector store.

    Chroma persists the collection under the project directory.
    """

    def __init__(
        self,
        embeddings,
        persist_directory: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIRECTORY", "data/chroma"
        )
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION", "investor_reports"
        )

        os.makedirs(self.persist_directory, exist_ok=True)

        self.client = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def upload_chunks(
        self,
        chunks: list[Document],
        company: str,
        year: str,
        source_file: str,
    ) -> None:
        """Add document chunks and metadata to Chroma."""
        documents = []
        ids = []

        for chunk in chunks:
            # Preserve existing metadata from chunking while adding company, year, source_file
            meta = dict(chunk.metadata) if chunk.metadata else {}
            meta.update(
                {
                    "company": company,
                    "year": str(year),
                    "source_file": source_file,
                }
            )
            doc = Document(page_content=chunk.page_content, metadata=meta)
            documents.append(doc)
            ids.append(str(uuid.uuid4()))

        if documents:
            self.client.add_documents(
                documents=documents,
                ids=ids,
            )

        print(f"Stored {len(documents)} chunks in Chroma.")

    def similarity_search(
        self,
        query: str,
        company: str | None = None,
        year: int | str | None = None,
        top_k: int = 20,
    ):
        """Retrieve the most relevant chunks, optionally filtered by company/year."""
        filters = {}

        if company:
            filters["company"] = company
        if year is not None:
            filters["year"] = str(year)

        if len(filters) == 1:
            where = filters
        elif len(filters) > 1:
            where = {"$and": [{k: v} for k, v in filters.items()]}
        else:
            where = None

        kwargs = {"k": top_k}
        if where:
            kwargs["filter"] = where

        return self.client.similarity_search(query, **kwargs)


class Retriever:
    """Small compatibility wrapper around Chroma retrieval."""

    def __init__(self, vector_store: ChromaVectorStore):
        self.vector_store = vector_store

    def invoke(
        self,
        query: str,
        company: str | None = None,
        year: int | str | None = None,
        top_k: int = 20,
    ) -> list:
        return self.vector_store.similarity_search(
            query=query,
            company=company,
            year=year,
            top_k=top_k,
        )
