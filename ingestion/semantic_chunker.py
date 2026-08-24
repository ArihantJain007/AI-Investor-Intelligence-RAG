from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def read_markdown(markdown_file: str) -> str:
    return Path(markdown_file).read_text(encoding="utf-8")


def chunk_markdown(
    markdown_file: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split markdown into chunks using RecursiveCharacterTextSplitter.

    Args:
        markdown_file: Path to the markdown file produced by PyMuPDF4LLM.
        chunk_size: Maximum number of characters per chunk (default 1000).
        chunk_overlap: Number of overlapping characters between chunks (default 200).

    Returns:
        List of LangChain Document objects.
    """
    markdown_content = read_markdown(markdown_file)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    return splitter.create_documents([markdown_content])
