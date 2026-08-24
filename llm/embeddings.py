import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpointEmbeddings

load_dotenv()


def get_embeddings():
    """Return Hugging Face Inference API embeddings."""
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing Hugging Face API token. Set HUGGINGFACEHUB_API_TOKEN in your .env file."
        )

    return HuggingFaceEndpointEmbeddings(
        model=os.getenv(
            "HF_EMBEDDING_MODEL",
            "sentence-transformers/all-MiniLM-L6-v2",
        ),
        task="feature-extraction",
        huggingfacehub_api_token=token,
    )
