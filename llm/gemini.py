import os

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Default to gemini-3.1-flash-lite for low quota/cost usage.
# Override with GEMINI_MODEL environment variable.
_DEFAULT_MODEL = "gemini-3.1-flash-lite"


def get_chat_model() -> ChatGoogleGenerativeAI:
    """Return the Gemini chat model used by the application.

    The model is configurable via the GEMINI_MODEL environment variable.
    Defaults to gemini-3.1-flash-lite for low-cost free-tier usage.
    """
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing Gemini API key. Set GEMINI_API_KEY in your .env file."
        )

    return ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", _DEFAULT_MODEL),
        google_api_key=api_key,
        temperature=0.2,
    )


def get_structured_completion(
    prompt: str,
    response_model: type[BaseModel],
) -> BaseModel:
    """Generate a Pydantic-validated structured response with Gemini."""
    model = get_chat_model().with_structured_output(
        response_model,
        method="json_schema",
    )
    result = model.invoke(prompt)

    if isinstance(result, response_model):
        return result

    if isinstance(result, dict):
        return response_model.model_validate(result)

    raise RuntimeError(
        f"Gemini returned an unexpected structured-output type: {type(result)}"
    )
