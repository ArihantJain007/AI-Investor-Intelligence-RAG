from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm.embeddings import get_embeddings
from llm.gemini import get_chat_model
from vectorstore.chroma_store import ChromaVectorStore, Retriever
from rag.kpi_extractor_rag import retrieve_context

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    company: str | None = None
    year: int | None = None


@router.post("/chat")
async def chat(request: ChatRequest):
    try:
        embeddings = get_embeddings()
        vector_store = ChromaVectorStore(embeddings=embeddings)
        retriever = Retriever(vector_store)

        # Determine if the question is about a financial metric
        metric_keywords = [
            "revenue",
            "net income",
            "operating income",
            "cash flow",
            "assets",
            "liabilities",
        ]
        if any(kw in request.question.lower() for kw in metric_keywords):
            # Use targeted retrieval similar to KPI extractor
            context = retrieve_context(
                retriever=retriever,
                company=request.company,
                year=request.year,
            )
        else:
            docs = retriever.invoke(
                query=request.question,
                company=request.company,
                year=request.year,
                top_k=12,
            )
            if not docs:
                return {
                    "answer": (
                        "No relevant context was found in the knowledge base for "
                        f"'{request.company or 'the requested company'} "
                        f"({request.year or 'any year'}). "
                        "Please ingest an annual report for this company/year first."
                    ),
                    "context_missing": True,
                }


            context = "\n\n".join(doc.page_content for doc in docs)
            # If no context was retrieved, return an informative message
        if not context or not context.strip():
            return {
                "answer": (
                    "No relevant context was found in the knowledge base for "
                    f"'{request.company or 'the requested company'}' "
                    f"({request.year or 'any year'}). "
                    "Please ingest an annual report for this company/year first."
                ),
                "context_missing": True,
            }

        prompt = f"""You are an expert financial analyst.

Answer the user's question using only the retrieved context from
corporate financial reports.

If the context does not contain enough information to fully answer
the question, say so clearly instead of inventing facts.

Retrieved context:
{context}

User question:
{request.question}
"""

        response = get_chat_model().invoke(prompt)

        answer = response.content if hasattr(response, "content") else str(response)

        return {"answer": answer, "context_missing": False}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
