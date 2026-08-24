import shutil
from pathlib import Path

from fastapi import APIRouter, File, UploadFile

from ingestion.ingest_documents import ingest_document
from llm.embeddings import get_embeddings
from vectorstore.chroma_store import ChromaVectorStore

router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept a PDF upload, save it, ingest into Chroma, and extract KPIs.

    KPI extraction is skipped if metrics already exist for the parsed
    company/year unless the caller passes ?force=true as a query param
    (currently always defaults to False for the web uploader).
    """
    upload_dir = Path("data/raw_pdfs")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename

    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    embeddings = get_embeddings()
    vector_store = ChromaVectorStore(embeddings=embeddings)

    ingest_document(
        pdf_path=str(file_path),
        embeddings=embeddings,
        vector_store=vector_store,
        force_kpi=False,
    )

    return {
        "message": "Document uploaded and ingested successfully.",
        "file_name": file.filename,
    }
