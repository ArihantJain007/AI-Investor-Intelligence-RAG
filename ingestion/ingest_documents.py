"""Ingestion pipeline: PDF → Markdown → chunks → Chroma → KPI extraction.

Usage
-----
Ingest all PDFs in data/raw_pdfs/:

    python -m ingestion.ingest_documents

Re-extract KPIs even if they already exist:

    python -m ingestion.ingest_documents --force
"""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from database.json_store import get_metrics, save_metrics
from ingestion.pdf_to_markdown import PDFToMarkdownConverter
from ingestion.semantic_chunker import chunk_markdown
from llm.embeddings import get_embeddings
from rag.kpi_extractor_rag import extract_financial_metrics
from vectorstore.chroma_store import ChromaVectorStore, Retriever

load_dotenv()


def parse_company_year(pdf_file: Path) -> tuple[str, str]:
    """Parse company and year from a filename.

    Supported patterns:
        Apple_2024.pdf  →  company="Apple",  year="2024"
        2024_Apple.pdf  →  company="Apple",  year="2024"
        Apple.pdf       →  company="Apple",  year=""
    """
    stem = pdf_file.stem
    parts = stem.split("_")

    if parts and parts[0].isdigit():
        return "_".join(parts[1:]), parts[0]
    if len(parts) >= 2 and parts[-1].isdigit():
        return "_".join(parts[:-1]), parts[-1]
    return stem, ""


def _metrics_exist(company: str, year: str) -> bool:
    """Return True if financial metrics already exist for company/year."""
    existing = get_metrics()
    for row in existing:
        if row.get("company") == company and str(row.get("year", "")) == year:
            return True
    return False


def ingest_document(
    pdf_path: str,
    embeddings,
    vector_store: ChromaVectorStore,
    force_kpi: bool = False,
) -> None:
    """Convert, chunk, embed, store, and (optionally) extract KPIs from one PDF."""
    pdf_file = Path(pdf_path)
    company, year = parse_company_year(pdf_file)

    print(
        f"\n[ingest] {pdf_file.name}  ->  company={company!r}, year={year!r}"
    )

    converter = PDFToMarkdownConverter()
    markdown_file = converter.convert_pdf(
        pdf_path=pdf_path,
        output_dir="data/markdown",
    )

    chunks = chunk_markdown(markdown_file)
    print(f"[ingest] Generated {len(chunks)} chunks for {pdf_file.name}")

    vector_store.upload_chunks(
        chunks=chunks,
        company=company,
        year=year,
        source_file=pdf_file.name,
    )

    if not force_kpi and _metrics_exist(company, year):
        print(
            f"[ingest] KPI metrics already exist for {company} {year}. "
            "Skipping extraction (use --force to re-extract)."
        )
        return

    print(f"[ingest] Extracting KPIs for {company} {year} …")
    metrics = extract_financial_metrics(
        retriever=Retriever(vector_store),
        company=company,
        year=int(year) if year.isdigit() else None,
    )

    if metrics:
        save_metrics(
            company=company,
            year=int(year) if year.isdigit() else year,
            metrics=metrics.model_dump(),
        )
        print(f"[ingest] KPIs saved for {company} {year}.")
    else:
        print(f"[ingest] KPI extraction returned no results for {company} {year}.")


def ingest_directory(input_dir: str, force_kpi: bool = False) -> None:
    """Ingest every PDF found in *input_dir*."""
    embeddings = get_embeddings()
    vector_store = ChromaVectorStore(embeddings=embeddings)

    pdf_files = list(Path(input_dir).glob("*.pdf"))
    if not pdf_files:
        print(f"[ingest] No PDF files found in {input_dir}.")
        return

    print(f"[ingest] Found {len(pdf_files)} PDF(s) in {input_dir}")

    for pdf_file in pdf_files:
        ingest_document(
            pdf_path=str(pdf_file),
            embeddings=embeddings,
            vector_store=vector_store,
            force_kpi=force_kpi,
        )

    print("\n[ingest] All documents processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest PDF annual reports into the Chroma vector store."
    )
    parser.add_argument(
        "--input-dir",
        default="data/raw_pdfs",
        help="Directory containing PDF files to ingest (default: data/raw_pdfs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-extract KPIs even if metrics already exist "
            "for a company/year combination."
        ),
    )
    args = parser.parse_args()

    ingest_directory(input_dir=args.input_dir, force_kpi=args.force)
