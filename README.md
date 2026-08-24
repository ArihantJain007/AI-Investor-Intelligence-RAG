# AI-Powered Investor Intelligence Platform

A local-first RAG (Retrieval-Augmented Generation) platform that ingests corporate annual report PDFs, builds a searchable vector store, and answers investor questions with retrieved context using Google Gemini.

---



---

## Project Purpose

This platform lets investors ask natural-language questions about a company's annual report and receive answers grounded in the actual document — not in a general-purpose LLM's training data. It also automatically extracts structured financial KPIs (revenue, net income, risk factors, drivers, etc.) from each report.

Everything runs locally. No external database servers or cloud infrastructure setups are required.

---

## Architecture

```
PDF annual report
  │
  ▼
PyMuPDF4LLM          (PDF → Markdown)
  │
  ▼
RecursiveCharacterTextSplitter   (Markdown → fixed-size overlapping chunks)
  │
  ▼
Hugging Face Inference API       (chunks → text embeddings)
  │
  ▼
Chroma (local)                   (store & persist embeddings + metadata)
  │
  ▼
Filtered Retriever               (similarity search filtered by company + year)
  │
  ▼
Google Gemini                    (retrieved context → chat answer / KPI extraction)
  │
  ▼
Chat answer  /  data/financial_metrics.json
```

---

## Technology Stack

| Component | Technology |
|---|---|
| Web API | FastAPI |
| Dashboard UI | Jinja2 + Vanilla CSS |
| LLM (chat + KPI extraction) | Google Gemini via `langchain-google-genai` |
| Text embeddings | Hugging Face Inference API via `langchain-huggingface` |
| Vector store | Chroma (local, `data/chroma/`) |
| PDF processing | PyMuPDF4LLM |
| Metrics persistence | JSON file (`data/financial_metrics.json`) |
| Configuration | python-dotenv (`.env`) |

---

## Validation

The local RAG pipeline and evaluation suite were executed against the test suite (`evaluation/questions.json`). Below is the verified evaluation baseline result:

- **26/26** API calls successful
- **17/17** KPI tests passed
- **4/4** unsupported-query tests passed
- **4/4** cross-company isolation tests passed
- **1/1** qualitative test passed

---

## Installation & Setup

### 1. Create a virtual environment

```bash
python -m venv .venv
```

Activate the virtual environment:

Windows (PowerShell / Command Prompt):
```bash
.venv\Scripts\activate
```

macOS / Linux:
```bash
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```bash
copy .env.example .env   # Windows
# or
cp .env.example .env     # macOS / Linux
```

Fill in your API keys and configuration in `.env`:

```env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_MODEL=gemini-3.1-flash-lite

HUGGINGFACEHUB_API_TOKEN=your_huggingface_read_token
HF_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | ✅ | — | Google Gemini API key |
| `GEMINI_MODEL` | ❌ | `gemini-3.1-flash-lite` | Gemini model name |
| `HUGGINGFACEHUB_API_TOKEN` | ✅ | — | Hugging Face read token (Inference API) |
| `HF_EMBEDDING_MODEL` | ❌ | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `CHROMA_PERSIST_DIRECTORY` | ❌ | `data/chroma` | Chroma persistence path |
| `CHROMA_COLLECTION` | ❌ | `investor_reports` | Chroma collection name |
| `METRICS_FILE` | ❌ | `data/financial_metrics.json` | KPI metrics file path |

---

## Ingest Reports

Place PDF annual reports in `data/raw_pdfs/`.

Recommended filename format: `Company_YYYY.pdf` (e.g. `Apple_2024.pdf`, `Microsoft_2024.pdf`).

Ingest all PDFs into Chroma and extract financial KPIs:

```bash
python -m ingestion.ingest_documents
```

Force KPI re-extraction even if metrics already exist:

```bash
python -m ingestion.ingest_documents --force
```

---

## Run the API

Start the FastAPI application with uvicorn:

```bash
python -m uvicorn app:app --reload
```

Access the dashboard UI at: `http://localhost:8000`

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Dashboard UI (Jinja2 HTML) |
| `GET` | `/api/metrics` | Extracted financial metrics (JSON) |
| `POST` | `/api/chat` | Query RAG system for answers grounded in reports |
| `POST` | `/api/upload` | Upload a PDF to ingest into vector store |
| `GET` | `/health` | API Health Check |

### Chat Request Example

```http
POST /api/chat
Content-Type: application/json

{
  "question": "What was Apple's total revenue in 2024?",
  "company": "Apple",
  "year": 2024
}
```

Response:

```json
{
  "answer": "Apple's total net sales for fiscal year 2024 were $391,035 million.",
  "context_missing": false
}
```

---

## Run Evaluation

To run the local evaluation script against the test suite:

```bash
python -m evaluation.evaluate_rag
```

This sends evaluation cases to `/api/chat`, verifies numeric KPIs, unsupported refusals, and cross-company isolation, saving results to `evaluation/results.json`.

---

## Local Storage

```
data/
├── raw_pdfs/               ← place annual report PDFs here (.gitkeep preserved)
├── markdown/               ← auto-generated Markdown conversions
├── chroma/                 ← Chroma vector store (auto-created)
└── financial_metrics.json  ← extracted KPIs (auto-created)
```
