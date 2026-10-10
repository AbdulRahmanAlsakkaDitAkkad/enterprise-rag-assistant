# Enterprise RAG Assistant

A local, PDF-based Retrieval-Augmented Generation (RAG) assistant built with FastAPI, Ollama, sentence-transformers, and FAISS.

The application processes PDF documents and answers questions using retrieved document evidence. It supports single-document and multi-document chat, source citations, and chat sessions.

## Features

- PDF upload and text extraction
- OCR support using Tesseract
- Document chunking and embedding generation
- FAISS vector indexing and semantic search
- Local LLM inference through Ollama
- Single-document and multi-document question answering
- Source citations and evidence-verification logic
- Document registration using SQLite
- Interactive API documentation through FastAPI

## Technology Stack

- Python 3.11
- FastAPI
- Ollama with `qwen2.5:7b`
- Sentence Transformers (`all-MiniLM-L6-v2`)
- FAISS
- PyMuPDF and Tesseract OCR
- SQLite
- pytest

## Requirements

Install the following before starting:

1. Python 3.11
2. Ollama
3. Tesseract OCR (for OCR support)

The application expects Ollama at:

`http://127.0.0.1:11434`

The currently configured language model is:

`qwen2.5:7b`

## Installation (Windows PowerShell)

Clone the repository and enter the project directory:

```powershell
git clone <repository-url>
cd enterprise-rag-assistant
```

Create and activate a virtual environment:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Install Ollama separately and download the required model:

```powershell
ollama pull qwen2.5:7b
```

Make sure Ollama is running and Tesseract is installed and accessible through PATH.

## Run the API

From the project root, with the virtual environment activated:

```powershell
python -m uvicorn app.main:app --reload
```

Open the interactive API documentation:

http://127.0.0.1:8000/docs

Alternative documentation:

http://127.0.0.1:8000/redoc

## Recommended Workflow

### 1. Process a PDF

Use:

`POST /documents/process`

Upload a PDF using the multipart form field named `file`.

This endpoint runs the full ingestion pipeline:

1. Save the PDF
2. Extract text, using OCR where applicable
3. Create text chunks
4. Generate embeddings
5. Build a FAISS index
6. Register the document and return its `document_id`

Save the returned `document_id` for chat requests.

Note: `POST /documents/upload` extracts text only and does not run the full indexing pipeline.

### 2. Ask a question about one document

Use:

`POST /chat`

Example JSON request:

```json
{
  "document_id": "YOUR_DOCUMENT_ID",
  "question": "What are the main findings?",
  "top_k": 5
}
```

The response includes the question, answer, session ID, document ID, and sources.

### 3. Ask a question across multiple documents

Use:

`POST /chat/multi`

Example JSON request:

```json
{
  "document_ids": [
    "FIRST_DOCUMENT_ID",
    "SECOND_DOCUMENT_ID"
  ],
  "question": "What do these documents have in common?",
  "top_k": 5
}
```

At least two document IDs are required.

Both chat endpoints support an optional `session_id` to continue a chat session.

`top_k` accepts values from 1 to 10 and defaults to 5.

## API Endpoints

| Method | Endpoint | Description |
| --- | --- | --- |
| GET | `/health` | Health check |
| POST | `/documents/upload` | Upload PDF and extract text |
| POST | `/documents/process` | Full PDF processing and registration |
| GET | `/documents/{document_id}` | Retrieve registered document metadata |
| POST | `/processing/chunks` | Generate text chunks |
| POST | `/processing/embeddings` | Generate embeddings |
| POST | `/processing/faiss-index` | Build FAISS index |
| POST | `/search/semantic` | Semantic vector search |
| POST | `/chat` | Single-document RAG chat |
| POST | `/chat/multi` | Multi-document RAG chat |

The `/processing/*` endpoints expose individual pipeline stages for development and debugging.

## Local Data Storage

| Directory or file | Purpose |
| --- | --- |
| `data/uploads/` | Uploaded PDF files |
| `data/extracted/` | Extracted text and chunks |
| `data/indexes/` | Embeddings and FAISS indexes |
| `data/documents.db` | SQLite document registry |

These paths are relative to the working directory. Start the application from the project root.

## Tests

Run the automated test suite:

```powershell
python -m pytest -q
```

The latest verified local test run passed 42 tests.

## Configuration Notes

The Ollama host and model are currently defined directly in:

- `app/services/llm_service.py`
- `app/services/evidence_validator.py`
- `app/services/query_rewriter.py`

The configured host is `http://127.0.0.1:11434`, and the configured model is `qwen2.5:7b`.

Changing either currently requires updating the relevant modules.

## Project Status

This is a locally running RAG development project. Automated tests cover selected application behavior, including citation alignment and evidence-handling scenarios. Passing tests does not constitute a production security or deployment audit.
