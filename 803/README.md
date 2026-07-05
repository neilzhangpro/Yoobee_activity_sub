# CV Feedback RAG Agent

A minimal FastAPI backend that accepts CV uploads, runs a LangChain agent with RAG over the document, and returns structured feedback plus an exportable optimized CV (`.docx` or `.pdf`).

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" height="20" alt="Python"/>
  <img src="https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white" height="20" alt="FastAPI"/>
  <img src="https://img.shields.io/badge/LangChain-1C3C3C?style=flat-square&logo=langchain&logoColor=white" height="20" alt="LangChain"/>
  <img src="https://img.shields.io/badge/LangGraph-1C3C3C?style=flat-square&logo=langchain&logoColor=white" height="20" alt="LangGraph"/>
  <img src="https://img.shields.io/badge/Chroma-FF6F00?style=flat-square&logo=chromadb&logoColor=white" height="20" alt="Chroma"/>
  <img src="https://img.shields.io/badge/OpenAI-412991?style=flat-square&logo=openai&logoColor=white" height="20" alt="OpenAI"/>
  <img src="https://img.shields.io/badge/Git-F05032?style=flat-square&logo=git&logoColor=white" height="20" alt="Git"/>
  <img src="https://img.shields.io/badge/GitHub-181717?style=flat-square&logo=github&logoColor=white" height="20" alt="GitHub"/>
  <img src="https://img.shields.io/badge/VS%20Code-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white" height="20" alt="VS Code"/>
</p>

## Features

- Upload a CV as **PDF**, **DOCX**, or **TXT**
- Chunk and index the CV locally with **Chroma** and custom **HashEmbeddings** (no extra embedding API key)
- LangChain **agent** with a `retrieve_cv_context` tool for evidence-based feedback
- Structured JSON feedback: summary, strengths, and content / structure / presentation improvements
- Export an optimized CV as **DOCX** or **PDF**

## Architecture

```text
Client
  |
  | POST /feedback or /optimize
  v
FastAPI Backend
  |
  +--> File Parser
  |      +-- PDF  -> pypdf
  |      +-- DOCX -> python-docx
  |      +-- TXT  -> UTF-8 text
  |
  +--> RAG Indexing
  |      +-- RecursiveCharacterTextSplitter (900 / 150 overlap)
  |      +-- HashEmbeddings (512-dim, BLAKE2b token hashing)
  |      +-- Chroma vector store (per-request collection)
  |
  +--> LangChain Agent (create_agent)
  |      +-- ChatOpenAI (temperature=0.2)
  |      +-- Tool: retrieve_cv_context(query) -> top-k similarity search
  |
  +--> Output
         +-- JSON feedback (/feedback)
         +-- optimized CV file (/optimize -> outputs/)
```

## Tech Stack

| Layer | Technology |
| --- | --- |
| Runtime | Python |
| API | FastAPI · Uvicorn |
| Agent & RAG | LangChain · LangGraph · Chroma |
| LLM | ChatOpenAI (OpenAI, OpenRouter, or compatible base URL) |
| Embeddings | Custom `HashEmbeddings` (local, no API key) |
| CV parsing | pypdf · python-docx · plain text |
| Export | python-docx · ReportLab |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` with your LLM credentials. The app loads environment variables on startup.

### OpenRouter (recommended)

```bash
LLM_API_KEY=sk-or-...
LLM_BASE_URL=https://openrouter.ai/api/v1
LLM_MODEL=openai/gpt-4o-mini
OPENROUTER_SITE_URL=http://127.0.0.1:8000
OPENROUTER_APP_NAME=CV Feedback RAG Agent
```

### OpenAI (or compatible proxy)

```bash
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=
```

Supported API key env vars (first match wins): `LLM_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`.

Model and base URL fall back to `LLM_MODEL` / `LLM_BASE_URL`, then `OPENAI_MODEL` / `OPENAI_BASE_URL`. Default model: `gpt-4o-mini`.

## Run

```bash
uvicorn app:app --reload
```

Interactive API docs:

```text
http://127.0.0.1:8000/docs
```

## API

### `POST /feedback`

Generate structured CV feedback.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | required | CV file (`.pdf`, `.docx`, or `.txt`) |
| `target_role` | string | `""` | Target job title; empty uses a general professional role |
| `top_k` | int | `4` | Retrieved chunks per query (1–10) |

```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
  -F "file=@cv.pdf" \
  -F "target_role=Backend Developer" \
  -F "top_k=4"
```

Response:

```json
{
  "summary": "...",
  "strengths": ["..."],
  "content_improvements": ["..."],
  "structure_improvements": ["..."],
  "presentation_improvements": ["..."],
  "optimized_cv_text": "..."
}
```

### `POST /optimize`

Run the same analysis and download the optimized CV.

| Field | Type | Default | Description |
| --- | --- | --- | --- |
| `file` | file | required | CV file (`.pdf`, `.docx`, or `.txt`) |
| `target_role` | string | `""` | Target job title |
| `output_format` | string | `docx` | `docx` or `pdf` |
| `top_k` | int | `4` | Retrieved chunks per query (1–10) |

DOCX:

```bash
curl -X POST "http://127.0.0.1:8000/optimize" \
  -F "file=@cv.pdf" \
  -F "target_role=Backend Developer" \
  -F "output_format=docx" \
  -F "top_k=4" \
  -o optimized_cv.docx
```

PDF:

```bash
curl -X POST "http://127.0.0.1:8000/optimize" \
  -F "file=@cv.pdf" \
  -F "output_format=pdf" \
  -o optimized_cv.pdf
```

Generated files are written under `outputs/` and returned as a download.

## RAG pipeline

1. Extract plain text from the uploaded CV.
2. Split into overlapping chunks (`chunk_size=900`, `chunk_overlap=150`).
3. Embed chunks with `HashEmbeddings` (512-dimensional normalized bag-of-token vectors).
4. Store embeddings in an ephemeral Chroma collection (one per request).
5. The agent calls `retrieve_cv_context(query)` to fetch the top-`k` relevant chunks.
6. The LLM produces JSON feedback and an improved CV draft grounded in retrieved evidence.

The system prompt instructs the model not to invent jobs, qualifications, or metrics, and to treat retrieved CV text as user data only.

## Quick check

Run the built-in self-check (JSON parsing, env helpers, embeddings):

```bash
python app.py --self-check
```

Expected output: `self-check passed`
