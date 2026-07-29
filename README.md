# 🛡️ Smart Insurance Document Router & AgenticIE System

An enterprise-grade, privacy-first, 100% offline Intelligent Document Processing (IDP), RAG Chunking, and Agentic Information Extraction system for insurance documents. Powered by local Ollama LLMs (`qwen2.5`), local ChromaDB vector storage, and a 3-role AgenticIE framework (`Planner`, `Executor`, `Responder`).

---

## 📑 Table of Contents
1. [Executive Summary & Purpose](#-executive-summary--purpose)
2. [High-Level Architecture](#-high-level-architecture)
3. [Project Directory & File Structure](#-project-directory--file-structure)
4. [Deep-Dive Module Breakdown](#-deep-dive-module-breakdown)
   - [Pipeline & Ingestion (`pipeline/`)](#1-pipeline--ingestion-pipeline)
   - [AgenticIE Framework (`agents/`)](#2-agenticie-framework-agents)
   - [Pydantic Schemas & Data Models (`models/`)](#3-pydantic-schemas--data-models-models)
   - [Database & Auth Layer (`db/`)](#4-database--auth-layer-db)
   - [Logging & UI Utilities (`utils/`)](#5-logging--ui-utilities-utils)
5. [FastAPI Endpoints Specification](#-fastapi-endpoints-specification)
6. [Streamlit Web UI Guide](#-streamlit-web-ui-guide)
7. [Docker Containerization Setup](#-docker-containerization-setup)
8. [Getting Started & Local Execution](#-getting-started--local-execution)

---

## 🎯 Executive Summary & Purpose

Insurance document processing requires high precision across diverse, multi-format inputs (e.g. policy schedules, claim intimations, surveyor loss reports, hospital bills, motor repair estimates). Standard OCR and simple chunking strategies often break tabular data, lose document-level context, and produce ungrounded LLM hallucinations.

This project solves these challenges through:
1. **Multi-Format Ingestion with Automatic OCR**: Supports `.pdf`, `.docx`, `.doc`, `.txt`, `.md`, `.png`, `.jpg`, `.jpeg`, `.tiff`, and `.bmp`. Automatically extracts native text layers AND runs PyTesseract OCR on embedded scanned images & hybrid PDF pages deterministically at load time.
2. **2-Stage Structural & Contextual Chunker**:
   - *Stage 1 (Structural)*: Preserves Markdown and tabbed tables intact (`|...|`), tracks hierarchical section breadcrumbs (`"Policy Overview > Coverage Limits"`), and merges micro-chunks (< 80–150 tokens).
   - *Stage 2 (Contextual Enrichment)*: Adapts Anthropic's Contextual Retrieval strategy using a local Ollama model (`qwen2.5`) to inject document & section-level summaries into `vector_text` (~8% token inflation), ensuring high-precision vector retrieval.
3. **100% Local ChromaDB Vector Store**: Disk-persisted vector store at `./chroma_db` using local Ollama embeddings (`nomic-embed-text` / `qwen2.5`).
4. **3-Role AgenticIE Framework**: A Planner-Executor-Responder multi-agent state loop with factual grounding verification to eliminate hallucinations.
5. **Interactive RAG Document Chatbot**: Allows multi-turn follow-up Q&A regarding any uploaded document using ChromaDB vector retrieval.

---

## 🏗️ High-Level Architecture

```mermaid
flowchart TD
    A[Source Document\nPDF, DOCX, TXT, MD, Images] --> B[DocLoader\nAuto OCR & Text Normalization]
    B --> C[2-Stage Contextual Chunker]
    
    subgraph Stage1 [Stage 1: Structural Splitting]
        C1[Header Breadcrumb Parser]
        C2[Table Preservation |...|]
        C3[Micro-Chunk Merger]
    end
    
    subgraph Stage2 [Stage 2: Contextual Enrichment]
        C4[Doc Summary Generation]
        C5[Anthropic Contextual Prompt]
        C6[Enriched vector_text Generation]
    end

    C --> Stage1 --> Stage2 --> D[(ChromaDB Vector Store\nLocal Ollama Embeddings)]

    subgraph AgenticIE [3-Role AgenticIE Loop]
        E[Planner Agent\nDecides Next Action] --> F[Executor Agent\nRuns Tool & ChromaDB Search]
        F --> G[Responder Agent\nFactual Grounding Gate]
        G -- Grounding Check Failed --> E
    end

    D --> AgenticIE
    AgenticIE --> H[FastAPI REST Backend & Streamlit Web UI]
```

---

## 📁 Project Directory & File Structure

```text
Insurance Agent/
├── models/                    # Pydantic Schemas & Data Transfer Objects
│   ├── __init__.py
│   ├── docmodels.py           # Classification & Extraction Pydantic schemas, Domain Metadata
│   ├── agentic_models.py       # PlannerAction, ExecutionStep, AgentState
│   ├── authmodels.py          # Auth Request/Response DTOs
│   └── documents_models.py    # Document storage API response DTOs
│
├── db/                        # Database ORM & Auth Storage Layer
│   ├── __init__.py
│   ├── authdb.py              # User table & SQLAlchemy Engine
│   ├── documentsdb.py         # Document & AgenticExecution tables
│   └── auth.py                # JWT tokens, bcrypt, get_current_user security dependency
│
├── pipeline/                  # Ingestion, Chunking & RAG Vector Engine
│   ├── __init__.py
│   ├── docloader.py           # Multi-format File Loaders & Automatic OCR
│   ├── chunker.py             # 2-Stage Contextual & Structural Chunker
│   ├── docclient.py           # Ollama LLM REST API Client
│   ├── docparsers.py          # JSON parsers & model validators
│   ├── docorchestrator.py     # Two-turn deterministic DocOrchestrator pipeline
│   └── vectorstore.py         # ChromaDB Vector Store & Ollama Embeddings
│
├── agents/                    # 3-Role AgenticIE System
│   ├── __init__.py
│   └── agentic_ie.py          # Planner, Executor, Responder & Orchestrator
│
├── utils/                     # Loggers & UI Helper Utilities
│   ├── __init__.py
│   ├── doc_pipeline_logger.py     # Pipeline JSONL event logger
│   ├── agentic_pipeline_logger.py # Agentic JSONL event logger
│   └── ui_helpers.py              # Streamlit UI display helpers
│
├── tests/                     # Test Suite (pytest)
│   ├── test_chunker.py
│   ├── test_vectorstore.py
│   ├── testorchestrator.py
│   ├── test_agentic_ie.py
│   └── testapp.py
│
├── app.py                     # FastAPI Application Entrypoint
├── authrouter.py              # Auth API Router (/register, /token)
├── streamlit_app.py           # Streamlit Web UI Frontend
├── requirements.txt           # Python dependencies
├── README.md                  # Complete Project Documentation
├── dockerfile                 # Multi-stage Docker Container build
└── docker-compose.yml         # Multi-container orchestration (Backend + Frontend)
```

---

## 🔬 Deep-Dive Module Breakdown

### 1. Pipeline & Ingestion (`pipeline/`)

#### `pipeline/docloader.py`
* **Purpose**: Ingests raw source documents, extracts text layers, performs automatic OCR, and normalizes text.
* **Key Classes & Functions**:
  * `DocLoader`: Central dispatcher routing files to registered `FileTypeLoader` handlers.
  * `PdfFileLoader`: Extracts PDF text layers per page via `pypdf`. Automatically inspects `page.images` for embedded scans or image objects and runs Tesseract OCR (`_perform_ocr_on_bytes`), combining native text with image OCR text.
  * `DocxFileLoader`: Extracts `.docx` paragraphs and table row cells.
  * `TextFileLoader`: Reads `.txt` and `.md` files in UTF-8.
  * `ImageFileLoader`: Ingests `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp` and executes `pytesseract.image_to_string` automatically.
  * `_normalize_text()`: Unicode NFKC normalization, whitespace cleanup, and letter-spacing collapsing (`_collapse_letter_spacing`).

#### `pipeline/chunker.py`
* **Purpose**: Implements the 2-stage structural and contextual retrieval chunker.
* **Stage 1 (`StructuralSplitter`)**:
  * **Header Breadcrumb Parser**: Detects Markdown (`#`) and plain text titles (`SECTION 1`, `ARTICLE II`) to build breadcrumb trails (`["Policy Schedule", "Section A: Benefits"]`).
  * **Table Preservation**: Identifies Markdown (`|...|`) and tabbed table blocks, keeping them intact so rows and columns are never split across chunks.
  * **Recursive Character Splitter**: Splits large blocks by `\n\n`, `\n`, `. `, `; `, or space if text exceeds `target_chunk_tokens` (400 tokens).
  * **Micro-Chunk Merger**: Merges adjacent chunks below `min_chunk_tokens` (80 tokens) sharing the same section breadcrumb trail.
* **Stage 2 (`ContextualEnricher`)**:
  * Prompts local Ollama LLM (`qwen2.5:3b`) using Anthropic's Contextual Retrieval strategy to generate a 1–2 sentence document/section overview summary.
  * Prepends context summary and breadcrumbs into `vector_text` (~8% token inflation), leaving clean text in `content`.
  * Token counting powered by `tiktoken` (`cl100k_base`) with ratio estimation fallback.

#### `pipeline/vectorstore.py`
* **Purpose**: Persistent local vector database management.
* **Key Components**:
  * `VectorStoreManager`: Manages persistent **ChromaDB** client (`chromadb.PersistentClient`) stored on disk at `./chroma_db` for collection `"insurance_chunks"`.
  * `OllamaEmbeddingFunction`: Local vector embeddings via Ollama `/api/embeddings` (`nomic-embed-text` or `qwen2.5`). Includes fallback hash-vector cosine similarity search if ChromaDB native library is uninstalled.
  * `add_chunks()`: Converts Pydantic `ChunkMetadata` into primitive metadata dictionaries (`policy_number`, `claim_number`, `company`, `fiscal_year`, `breadcrumbs`) for vector filtering (`where_filter`).
  * `search()`: Performs cosine similarity vector search returning top-$K$ matching chunks with similarity scores.

#### `pipeline/docclient.py`
* **Purpose**: Lightweight HTTP client wrapping Ollama's local REST API (`/api/generate`).
* **Key Features**:
  * Configured via environment variables (`OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT`).
  * Tracks exact model token counts (`prompt_eval_count`, `eval_count`) and wall-clock inference time.
  * `health()`: Validates Ollama server reachability via `/api/tags`.

#### `pipeline/docparsers.py`
* **Purpose**: Converts raw LLM output text into validated Pydantic model instances.
* **Key Functions**:
  * `_extract_json()`: Extracts JSON objects from fenced code blocks (` ```json...``` `) or raw text.
  * `parse_classification()`: Validates classification output against `ClassificationResult`.
  * `parse_extraction()`: Validates extraction output against `EXTRACTION_MODEL_MAP` or `GenericInsuranceFields`.

#### `pipeline/docorchestrator.py`
* **Purpose**: Executes the two-turn deterministic classification and extraction pipeline with retries and JSONL event logging.

---

### 2. AgenticIE Framework (`agents/`)

#### `agents/agentic_ie.py`
Based on the **AgenticIE** paper for multi-turn document information extraction and grounded Q&A.

```
                  ┌─────────────────────────────┐
                  │         AgentState          │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                  ┌─────────────────────────────┐
                  │       1. Planner Agent      │
                  │ Selects next optimal tool   │
                  └──────────────┬──────────────┘
                                 │ PlannerAction
                                 ▼
                  ┌─────────────────────────────┐
                  │       2. Executor Agent     │
                  │ Runs extraction & ChromaDB  │
                  └──────────────┬──────────────┘
                                 │ Mutates State
                                 ▼
                  ┌─────────────────────────────┐
                  │      3. Responder Agent     │
                  │  Factual Grounding Gate     │
                  └──────────────┬──────────────┘
                                 │
                 Passed? ────────┴──────── Failed?
                    │                         │
                    ▼                         ▼
                 [FINISH]           [Route Back to Planner]
```

* **`Planner Agent`**: Inspects `AgentState` and selects the next `PlannerToolName`:
  * `CLASSIFY_DOCUMENT`: Turn 1 classification.
  * `EXTRACT_FIXED_FIELDS`: Turn 2 fixed schema extraction.
  * `EXTRACT_OPEN_SCHEMA`: Dynamic key-value pair and table discovery.
  * `RETRIEVE_CHUNKS`: ChromaDB vector search.
  * `ANSWER_QUESTION`: Grounded QA generation.
  * `FINISH`: Mark complete.
* **`Executor Agent`**: Executes the tool, mutates `AgentState`, and indexes/queries ChromaDB chunks.
* **`Responder Agent`**: Factual grounding gate evaluating extracted schemas and QA answers against raw source text. Performs number/amount regex checks + LLM verification prompt (`RESPONDER_VERIFICATION_PROMPT`). If ungrounded claims are found, routes feedback back to Planner for retry.

---

### 3. Pydantic Schemas & Data Models (`models/`)

* **`models/docmodels.py`**:
  * Enums: `DocumentCategory` (15 categories), `DocumentType` (40+ types), `CATEGORY_TYPE_MAP`.
  * Schemas: `ClassificationResult`, `PolicyFields`, `ClaimFields`, `SurveyReportFields`, `PremiumBillingFields`, `MedicalReportFields`, `GenericInsuranceFields`, `ExtractionResult`, `PipelineResult`.
  * Metadata Schemas: `InsuranceDomainMetadata`, `DocumentFileMetadata`, `PipelineAuditMetadata`, `EnrichedDocumentPayload`.
* **`models/agentic_models.py`**:
  * `PlannerToolName`, `PlannerAction`, `ExecutionStep`, `OpenSchemaResult`, `VerificationResult`, `AgentState`.
* **`models/authmodels.py`**:
  * `UserCreate`, `UserOut`, `Token`, `TokenData`.
* **`models/documents_models.py`**:
  * `DocumentSummary`, `DocumentDetail`.

---

### 4. Database & Auth Layer (`db/`)

* **`db/authdb.py`**: SQLite database setup (`auth.db`) via SQLAlchemy. Defines `User` table, `create_user()`, and `get_user_by_username()`.
* **`db/documentsdb.py`**: Defines `Document` and `AgenticExecution` tables. Functions: `save_document()`, `save_agentic_state()`, `list_documents_for_user()`, `get_document_for_user()`.
* **`db/auth.py`**: Password hashing via `bcrypt`, JWT token generation & validation via `python-jose`, and `get_current_user` security dependency.

---

### 5. Logging & UI Utilities (`utils/`)

* **`utils/doc_pipeline_logger.py`**: Thread-safe `JsonlPipelineLogger` logging events to `pipeline_events.jsonl`.
* **`utils/agentic_pipeline_logger.py`**: Thread-safe `AgenticJsonlLogger` logging agentic execution steps to `agentic_pipeline_events.jsonl`.
* **`utils/ui_helpers.py`**: Formatting utilities for displaying extraction fields, token metrics, and document details in Streamlit.

---

## ⚡ FastAPI Endpoints Specification

| Method | Endpoint | Protection | Description |
| :--- | :--- | :--- | :--- |
| `POST` | `/register` | Public | Registers a new user (`UserCreate`). |
| `POST` | `/token` | Public | OAuth2 login form returning JWT `Token`. |
| `GET` | `/health` | Public | Liveness check & Ollama reachability status. |
| `POST` | `/classify` | Bearer Token | Runs 2-turn deterministic pipeline (`DocOrchestrator`). |
| `POST` | `/agentic/process` | Bearer Token | Runs 3-role AgenticIE pipeline (`AgenticIEOrchestrator`). |
| `POST` | `/agentic/chat` | Bearer Token | RAG Document Chatbot endpoint querying ChromaDB & Ollama. |
| `GET` | `/agentic/executions`| Bearer Token | Lists agentic execution history for logged-in user. |
| `GET` | `/documents` | Bearer Token | Lists stored documents for logged-in user (`DocumentSummary`). |
| `GET` | `/documents/{id}` | Bearer Token | Returns full document details (`DocumentDetail`). |

---

## 🖥️ Streamlit Web UI Guide

The Streamlit UI (`streamlit_app.py`) provides an interactive interface:

1. **📥 Classify (Deterministic)**: Upload documents to run the deterministic two-turn pipeline.
2. **🤖 AgenticIE Router & QA**: Upload documents to trigger the 3-role AgenticIE Planner, Executor & Responder loop with dynamic open schema discovery and grounded QA.
3. **💬 Document QA & Interactive Chatbot**: Under the AgenticIE QA tab, click **`💬 Enable Interactive Chatbot Interface`** to chat with any document via local ChromaDB RAG vector retrieval. Inspect retrieved vector chunks, breadcrumbs, and similarity scores per chat turn.
4. **🧩 Contextual Chunks**: Displays distinct chunks, hierarchical section breadcrumbs, table preservation badges (`📊 [Table Preserved Intact]`), token inflation metrics, raw `content` vs enriched `vector_text`, and vector similarity search.
5. **📁 My Documents**: View and search previously processed documents.

---

## 🐳 Docker Containerization Setup

### 1. `docker-compose.yml`
Orchestrates two services:
* **`backend`**: FastAPI app running on port `8000`. Configured with `host.docker.internal` to connect to local Ollama on host (`http://host.docker.internal:11434`). Binds persistent volumes for `auth.db`, `chroma_db`, `uploads`, and `logs`.
* **`frontend`**: Streamlit UI running on port `8501`. Connects internally to `http://backend:8000`.

### 2. Launch Commands:
```bash
# 1. Ensure local Ollama is running on host
ollama serve

# 2. Build and start containers
docker-compose up --build
```
* Access Streamlit UI: `http://localhost:8501`
* Access API Docs: `http://localhost:8000/docs`

---

## 🚀 Getting Started & Local Execution

### Prerequisites
* Python 3.11+
* [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on host system
* [Ollama](https://ollama.ai/) installed and running locally

### Local Step-by-Step Setup

1. **Clone repository & set up environment**:
   ```bash
   git clone <repo_url>
   cd "Insurance Agent"
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Pull Ollama Models**:
   ```bash
   ollama pull qwen2.5:3b
   ollama pull nomic-embed-text
   ```

3. **Set Environment Variables**:
   ```bash
   export SECRET_KEY="insurance-agent-super-secret-jwt-key"
   export OLLAMA_BASE_URL="http://localhost:11434"
   export OLLAMA_MODEL="qwen2.5:3b"
   ```

4. **Run FastAPI Backend**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **Run Streamlit Frontend**:
   ```bash
   streamlit run streamlit_app.py
   ```

6. **Run Test Suite**:
   ```bash
   python -m pytest tests/
   ```
