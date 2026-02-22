# Sharjah HR RAG Assistant

A Retrieval-Augmented Generation (RAG) project that answers HR policy questions from the **Sharjah HR law document** using:

- a **FastAPI** backend,
- a **React + Vite** frontend,
- **Qdrant** local vector stores,
- **BAAI embeddings + reranker** for retrieval quality,
- and **OpenAI GPT** for final answer generation.

The assistant supports both **Arabic and English** queries and can return related **table images** when the answer is grounded in tabular data.

---

## Project structure

```text
.
├── backend/
│   ├── api.py                     # FastAPI app and /api/ask endpoint
│   ├── etl/                       # Data preparation and indexing scripts
│   ├── data/                      # Generated ETL outputs (JSON logs)
│   ├── db/                        # Local Qdrant DB paths (created at runtime)
│   └── requirements.txt
├── frontend/
│   ├── src/App.jsx                # Main UI
│   └── package.json
├── materials/
│   ├── Canonicalization/          # Table text sources
│   ├── Normal_Text_Extraction/    # Extracted non-table page text
│   └── tables_images/             # Table images served by backend
├── sharjah_hr_law 8.pdf
└── README.md
```

---

## How it works

1. User asks a question from the frontend.
2. Backend embeds the query with `BAAI/bge-m3`.
3. Backend retrieves candidates from:
   - table collection (`hr_tables`)
   - text collection (`hr_text_hybrid`)
4. Candidates are reranked with `BAAI/bge-reranker-v2-m3`.
5. Top context is sent to OpenAI (`gpt-4o`) with language-aware prompt rules.
6. Backend returns:
   - generated answer,
   - context caption,
   - table image URLs (when relevant).

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- npm
- OpenAI API key

> First run may take time because embedding/reranker models are downloaded.

---

## Environment setup

Create a `.env` file in the **project root**:

```bash
OPENAI_API_KEY=your_api_key_here
```

---

## Backend setup

From the project root:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install fastapi uvicorn python-dotenv numpy requests pdfplumber python-bidi
```

Run the API:

```bash
python api.py
```

The backend runs on `http://localhost:8000`.

---

## Frontend setup

From the project root:

```bash
cd frontend
npm install
npm run dev
```

Frontend default URL is shown by Vite (usually `http://localhost:5173`).

---

## ETL and indexing pipeline (optional rebuild)

If you need to regenerate text/table vectors and local Qdrant collections, run scripts in this order from `backend/etl`:

```bash
python text_1_extract.py
python text_2_chunk.py
python text_3_embed.py
python text_4_load_db.py
python tables_1_embed.py
```

Generated artifacts are written under `backend/data/` and local Qdrant storage under `backend/db/`.

---
