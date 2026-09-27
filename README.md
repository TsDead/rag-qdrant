# RAG · Qdrant — semantic search over your documents

Retrieval-Augmented Generation with a **real vector database (Qdrant)** instead
of an in-memory search. Chunk a document → store embeddings in Qdrant → answer
questions strictly from the retrieved passages, with citations.

**Stack:** Python · FastAPI · Qdrant · fastembed (local embeddings) · LLM (Groq) · vanilla-JS UI

## Why a vector DB (vs in-memory numpy)

| | In-memory (numpy) | Qdrant |
|---|---|---|
| Search | brute-force `V @ q`, **O(N)** | **HNSW** approximate NN, ~**O(log N)** |
| Persistence | lost on restart | on disk / server |
| Metadata & filters | manual | payloads + filters built in |
| Scale | a few hundred chunks | millions, as a service |

**HNSW** (Hierarchical Navigable Small World) is a graph index you *navigate*
to jump near the answer fast — trading a hair of accuracy for a huge speed-up.
This is what production RAG uses.

## How it works

```
document ─► chunk ─► embed (fastembed, 384-dim) ─► upsert into Qdrant collection
                                                        │ (HNSW · Cosine)
question ─► embed ─► Qdrant search (top-k) ─► passages ─► LLM ─► answer + citations
```

- `store.py` — Qdrant wrapper (collection, upsert, search, info).
- `rag.py` — chunking, embeddings, ingest→upsert, answer→search+LLM.
- `app.py` — FastAPI endpoints; `static/index.html` — UI with a live vector-DB status strip.

## Run

```bash
pip install -r requirements.txt
cp .env.example .env          # add a free Groq key
uvicorn app:app --reload      # http://127.0.0.1:8000
```

### Vector store — two modes, same code
- **Local (default):** no Docker — `qdrant-client` runs an on-disk engine at `./qdrant_data`.
- **Server (production):** `docker compose up -d`, then set `QDRANT_URL=http://localhost:6333` in `.env`.

## API
- `GET  /api/stats` — vector-store info (mode, collection, points, index)
- `POST /api/ingest` `{text}` → chunks the doc and upserts embeddings into Qdrant
- `POST /api/upload` — .txt / .pdf upload
- `POST /api/ask` `{question, k}` → `{answer, sources[]}` (each source has a similarity score)

---
© 2026 NOVACODE · [new-coder.ru](https://new-coder.ru)
