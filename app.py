"""RAG на Qdrant — FastAPI. Загрузи документ (текст/PDF) → задавай вопросы →
ответы с опорой на источник. Ретривал живёт в векторной БД Qdrant, а не в памяти.
"""

from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import rag
import llm

app = FastAPI(title="RAG · Qdrant")
HTML = (Path(__file__).parent / "static" / "index.html").read_text(encoding="utf-8")
SAMPLE = (Path(__file__).parent / "eval" / "sample_doc.txt").read_text(encoding="utf-8")


class IngestReq(BaseModel):
    text: str


class AskReq(BaseModel):
    question: str
    k: int = 4


@app.get("/", response_class=HTMLResponse)
def index():
    return HTML


@app.get("/api/sample")
def sample():
    return {"text": SAMPLE}


@app.get("/api/stats")
def stats():
    return rag.stats()


@app.post("/api/ingest")
def ingest(r: IngestReq):
    if not r.text.strip():
        return {"error": "Пустой документ."}
    return {"chunks": rag.ingest(r.text), "store": rag.stats()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    raw = await file.read()
    name = (file.filename or "").lower()
    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io
            reader = PdfReader(io.BytesIO(raw))
            text = "\n".join((p.extract_text() or "") for p in reader.pages)
        except Exception as e:
            return {"error": f"Не смог прочитать PDF: {e}"}
    else:
        text = raw.decode("utf-8", errors="ignore")
    if not text.strip():
        return {"error": "В файле не нашлось текста."}
    return {"chunks": rag.ingest(text), "chars": len(text), "store": rag.stats()}


@app.post("/api/ask")
def ask(r: AskReq):
    if not r.question.strip():
        return {"error": "Пустой вопрос."}
    if not llm.available():
        return {"error": "Не задан GROQ_API_KEY в .env."}
    if not rag.stats()["points"]:
        return {"error": "Сначала загрузите документ."}
    return rag.answer(r.question, max(1, min(r.k, 8)))
