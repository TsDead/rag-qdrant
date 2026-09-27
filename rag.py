"""RAG-конвейер поверх Qdrant.

Отличие от in-memory версии — только слой хранения: эмбеддинги уходят в
векторную БД (store.py), а не в numpy-матрицу. Чанкинг, эмбеддинги и генерация
ответа те же. Это и есть смысл P6: вынести ретривал в продакшн-хранилище.
"""

import numpy as np

import llm
import store

_model = None


def _embedder():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")   # 384-мерные векторы, локально
    return _model


def embed(texts):
    """Текст → нормированные векторы (Qdrant всё равно считает Cosine, но так чище)."""
    vecs = np.array(list(_embedder().embed(texts)), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / np.clip(norms, 1e-9, None)


def chunk_text(text, size=90, overlap=20):
    """Режем по словам: ~size слов на чанк с перекрытием overlap."""
    words = text.split()
    if not words:
        return []
    chunks, i, step = [], 0, max(1, size - overlap)
    while i < len(words):
        chunks.append(" ".join(words[i:i + size]))
        i += step
    return chunks


def ingest(text: str) -> int:
    """Индексируем документ В QDRANT: чанки → эмбеддинги → upsert в коллекцию."""
    chunks = chunk_text(text)
    if not chunks:
        return 0
    vecs = embed(chunks)
    store.reset(vecs.shape[1])       # пересоздать коллекцию под размерность (384)
    store.upsert(chunks, vecs)       # залить точки — Qdrant строит HNSW-индекс
    return len(chunks)


SYSTEM = (
    "You answer questions strictly from the provided context passages. "
    "If the answer is not in the context, say you don't know. Be concise. "
    "Cite the passage numbers you used, like [1], [2]."
)


def answer(query: str, k: int = 4):
    """Поиск в Qdrant → сборка контекста → ответ LLM с цитатами."""
    qv = embed([query])[0]
    hits = store.search(qv, k)                       # HNSW-поиск по векторной БД
    if not hits:
        return {"answer": "Сначала загрузите документ.", "sources": []}
    context = "\n\n".join(f"[{n+1}] {text}" for n, (_, _, text) in enumerate(hits))
    user = f"CONTEXT:\n{context}\n\nQUESTION: {query}\n\nANSWER:"
    reply, _ = llm.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        max_tokens=500, temperature=0.1,
    )
    sources = [{"n": n + 1, "score": round(s, 3), "text": t[:220]}
               for n, (s, _, t) in enumerate(hits)]
    return {"answer": reply, "sources": sources}


def stats():
    return store.info()
