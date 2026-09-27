"""Векторное хранилище на Qdrant — ядро P6.

Qdrant — специализированная векторная БД. В отличие от numpy-поиска в памяти:
  • персистентность (векторы на диске, переживают перезапуск);
  • HNSW-индекс → поиск ~O(log N) вместо перебора всех векторов;
  • payload (метаданные) к каждому вектору + фильтры;
  • масштаб на миллионы, работает как сервис.

Один и тот же код работает в двух режимах:
  • QDRANT_URL задан → подключаемся к «боевому» серверу Qdrant (docker-compose);
  • не задан → локальный движок на диске (без Docker, тот же API).
"""

import os
from qdrant_client import QdrantClient, models

COLLECTION = "documents"
_client = None


def client() -> QdrantClient:
    """Ленивое подключение. URL → сервер; иначе — локальный движок на диске."""
    global _client
    if _client is None:
        url = os.getenv("QDRANT_URL", "").strip()
        if url:
            _client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY") or None)
        else:
            _client = QdrantClient(path=os.getenv("QDRANT_PATH", "./qdrant_data"))
    return _client


def reset(dim: int):
    """Пересоздать коллекцию под размерность эмбеддингов (Cosine-дистанция).
    Коллекция ≈ таблица: хранит точки (id + вектор + payload)."""
    c = client()
    if c.collection_exists(COLLECTION):
        c.delete_collection(COLLECTION)
    c.create_collection(
        COLLECTION,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )


def upsert(texts, vectors):
    """Залить чанки: каждая точка = id + вектор + payload{text}. Qdrant строит HNSW-индекс."""
    points = [
        models.PointStruct(id=i, vector=vectors[i].tolist(), payload={"text": texts[i]})
        for i in range(len(texts))
    ]
    client().upsert(COLLECTION, points=points)


def search(vector, k: int = 4):
    """Топ-k ближайших векторов через HNSW. Возвращает [(score, id, text)]."""
    res = client().query_points(COLLECTION, query=vector.tolist(), limit=k, with_payload=True).points
    return [(float(p.score), int(p.id), p.payload.get("text", "")) for p in res]


def count() -> int:
    c = client()
    return c.count(COLLECTION).count if c.collection_exists(COLLECTION) else 0


def info() -> dict:
    """Инфа о коллекции для дашборда — режим, размерность, метрика, число точек."""
    url = os.getenv("QDRANT_URL", "").strip()
    return {
        "mode": "server" if url else "local (on-disk)",
        "target": url or os.getenv("QDRANT_PATH", "./qdrant_data"),
        "collection": COLLECTION,
        "points": count(),
        "index": "HNSW · Cosine",
    }
