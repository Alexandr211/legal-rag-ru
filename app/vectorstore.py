import os
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.embeddings import embed_texts, embed_query

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_docs_ru")
GK_QDRANT_COLLECTION = os.getenv("GK_QDRANT_COLLECTION", "gk_rf_ru")

client = QdrantClient(url=QDRANT_URL)


def ensure_collection(*, collection_name: str, vector_size: int = 384) -> None:
    """
    Ensure Qdrant collection exists with the expected vector size.
    """
    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    if collection_name not in existing_names:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def index_chunks(doc_id: str, filename: str, chunks: list[str]) -> int:
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    vector_size = len(vectors[0])
    ensure_collection(collection_name=QDRANT_COLLECTION, vector_size=vector_size)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,
                },
            )
        )

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def search_chunks(query: str, limit: int = 5, *, doc_id: str | None = None):
    query_vector = embed_query(query)
    ensure_collection(collection_name=QDRANT_COLLECTION, vector_size=len(query_vector))

    query_filter = None
    if doc_id:
        query_filter = Filter(
            must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
        )

    # In qdrant-client 1.17.x the search API is exposed as `query_points`.
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        query_filter=query_filter,
        limit=limit,
        with_payload=True,
    )
    return response.points


def index_gk_chunks(
    *,
    gk_id: str,
    title: str,
    chunks: list[str],
    article: str | None = None,
) -> int:
    """
    Index chunks of GK RF text into a dedicated Qdrant collection.
    """
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    vector_size = len(vectors[0])
    ensure_collection(collection_name=GK_QDRANT_COLLECTION, vector_size=vector_size)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "source": "gk_rf",
                    "gk_id": gk_id,
                    "title": title,
                    "article": article,
                    "chunk_index": i,
                    "text": chunk,
                },
            )
        )

    client.upsert(collection_name=GK_QDRANT_COLLECTION, points=points)
    return len(points)


def search_gk(query: str, limit: int = 5):
    """
    Search GK RF collection and return scored points.
    """
    query_vector = embed_query(query)
    ensure_collection(collection_name=GK_QDRANT_COLLECTION, vector_size=len(query_vector))
    response = client.query_points(
        collection_name=GK_QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    return response.points


def gk_health() -> dict[str, object]:
    """
    Health info for GK RF collection.
    """
    collections = client.get_collections().collections
    names = {c.name for c in collections}
    exists = GK_QDRANT_COLLECTION in names

    points_count = 0
    if exists:
        res = client.count(collection_name=GK_QDRANT_COLLECTION, exact=True)
        points_count = int(res.count)

    return {
        "collection": GK_QDRANT_COLLECTION,
        "exists": exists,
        "points_count": points_count,
        "qdrant_url": QDRANT_URL,
    }


def list_documents(limit: int = 50) -> list[dict[str, str]]:
    """
    Return a deduplicated list of indexed documents (doc_id + filename).

    Note: This derives metadata from chunk payloads (MVP approach).
    """
    if limit <= 0:
        return []

    # Over-fetch to increase chance of collecting `limit` unique doc_ids.
    fetch_limit = max(200, limit * 20)
    records, _next = client.scroll(
        collection_name=QDRANT_COLLECTION,
        limit=fetch_limit,
        with_payload=True,
        with_vectors=False,
    )

    seen: set[str] = set()
    docs: list[dict[str, str]] = []

    for rec in records:
        payload = rec.payload or {}
        doc_id = payload.get("doc_id")
        filename = payload.get("filename")
        if not doc_id or not filename:
            continue
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append({"doc_id": str(doc_id), "filename": str(filename)})
        if len(docs) >= limit:
            break

    return docs


def get_filename_for_doc_id(doc_id: str) -> str | None:
    """
    Get filename for a doc_id by looking at a single matching payload record.
    """
    query_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
    records, _next = client.scroll(
        collection_name=QDRANT_COLLECTION,
        scroll_filter=query_filter,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if not records:
        return None
    payload = records[0].payload or {}
    filename = payload.get("filename")
    return str(filename) if filename else None


def count_chunks_for_doc_id(doc_id: str) -> int:
    """
    Count how many chunks(points) are indexed for a specific doc_id.
    """
    query_filter = Filter(
        must=[FieldCondition(key="doc_id", match=MatchValue(value=doc_id))]
    )
    res = client.count(
        collection_name=QDRANT_COLLECTION,
        count_filter=query_filter,
        exact=True,
    )
    return int(res.count)

