import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DocumentMetadata:
    doc_id: str
    filename: str
    original_path: str
    extracted_text_path: str
    chunks_path: str
    chars: int
    chunks_count: int
    indexed_count: int
    uploaded_at: str


def _connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn


def init_metadata_db(db_path: Path) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
              doc_id TEXT PRIMARY KEY,
              filename TEXT NOT NULL,
              original_path TEXT NOT NULL,
              extracted_text_path TEXT NOT NULL,
              chunks_path TEXT NOT NULL,
              chars INTEGER NOT NULL,
              chunks_count INTEGER NOT NULL,
              indexed_count INTEGER NOT NULL,
              uploaded_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def upsert_document_metadata(db_path: Path, meta: DocumentMetadata) -> None:
    conn = _connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO documents (
              doc_id, filename, original_path, extracted_text_path, chunks_path,
              chars, chunks_count, indexed_count, uploaded_at
            )
            VALUES (
              :doc_id, :filename, :original_path, :extracted_text_path, :chunks_path,
              :chars, :chunks_count, :indexed_count, :uploaded_at
            )
            ON CONFLICT(doc_id) DO UPDATE SET
              filename=excluded.filename,
              original_path=excluded.original_path,
              extracted_text_path=excluded.extracted_text_path,
              chunks_path=excluded.chunks_path,
              chars=excluded.chars,
              chunks_count=excluded.chunks_count,
              indexed_count=excluded.indexed_count,
              uploaded_at=excluded.uploaded_at
            """,
            meta.__dict__,
        )
        conn.commit()
    finally:
        conn.close()


def list_documents(db_path: Path, limit: int = 50) -> list[dict[str, Any]]:
    conn = _connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT doc_id, filename
            FROM documents
            ORDER BY uploaded_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [{"doc_id": r["doc_id"], "filename": r["filename"]} for r in rows]
    finally:
        conn.close()


def get_document_metadata(db_path: Path, doc_id: str) -> dict[str, Any] | None:
    conn = _connect(db_path)
    try:
        row = conn.execute(
            """
            SELECT
              doc_id, filename, original_path, extracted_text_path, chunks_path,
              chars, chunks_count, indexed_count, uploaded_at
            FROM documents
            WHERE doc_id = ?
            """,
            (doc_id,),
        ).fetchone()

        if row is None:
            return None

        return {
            "doc_id": row["doc_id"],
            "filename": row["filename"],
            "original_path": row["original_path"],
            "extracted_text_path": row["extracted_text_path"],
            "chunks_path": row["chunks_path"],
            "chars": row["chars"],
            "chunks_count": row["chunks_count"],
            "indexed_count": row["indexed_count"],
            "uploaded_at": row["uploaded_at"],
        }
    finally:
        conn.close()


def build_metadata(
    *,
    doc_id: str,
    filename: str,
    original_path: str,
    extracted_text_path: str,
    chunks_path: str,
    chars: int,
    chunks_count: int,
    indexed_count: int,
) -> DocumentMetadata:
    return DocumentMetadata(
        doc_id=doc_id,
        filename=filename,
        original_path=original_path,
        extracted_text_path=extracted_text_path,
        chunks_path=chunks_path,
        chars=chars,
        chunks_count=chunks_count,
        indexed_count=indexed_count,
        uploaded_at=datetime.now(timezone.utc).isoformat(),
    )

