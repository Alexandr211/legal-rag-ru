from pathlib import Path
import json
import os

from dotenv import load_dotenv
load_dotenv()

from typing import Literal
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.extractors import extract_text
from app.chunking import split_text
from app.vectorstore import (
    index_chunks,
    search_chunks,
    list_documents,
    count_chunks_for_doc_id,
    get_filename_for_doc_id,
    search_gk,
    gk_health,
)
from app.llm import generate_answer
from app.metadata_store import (
    build_metadata,
    init_metadata_db,
    list_documents as list_documents_from_db,
    get_document_metadata,
    upsert_document_metadata,
)


APP_TITLE = "Legal RAG RU"

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/datasets/legal-rag-ru"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_ROOT / "uploads")))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(DATA_ROOT / "processed")))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = PROCESSED_DIR / "metadata.sqlite"

init_metadata_db(DB_PATH)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

app = FastAPI(title=APP_TITLE)


class AskRequest(BaseModel):
    question: str
    limit: int = 5
    doc_id: str | None = None
    include_gk_rf: bool = True
    gk_limit: int = 3
    llm_provider: Literal["ollama", "openai"] = "ollama"


@app.get("/health")
def health():
    return {
        "status": "ok",
        "app": APP_TITLE,
        "upload_dir": str(UPLOAD_DIR),
        "processed_dir": str(PROCESSED_DIR),
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    doc_id = str(uuid.uuid4())
    safe_name = Path(file.filename).name

    original_path = UPLOAD_DIR / f"{doc_id}_{safe_name}"
    with open(original_path, "wb") as f:
        f.write(content)

    try:
        text = extract_text(safe_name, content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text extraction failed: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Could not extract text from file")

    extracted_path = PROCESSED_DIR / f"{doc_id}.txt"
    with open(extracted_path, "w", encoding="utf-8") as f:
        f.write(text)

    chunks = split_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="Could not split text into chunks")

    chunks_path = PROCESSED_DIR / f"{doc_id}.chunks.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    try:
        indexed_count = index_chunks(doc_id=doc_id, filename=safe_name, chunks=chunks)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")

    meta = build_metadata(
        doc_id=doc_id,
        filename=safe_name,
        original_path=str(original_path),
        extracted_text_path=str(extracted_path),
        chunks_path=str(chunks_path),
        chars=len(text),
        chunks_count=len(chunks),
        indexed_count=indexed_count,
    )
    upsert_document_metadata(db_path=DB_PATH, meta=meta)

    preview = text[:1000]

    return {
        "doc_id": doc_id,
        "filename": safe_name,
        "original_path": str(original_path),
        "extracted_text_path": str(extracted_path),
        "chunks_path": str(chunks_path),
        "chars": len(text),
        "chunks_count": len(chunks),
        "indexed_count": indexed_count,
        "preview": preview,
    }


@app.post("/ask")
def ask(payload: AskRequest):
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question is empty")

    try:
        results = search_chunks(question, limit=payload.limit, doc_id=payload.doc_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")

    matches = []
    gk_matches = []
    doc_context_parts = []
    gk_context_parts = []

    for i, item in enumerate(results, start=1):
        text = item.payload.get("text", "")
        filename = item.payload.get("filename")
        chunk_index = item.payload.get("chunk_index")
        doc_id = item.payload.get("doc_id")
        source_id = f"M{i}"

        matches.append(
            {
                "source_id": source_id,
                "score": item.score,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": chunk_index,
                "text": text,
            }
        )

        doc_context_parts.append(
            f"[{source_id}] [Файл: {filename}; doc_id: {doc_id}; chunk: {chunk_index}]\n{text}"
        )

    if payload.include_gk_rf and payload.gk_limit > 0:
        try:
            gk_results = search_gk(question, limit=payload.gk_limit)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"GK RF search failed: {e}")

        for i, item in enumerate(gk_results, start=1):
            p = item.payload or {}
            title = p.get("title")
            article = p.get("article")
            text = p.get("text", "")
            source_id = f"G{i}"

            gk_matches.append(
                {
                    "source_id": source_id,
                    "score": item.score,
                    "gk_id": p.get("gk_id"),
                    "article": article,
                    "title": title,
                    "chunk_index": p.get("chunk_index"),
                    "text": text,
                }
            )
            header = f"[ГК РФ"
            if article:
                header += f"; статья: {article}"
            if title:
                header += f"; {title}"
            header += "]"
            gk_context_parts.append(f"[{source_id}] {header}\n{text}")

    if not doc_context_parts and not gk_context_parts:
        return {
            "question": question,
            "answer": "Не удалось найти релевантные фрагменты в загруженных документах.",
            "matches": [],
            "gk_matches": gk_matches,
        }

    context_sections = []
    if doc_context_parts:
        context_sections.append(
            "ИСТОЧНИКИ ДОГОВОРА (используй ссылки [M#]):\n"
            + "\n\n".join(doc_context_parts)
        )
    if gk_context_parts:
        context_sections.append(
            "ДОПОЛНИТЕЛЬНЫЙ ИСТОЧНИК: ГК РФ (используй ссылки [G#]):\n"
            + "\n\n".join(gk_context_parts)
        )
    context = "\n\n---\n\n".join(context_sections)

    try:
        answer = generate_answer(
            question=question,
            context=context,
            provider=payload.llm_provider,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM generation failed: {e}")

    return {
        "question": question,
        "answer": answer,
        "matches": matches,
        "gk_matches": gk_matches,
        "llm_provider": payload.llm_provider,
    }


@app.get("/documents")
def documents():
    docs = list_documents_from_db(db_path=DB_PATH, limit=50)
    if docs:
        return {"documents": docs}

    # Fallback for existing collections: if metadata DB is empty but Qdrant has payloads.
    return {"documents": list_documents(limit=50)}


@app.get("/documents/{doc_id}")
def document(doc_id: str):
    meta = get_document_metadata(db_path=DB_PATH, doc_id=doc_id)
    if meta is None:
        # Fallback: if this document exists in Qdrant but not yet in SQLite metadata DB,
        # derive minimal info from Qdrant and backfill SQLite.
        filename = get_filename_for_doc_id(doc_id)
        if not filename:
            raise HTTPException(status_code=404, detail="Document not found")

        chunks_count = count_chunks_for_doc_id(doc_id)
        meta_obj = build_metadata(
            doc_id=doc_id,
            filename=filename,
            original_path="",
            extracted_text_path="",
            chunks_path="",
            chars=0,
            chunks_count=chunks_count,
            indexed_count=chunks_count,
        )
        upsert_document_metadata(db_path=DB_PATH, meta=meta_obj)
        meta = get_document_metadata(db_path=DB_PATH, doc_id=doc_id)

    return {"document": meta}


@app.get("/gk/health")
def gk_health_endpoint():
    try:
        return gk_health()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"GK RF health failed: {e}")