from pathlib import Path
import os
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.extractors import extract_text


APP_TITLE = "Legal RAG RU"

DATA_ROOT = Path(os.getenv("DATA_ROOT", "/datasets/legal-rag-ru"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", str(DATA_ROOT / "uploads")))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", str(DATA_ROOT / "processed")))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

app = FastAPI(title=APP_TITLE)


class AskRequest(BaseModel):
    question: str


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

    preview = text[:1000]

    return {
        "doc_id": doc_id,
        "filename": safe_name,
        "original_path": str(original_path),
        "extracted_text_path": str(extracted_path),
        "chars": len(text),
        "preview": preview,
    }


@app.post("/ask")
def ask(payload: AskRequest):
    return {
        "answer": f"Пока заглушка. Вопрос получен: {payload.question}"
    }