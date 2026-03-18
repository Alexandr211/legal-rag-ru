from pathlib import Path
import tempfile
import os

import pypdf
import docx2txt


def extract_text_from_pdf(path: str) -> str:
    reader = pypdf.PdfReader(path)
    parts: list[str] = []
    for page in reader.pages:
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def extract_text_from_docx(path: str) -> str:
    return (docx2txt.process(path) or "").strip()


def extract_text_from_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()


def extract_text(filename: str, content: bytes) -> str:
    suffix = Path(filename).suffix.lower()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        if suffix == ".pdf":
            return extract_text_from_pdf(tmp_path)
        if suffix == ".docx":
            return extract_text_from_docx(tmp_path)
        if suffix == ".txt":
            return extract_text_from_txt(tmp_path)

        raise ValueError(f"Unsupported file type: {suffix}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)