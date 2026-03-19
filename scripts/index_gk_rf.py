import sys
import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.chunking import split_text
from app.vectorstore import index_gk_chunks


def _guess_article_and_title(path: Path) -> tuple[str | None, str]:
    """
    Best-effort metadata extraction from filename.
    Examples:
      "статья_450_расторжение.txt" -> article="450"
      "450.txt" -> article="450"
    """
    stem = path.stem.strip()
    m = re.search(r"\b(\d{1,4})\b", stem)
    article = m.group(1) if m else None
    title = stem.replace("_", " ")
    return article, title


def main() -> None:
    gk_dir = Path(os.getenv("GK_RF_DIR", "/datasets/legal-rag-ru/gk_rf"))
    if not gk_dir.exists():
        raise SystemExit(f"GK_RF_DIR does not exist: {gk_dir}")

    files = sorted([p for p in gk_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".txt", ".md"}])
    if not files:
        raise SystemExit(f"No .txt/.md files found in: {gk_dir}")

    total_points = 0
    for p in files:
        text = p.read_text(encoding="utf-8", errors="ignore").strip()
        if not text:
            continue

        article, title = _guess_article_and_title(p)
        chunks = split_text(text)
        gk_id = str(p.relative_to(gk_dir))
        total_points += index_gk_chunks(gk_id=gk_id, title=title, chunks=chunks, article=article)

    print(f"Indexed GK RF points: {total_points}")


if __name__ == "__main__":
    main()

