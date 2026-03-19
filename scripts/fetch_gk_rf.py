#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Скачивает и парсит ГК РФ (все 4 части) с Consultant.ru в:
- raw HTML
- consolidated TXT
- structured JSON
- per-article TXT (удобно для дальнейшей индексации в Qdrant)

Использует 4 публичные страницы Consultant.ru для частей I–IV ГК РФ:
  I  — cons_doc_LAW_5142
  II — cons_doc_LAW_9027
  III— cons_doc_LAW_34154
  IV — cons_doc_LAW_64629

Запуск:
  source .venv/bin/activate
  set -a; source .env; set +a
  python scripts/fetch_gk_rf.py

Зависимости:
  pip install requests beautifulsoup4 lxml
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


PART_URLS = [
    {
        "part": 1,
        "title": "Гражданский кодекс Российской Федерации. Часть первая",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_5142/",
    },
    {
        "part": 2,
        "title": "Гражданский кодекс Российской Федерации. Часть вторая",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_9027/",
    },
    {
        "part": 3,
        "title": "Гражданский кодекс Российской Федерации. Часть третья",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_34154/",
    },
    {
        "part": 4,
        "title": "Гражданский кодекс Российской Федерации. Часть четвертая",
        "url": "https://www.consultant.ru/document/cons_doc_LAW_64629/",
    },
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

STOP_MARKERS = [
    "Открыть полный текст документа",
    "Гражданский кодекс (ГК РФ)",
    "Жилищный кодекс (ЖК РФ)",
    "Налоговый кодекс (НК РФ)",
    "Трудовой кодекс (ТК РФ)",
    "Уголовный кодекс (УК РФ)",
]

ARTICLE_RE = re.compile(r"^Статья\s+(\d+(?:\.\d+)?)\.\s*(.*)$")
PART_RE = re.compile(r"^ЧАСТЬ\s+(.+)$")
SECTION_RE = re.compile(r"^Раздел\s+([IVXLC]+)\.?\s*(.*)$", re.IGNORECASE)
SUBSECTION_RE = re.compile(r"^Подраздел\s+(\d+)\.?\s*(.*)$", re.IGNORECASE)
CHAPTER_RE = re.compile(r"^Глава\s+(\d+(?:\.\d+)?)\.?\s*(.*)$", re.IGNORECASE)
PARAGRAPH_RE = re.compile(r"^§\s*(\d+(?:\.\d+)?)\.?\s*(.*)$")
LINE_NUMBER_RE = re.compile(r"^\s*\d+\s*$")

PART_WORD_RE = re.compile(r"^ЧАСТЬ\s+(ПЕРВАЯ|ВТОРАЯ|ТРЕТЬЯ|ЧЕТВЕРТАЯ)\s*$", re.IGNORECASE)

_PART_WORD_TO_NUM = {
    "ПЕРВАЯ": 1,
    "ВТОРАЯ": 2,
    "ТРЕТЬЯ": 3,
    "ЧЕТВЕРТАЯ": 4,
}

@dataclass
class ArticleRecord:
    part: int
    part_title: str
    section: Optional[str]
    subsection: Optional[str]
    chapter: Optional[str]
    paragraph: Optional[str]
    article_number: str
    article_title: str
    text: str
    source_url: str


def fetch_html(session: requests.Session, url: str) -> str:
    resp = session.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    resp.encoding = resp.encoding or "utf-8"
    return resp.text


def normalize_line(line: str) -> str:
    line = line.replace("\xa0", " ")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def html_to_lines(html: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text("\n")
    raw_lines = [normalize_line(x) for x in text.splitlines()]
    return [x for x in raw_lines if x]


def crop_relevant_text(lines: list[str]) -> list[str]:
    start_idx = 0
    for i, line in enumerate(lines):
        if line == "ГРАЖДАНСКИЙ КОДЕКС РОССИЙСКОЙ ФЕДЕРАЦИИ":
            start_idx = i
            break

    cropped = lines[start_idx:]

    end_idx = len(cropped)
    for i, line in enumerate(cropped):
        if any(marker in line for marker in STOP_MARKERS):
            end_idx = i
            break

    cropped = cropped[:end_idx]

    cleaned: list[str] = []
    for line in cropped:
        if LINE_NUMBER_RE.match(line):
            continue
        if line in {
            "* * *",
            "Принят",
            "Одобрен",
            "Государственной Думой",
            "Советом Федерации",
        }:
            cleaned.append(line)
            continue
        cleaned.append(line)

    return cleaned


def cleanup_article_text(text: str) -> str:
    lines = [normalize_line(x) for x in text.splitlines()]
    lines = [x for x in lines if x]

    filtered: list[str] = []
    for line in lines:
        if any(marker in line for marker in STOP_MARKERS):
            break
        if line.startswith("Список изменяющих документов"):
            continue
        if line.startswith("(в ред. Федеральных законов"):
            continue
        filtered.append(line)

    return "\n".join(filtered).strip()


def parse_articles(lines: list[str], part_num: int, part_url: str) -> list[ArticleRecord]:
    current_part_title: str = ""
    current_section: Optional[str] = None
    current_subsection: Optional[str] = None
    current_chapter: Optional[str] = None
    current_paragraph: Optional[str] = None

    articles: list[ArticleRecord] = []
    current_article: Optional[ArticleRecord] = None
    buffer: list[str] = []

    def flush_current() -> None:
        nonlocal current_article, buffer
        if current_article is None:
            return
        article_text = "\n".join(buffer).strip()
        current_article.text = cleanup_article_text(article_text)
        articles.append(current_article)
        current_article = None
        buffer = []

    for line in lines:
        if line == "ГРАЖДАНСКИЙ КОДЕКС РОССИЙСКОЙ ФЕДЕРАЦИИ":
            continue

        m = PART_RE.match(line)
        if m:
            flush_current()
            current_part_title = f"ЧАСТЬ {m.group(1).strip()}"
            continue

        m = SECTION_RE.match(line)
        if m:
            flush_current()
            roman = m.group(1).strip()
            tail = m.group(2).strip()
            current_section = f"Раздел {roman}. {tail}".strip().rstrip(".")
            current_subsection = None
            current_chapter = None
            current_paragraph = None
            continue

        m = SUBSECTION_RE.match(line)
        if m:
            flush_current()
            n = m.group(1).strip()
            tail = m.group(2).strip()
            current_subsection = f"Подраздел {n}. {tail}".strip().rstrip(".")
            current_chapter = None
            current_paragraph = None
            continue

        m = CHAPTER_RE.match(line)
        if m:
            flush_current()
            n = m.group(1).strip()
            tail = m.group(2).strip()
            current_chapter = f"Глава {n}. {tail}".strip().rstrip(".")
            current_paragraph = None
            continue

        m = PARAGRAPH_RE.match(line)
        if m:
            flush_current()
            n = m.group(1).strip()
            tail = m.group(2).strip()
            current_paragraph = f"§ {n}. {tail}".strip().rstrip(".")
            continue

        m = ARTICLE_RE.match(line)
        if m:
            flush_current()
            article_number = m.group(1).strip()
            article_title = m.group(2).strip()
            current_article = ArticleRecord(
                part=part_num,
                part_title=current_part_title,
                section=current_section,
                subsection=current_subsection,
                chapter=current_chapter,
                paragraph=current_paragraph,
                article_number=article_number,
                article_title=article_title,
                text="",
                source_url=part_url,
            )
            buffer = []
            continue

        if current_article is not None:
            buffer.append(line)

    flush_current()
    return articles


def write_txt(root: Path, all_articles: list[ArticleRecord]) -> Path:
    out_path = root / "output" / "gk_rf_all_parts.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    parts: list[str] = []

    for a in all_articles:
        header = [
            f"=== ЧАСТЬ {a.part}: {a.part_title} ===",
            a.section or "",
            a.subsection or "",
            a.chapter or "",
            a.paragraph or "",
            f"Статья {a.article_number}. {a.article_title}",
            f"Источник: {a.source_url}",
            "",
            a.text,
            "",
            "-" * 80,
            "",
        ]
        parts.append("\n".join(x for x in header if x is not None))

    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def write_json(root: Path, all_articles: list[ArticleRecord]) -> Path:
    out_path = root / "output" / "gk_rf_all_parts.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(a) for a in all_articles]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_articles(root: Path, all_articles: list[ArticleRecord]) -> int:
    """
    Writes each article to a separate text file for downstream indexing.
    """
    out_dir = root / "articles"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0

    for a in all_articles:
        safe_num = a.article_number.replace("/", "_")
        file_name = f"part{a.part}_article_{safe_num}.txt"
        p = out_dir / file_name
        body = [
            f"ГК РФ — Часть {a.part}: {a.part_title}",
            a.section or "",
            a.subsection or "",
            a.chapter or "",
            a.paragraph or "",
            f"Статья {a.article_number}. {a.article_title}",
            f"Источник: {a.source_url}",
            "",
            a.text,
            "",
        ]
        p.write_text("\n".join(x for x in body if x is not None).strip() + "\n", encoding="utf-8")
        written += 1

    return written


def save_raw_html(root: Path, part: int, html: str) -> Path:
    raw_dir = root / "raw_html"
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / f"gk_part_{part}.html"
    path.write_text(html, encoding="utf-8")
    return path


def infer_part_num(lines: list[str]) -> int | None:
    """
    Infer GK part number from extracted text lines.
    """
    for line in lines[:400]:
        m = PART_WORD_RE.match(line)
        if m:
            key = m.group(1).upper()
            return _PART_WORD_TO_NUM.get(key)
    return None


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=os.getenv("GK_RF_DIR", "/datasets/legal-rag-ru/gk_rf"),
        help="Output directory (defaults to GK_RF_DIR).",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=1.5,
        help="Sleep seconds between downloads (politeness).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if raw HTML already exists.",
    )
    ap.add_argument(
        "--local-raw",
        action="store_true",
        help=(
            "Parse existing HTML files from <out>/raw_html (any names) instead of downloading."
        ),
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    all_articles: list[ArticleRecord] = []

    raw_dir = root / "raw_html"

    # Mode A: parse any existing raw HTML files (manually added) from raw_html/.
    if args.local_raw:
        raw_files = sorted(
            [p for p in raw_dir.glob("*") if p.is_file() and p.suffix.lower() in {".html", ".htm"}]
        )
        if not raw_files:
            raise SystemExit(f"No .html/.htm files found in: {raw_dir}")

        by_part: dict[int, Path] = {}
        for p in raw_files:
            html = p.read_text(encoding="utf-8", errors="ignore")
            lines = crop_relevant_text(html_to_lines(html))
            part_num = infer_part_num(lines)
            if part_num is None:
                print(f"[WARN] Could not infer part number from: {p.name} (skipping)")
                continue
            if part_num in by_part:
                print(f"[WARN] Duplicate HTML for part {part_num}: {p.name} (keeping first: {by_part[part_num].name})")
                continue
            by_part[part_num] = p

        missing = [n for n in (1, 2, 3, 4) if n not in by_part]
        if missing:
            raise SystemExit(
                f"Could not find raw HTML for parts: {missing}. "
                f"Place 4 files in {raw_dir} containing lines like 'ЧАСТЬ ПЕРВАЯ' etc."
            )

        for part in (1, 2, 3, 4):
            p = by_part[part]
            html = p.read_text(encoding="utf-8", errors="ignore")
            lines = crop_relevant_text(html_to_lines(html))
            source = f"local_raw_html:{p}"
            articles = parse_articles(lines, part_num=part, part_url=source)
            print(f"[INFO] Parsed articles from part {part}: {len(articles)} ({p.name})")
            all_articles.extend(articles)

    # Mode B: download from public URLs, caching as gk_part_{n}.html in raw_html/.
    else:
        session = requests.Session()
        for item in PART_URLS:
            part = item["part"]
            title = item["title"]
            url = item["url"]

            raw_path = raw_dir / f"gk_part_{part}.html"
            if raw_path.exists() and not args.force:
                html = raw_path.read_text(encoding="utf-8", errors="ignore")
                print(f"[INFO] Using cached raw HTML for part {part}: {raw_path}")
            else:
                print(f"[INFO] Downloading part {part}: {title}")
                html = fetch_html(session, url)
                saved = save_raw_html(root, part, html)
                print(f"[INFO] Saved raw HTML: {saved}")
                time.sleep(args.sleep)

            lines = crop_relevant_text(html_to_lines(html))
            articles = parse_articles(lines, part_num=part, part_url=url)
            print(f"[INFO] Parsed articles from part {part}: {len(articles)}")
            all_articles.extend(articles)

    txt_path = write_txt(root, all_articles)
    json_path = write_json(root, all_articles)
    written = write_articles(root, all_articles)

    print("[DONE]")
    print(f"Articles total: {len(all_articles)}")
    print(f"Per-article files: {written} -> {root / 'articles'}")
    print(f"TXT:  {txt_path}")
    print(f"JSON: {json_path}")
    print(f"HTML: {root / 'raw_html'}")


if __name__ == "__main__":
    main()

