#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Парсит локальные TXT-файлы ГК РФ (например, скачанные из КонсультантПлюс) в:
- consolidated TXT
- structured JSON (по статьям)
- per-article TXT (для индексации)

Пример:
  source .venv/bin/activate
  set -a; source .env; set +a
  python scripts/parse_gk_rf_txt.py \
    "/home/alexandr/Downloads/Гражданский кодекс Российской Федерации (часть первая) от 3.txt" \
    "/home/alexandr/Downloads/Гражданский кодекс Российской Федерации (часть вторая) от 2.txt" \
    "/home/alexandr/Downloads/Гражданский кодекс Российской Федерации (часть третья) от 2.txt" \
    "/home/alexandr/Downloads/Гражданский кодекс Российской Федерации (часть четвертая) о.txt"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


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

PART_NUM_RE = re.compile(r"\(часть\s+(первая|вторая|третья|четвертая)\)", re.IGNORECASE)


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


def normalize_line(line: str) -> str:
    line = line.replace("\xa0", " ")
    line = re.sub(r"[ \t]+", " ", line)
    return line.strip()


def txt_to_lines(text: str) -> list[str]:
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


def parse_articles(lines: list[str], part_num: int, source: str) -> list[ArticleRecord]:
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
                source_url=source,
            )
            buffer = []
            continue

        if current_article is not None:
            buffer.append(line)

    flush_current()
    return articles


def _infer_part_num(path: Path, lines: list[str]) -> int:
    m = PART_NUM_RE.search(path.name)
    if m:
        word = m.group(1).lower()
        return {"первая": 1, "вторая": 2, "третья": 3, "четвертая": 4}[word]

    for line in lines[:200]:
        if line == "ЧАСТЬ ПЕРВАЯ":
            return 1
        if line == "ЧАСТЬ ВТОРАЯ":
            return 2
        if line == "ЧАСТЬ ТРЕТЬЯ":
            return 3
        if line == "ЧАСТЬ ЧЕТВЕРТАЯ":
            return 4

    raise SystemExit(f"Cannot infer GK part number from: {path}")


def write_txt(root: Path, all_articles: list[ArticleRecord]) -> Path:
    out_path = root / "output" / "gk_rf_all_parts_from_txt.txt"
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
    out_path = root / "output" / "gk_rf_all_parts_from_txt.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(a) for a in all_articles]
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_articles(root: Path, all_articles: list[ArticleRecord]) -> int:
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


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "inputs",
        nargs="+",
        help="Input TXT files for parts I–IV (any order).",
    )
    ap.add_argument(
        "--out",
        default=os.getenv("GK_RF_DIR", "/datasets/legal-rag-ru/gk_rf"),
        help="Output directory (defaults to GK_RF_DIR).",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)

    all_articles: list[ArticleRecord] = []

    for in_path_str in args.inputs:
        p = Path(in_path_str)
        text = p.read_text(encoding="utf-8", errors="ignore")
        raw_lines = txt_to_lines(text)
        part_num = _infer_part_num(p, raw_lines)
        lines = crop_relevant_text(raw_lines)
        source = f"local:{p}"
        articles = parse_articles(lines, part_num=part_num, source=source)
        print(f"[INFO] Parsed articles from part {part_num}: {len(articles)} ({p.name})")
        all_articles.extend(articles)

    all_articles.sort(key=lambda a: (a.part, float(a.article_number.replace(".", "")) if a.article_number.replace(".", "").isdigit() else 0))

    txt_path = write_txt(root, all_articles)
    json_path = write_json(root, all_articles)
    written = write_articles(root, all_articles)

    print("[DONE]")
    print(f"Articles total: {len(all_articles)}")
    print(f"Per-article files: {written} -> {root / 'articles'}")
    print(f"TXT:  {txt_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()

