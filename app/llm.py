import os
import re

import requests

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

SYSTEM_PROMPT = """
Ты — юридический AI-помощник по российским договорам.

Правила:
1. Отвечай только на основе переданного контекста.
2. Не выдумывай факты, которых нет в контексте.
3. Если данных недостаточно, прямо скажи об этом.
4. Язык ответа должен совпадать с языком вопроса.
5. Ответ делай структурированным.

Формат ответа:
1. Краткий вывод
2. Что найдено в документе
3. Риски для стороны
4. Что стоит изменить
5. Ограничение ответа
""".strip()


def _detect_language_hint(question: str) -> str:
    # Simple heuristic: Cyrillic -> Russian, otherwise English.
    return "ru" if _CYRILLIC_RE.search(question or "") else "en"


def generate_answer(question: str, context: str) -> str:
    lang_hint = _detect_language_hint(question)
    prompt = f"""{SYSTEM_PROMPT}

LANGUAGE_HINT: {lang_hint}

КОНТЕКСТ:
{context}

ВОПРОС:
{question}
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"].strip()

