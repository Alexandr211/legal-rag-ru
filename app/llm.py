import os
import re

import requests
from openai import OpenAI

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "1"))
_OLLAMA_SEED_RAW = os.getenv("OLLAMA_SEED", "").strip()
OLLAMA_SEED = int(_OLLAMA_SEED_RAW) if _OLLAMA_SEED_RAW else None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

_CYRILLIC_RE = re.compile(r"[А-Яа-яЁё]")

SYSTEM_PROMPT = """
Ты — юридический AI-помощник по российским договорам.

Правила:
1. Отвечай только на основе переданного контекста.
2. Не выдумывай факты, которых нет в контексте.
3. Если данных недостаточно, прямо скажи об этом.
4. Язык ответа должен совпадать с языком вопроса.
5. Ответ делай структурированным.
6. Каждый тезис обязан содержать ссылку на источник из контекста.
7. Формат ссылок: [M1], [M2] для договорных фрагментов и [G1], [G2] для ГК РФ.
8. Если для тезиса нет источника в контексте — явно напиши, что данных недостаточно.

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


def _build_prompt(question: str, context: str) -> str:
    lang_hint = _detect_language_hint(question)
    return f"""{SYSTEM_PROMPT}

LANGUAGE_HINT: {lang_hint}

Требование к обоснованию:
- Каждый отдельный тезис в разделах 1-4 заканчивается ссылкой вида [M#] и/или [G#].
- Не используй ссылки, которых нет в переданном контексте.

КОНТЕКСТ:
{context}

ВОПРОС:
{question}
"""


def _generate_with_ollama(prompt: str) -> str:
    options: dict[str, float | int] = {
        "temperature": OLLAMA_TEMPERATURE,
        "top_p": OLLAMA_TOP_P,
    }
    if OLLAMA_SEED is not None:
        options["seed"] = OLLAMA_SEED

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": options,
        },
        timeout=180,
    )
    response.raise_for_status()
    data = response.json()
    return data["response"].strip()


def _generate_with_openai(prompt: str) -> str:
    client = OpenAI()
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=prompt,
    )
    text = (response.output_text or "").strip()
    if text:
        return text
    raise RuntimeError("OpenAI Responses API returned empty output_text")


def generate_answer(
    question: str,
    context: str,
    *,
    provider: str = "ollama",
) -> str:
    prompt = _build_prompt(question=question, context=context)
    provider_norm = (provider or "ollama").lower().strip()
    if provider_norm == "openai":
        return _generate_with_openai(prompt)
    return _generate_with_ollama(prompt)

