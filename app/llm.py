import os
import re
import time
import uuid
import threading

import requests
import httpx
from openai import OpenAI

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
OLLAMA_TOP_P = float(os.getenv("OLLAMA_TOP_P", "1"))
_OLLAMA_SEED_RAW = os.getenv("OLLAMA_SEED", "").strip()
OLLAMA_SEED = int(_OLLAMA_SEED_RAW) if _OLLAMA_SEED_RAW else None
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

GIGACHAT_AUTHORIZATION_KEY = os.getenv("GIGACHAT_AUTHORIZATION_KEY", "")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_BASE_URL = os.getenv("GIGACHAT_BASE_URL", "https://gigachat.devices.sberbank.ru/api/v1")
GIGACHAT_OAUTH_URL = os.getenv(
    "GIGACHAT_OAUTH_URL",
    "https://ngw.devices.sberbank.ru:9443/api/v2/oauth",
)
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat")
GIGACHAT_VERIFY_SSL_CERTS = (os.getenv("GIGACHAT_VERIFY_SSL_CERTS", "true") or "").strip().lower() not in {
    "0",
    "false",
    "no",
}
GIGACHAT_CA_BUNDLE_FILE = (os.getenv("GIGACHAT_CA_BUNDLE_FILE", "") or "").strip()


def _gigachat_verify_param() -> bool | str:
    if GIGACHAT_CA_BUNDLE_FILE:
        return GIGACHAT_CA_BUNDLE_FILE
    return GIGACHAT_VERIFY_SSL_CERTS

_GIGACHAT_ACCESS_TOKEN: str | None = None
_GIGACHAT_EXPIRES_AT: int = 0
_GIGACHAT_LOCK = threading.Lock()

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


def _get_gigachat_access_token() -> str:
    """
    GigaChat OAuth token is short-lived (30 minutes).
    Cache it in-memory to avoid requesting it on every /ask call.
    """
    auth_key = (GIGACHAT_AUTHORIZATION_KEY or "").strip()
    if not auth_key:
        raise RuntimeError("GIGACHAT_AUTHORIZATION_KEY is not set")

    global _GIGACHAT_ACCESS_TOKEN, _GIGACHAT_EXPIRES_AT

    now = int(time.time())
    # Reuse token while it's still valid for at least 60s.
    if _GIGACHAT_ACCESS_TOKEN is not None and (_GIGACHAT_EXPIRES_AT - now) > 60:
        return _GIGACHAT_ACCESS_TOKEN

    with _GIGACHAT_LOCK:
        now = int(time.time())
        if _GIGACHAT_ACCESS_TOKEN is not None and (_GIGACHAT_EXPIRES_AT - now) > 60:
            return _GIGACHAT_ACCESS_TOKEN

        rq_uid = str(uuid.uuid4())
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
            "RqUID": rq_uid,
            "Authorization": f"Basic {auth_key}",
        }
        resp = requests.post(
            GIGACHAT_OAUTH_URL,
            headers=headers,
            data={"scope": GIGACHAT_SCOPE},
            timeout=30,
            verify=_gigachat_verify_param(),
        )
        resp.raise_for_status()
        data = resp.json() or {}
        access_token = (data.get("access_token") or "").strip()
        expires_at = int(data.get("expires_at") or 0)

        if not access_token or expires_at <= 0:
            # Fallback: assume 30 minutes from now if expires_at missing.
            expires_at = now + 30 * 60

        _GIGACHAT_ACCESS_TOKEN = access_token
        _GIGACHAT_EXPIRES_AT = expires_at
        return _GIGACHAT_ACCESS_TOKEN


def _generate_with_gigachat(prompt: str) -> str:
    access_token = _get_gigachat_access_token()

    # GigaChat provides OpenAI-compatible API via base_url.
    verify_param = _gigachat_verify_param()
    http_client = httpx.Client(verify=verify_param, timeout=60)
    client = OpenAI(api_key=access_token, base_url=GIGACHAT_BASE_URL, http_client=http_client)
    response = client.chat.completions.create(
        model=GIGACHAT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
    )
    text = (
        response.choices[0].message.content
        if response.choices
        else None
    )
    if text:
        return text.strip()
    raise RuntimeError("GigaChat chat.completions returned empty content")


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
    if provider_norm == "gigachat":
        return _generate_with_gigachat(prompt)
    return _generate_with_ollama(prompt)

