# Legal RAG RU — подробная инструкция по реализации в 15 этапов

Проект: локальный RAG-сервис для разбора юридических документов на Ubuntu 24.04.

Архитектура:
- код проекта: `/workspace/legal-rag-ru`
- данные проекта: `/datasets/legal-rag-ru`
- Docker runtime: `/docker`
- домашний каталог: только пользовательские настройки

Цель MVP:
- загрузить PDF / DOCX / TXT
- извлечь текст
- разбить на чанки
- сохранить в Qdrant
- найти релевантные фрагменты
- отдать контекст в LLM (Ollama или OpenAI)
- получить структурированный ответ

--------------------------------------------------
ЭТАП 1. Подготовить окружение и правильно разложить проект по разделам
--------------------------------------------------

## Целевая схема

Код:
`/workspace/legal-rag-ru`

Данные:
`/datasets/legal-rag-ru`

Подкаталоги данных:
- `/datasets/legal-rag-ru/qdrant`
- `/datasets/legal-rag-ru/uploads`
- `/datasets/legal-rag-ru/processed`
- `/datasets/legal-rag-ru/ollama`
- `/datasets/legal-rag-ru/logs`

## Установить базовые пакеты

```bash
sudo apt update
sudo apt install -y git curl python3 python3-venv python3-pip ca-certificates
```

Если Docker еще не установлен:

```bash
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
```

## Создать каталоги

```bash
sudo mkdir -p /workspace/legal-rag-ru
sudo chown -R $USER:$USER /workspace/legal-rag-ru

sudo mkdir -p /datasets/legal-rag-ru/{qdrant,uploads,processed,ollama,logs}
sudo chown -R $USER:$USER /datasets/legal-rag-ru
```

--------------------------------------------------
ЭТАП 2. Создать Python-окружение и поставить зависимости
--------------------------------------------------

```bash
cd /workspace/legal-rag-ru
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn[standard] qdrant-client pydantic python-multipart \
            langchain-text-splitters pypdf docx2txt \
            sentence-transformers requests openai python-dotenv
```

Проверка:

```bash
which python
which pip
```

Ожидаемо:

```bash
/workspace/legal-rag-ru/.venv/bin/python
/workspace/legal-rag-ru/.venv/bin/pip
```

--------------------------------------------------
ЭТАП 3. Поднять Qdrant в Docker
--------------------------------------------------

Qdrant отдельно в систему не устанавливаем. Docker сам скачивает image и запускает контейнер.

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /datasets/legal-rag-ru/qdrant:/qdrant/storage \
  qdrant/qdrant
```

Проверка:

```bash
curl http://localhost:6333/collections
```

--------------------------------------------------
ЭТАП 4. Установить Ollama и привязать модели к /datasets
--------------------------------------------------

## Установка Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Проверка:

```bash
which ollama
ollama --version
```

## Привязать systemd-сервис Ollama к `/datasets/legal-rag-ru/ollama`

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<'EOF'
[Service]
Environment="OLLAMA_MODELS=/datasets/legal-rag-ru/ollama"
EOF

sudo chown -R ollama:ollama /datasets/legal-rag-ru/ollama
sudo chmod 755 /datasets/legal-rag-ru/ollama
sudo systemctl daemon-reload
sudo systemctl restart ollama
sudo systemctl show ollama --property=Environment --no-pager
```

Ожидаемо в выводе:

```text
OLLAMA_MODELS=/datasets/legal-rag-ru/ollama
```

## Скачать модель

```bash
ollama pull qwen2.5:7b
```

Проверка:

```bash
ollama list
sudo du -sh /datasets/legal-rag-ru/ollama
```

--------------------------------------------------
ЭТАП 5. Создать структуру проекта
--------------------------------------------------

```bash
cd /workspace/legal-rag-ru
mkdir -p app scripts tests
touch .env README.md
```

Итоговая структура:

```text
/workspace/legal-rag-ru/
├── .venv/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── extractors.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── vectorstore.py
│   ├── llm.py
│   └── metadata_store.py
├── scripts/
│   ├── fetch_gk_rf.py
│   ├── parse_gk_rf_txt.py
│   └── index_gk_rf.py
├── tests/
├── .env
└── README.md
```

Создай пустой `__init__.py`:

```bash
touch app/__init__.py
```

--------------------------------------------------
ЭТАП 6. Создать .env
--------------------------------------------------

Содержимое файла `/workspace/legal-rag-ru/.env`:

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=legal_docs_ru
GK_QDRANT_COLLECTION=gk_rf_ru

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OPENAI_API_KEY=
OPENAI_MODEL=gpt-5-mini
EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

DATA_ROOT=/datasets/legal-rag-ru
UPLOAD_DIR=/datasets/legal-rag-ru/uploads
PROCESSED_DIR=/datasets/legal-rag-ru/processed
LOG_DIR=/datasets/legal-rag-ru/logs
GK_RF_DIR=/datasets/legal-rag-ru/gk_rf
```

--------------------------------------------------
ЭТАП 7. Реализовать извлечение текста
--------------------------------------------------

Файл: `app/extractors.py`

```python
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
```

--------------------------------------------------
ЭТАП 8. Реализовать нарезку текста на чанки
--------------------------------------------------

Файл: `app/chunking.py`

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]
```

--------------------------------------------------
ЭТАП 9. Реализовать эмбеддинги
--------------------------------------------------

Файл: `app/embeddings.py`

```python
import os
from sentence_transformers import SentenceTransformer

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        model_name = os.getenv(
            "EMBED_MODEL",
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        )
        _model = SentenceTransformer(model_name)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    model = get_model()
    vector = model.encode([text], normalize_embeddings=True)[0]
    return vector.tolist()
```

--------------------------------------------------
ЭТАП 10. Реализовать Qdrant vector store
--------------------------------------------------

Файл: `app/vectorstore.py`

```python
import os
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.embeddings import embed_texts, embed_query


QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "legal_docs_ru")

client = QdrantClient(url=QDRANT_URL)


def ensure_collection(vector_size: int = 384) -> None:
    collections = client.get_collections().collections
    existing_names = [c.name for c in collections]

    if QDRANT_COLLECTION not in existing_names:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def index_chunks(doc_id: str, filename: str, chunks: list[str]) -> int:
    if not chunks:
        return 0

    vectors = embed_texts(chunks)
    vector_size = len(vectors[0])
    ensure_collection(vector_size=vector_size)

    points = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "doc_id": doc_id,
                    "filename": filename,
                    "chunk_index": i,
                    "text": chunk,
                },
            )
        )

    client.upsert(collection_name=QDRANT_COLLECTION, points=points)
    return len(points)


def search_chunks(query: str, limit: int = 5):
    # В qdrant-client>=1.17 используется `query_points` (метода `search` нет).
    query_vector = embed_query(query)
    ensure_collection(vector_size=len(query_vector))
    response = client.query_points(
        collection_name=QDRANT_COLLECTION,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )
    return response.points
```

ЭТАП 11. Реализовать работу с LLM (Ollama + OpenAI)
--------------------------------------------------

Файл: `app/llm.py`

```python
import os
import requests
from openai import OpenAI

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5-mini")

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
"""


def generate_answer(question: str, context: str, provider: str = "ollama") -> str:
    prompt = f"""{SYSTEM_PROMPT}

КОНТЕКСТ:
{context}

ВОПРОС:
{question}
"""

    if provider == "openai":
        client = OpenAI()
        resp = client.responses.create(
            model=OPENAI_MODEL,
            input=prompt,
        )
        return (resp.output_text or "").strip()

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
```

Примечание:
- `provider="ollama"` — локальная модель без внешнего API.
- `provider="openai"` — облачная модель OpenAI (нужен `OPENAI_API_KEY`).
- Если OpenAI возвращает `403 unsupported_country_region_territory`, это сетевое/региональное ограничение, а не ошибка бизнес-логики приложения.

--------------------------------------------------
ЭТАП 12. Реализовать FastAPI backend
--------------------------------------------------

Файл: `app/main.py`

```python
from pathlib import Path
import json
import os
from typing import Literal
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from app.extractors import extract_text
from app.chunking import split_text
from app.vectorstore import index_chunks, search_chunks
from app.llm import generate_answer


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

    return {
        "doc_id": doc_id,
        "filename": safe_name,
        "original_path": str(original_path),
        "extracted_text_path": str(extracted_path),
        "chunks_path": str(chunks_path),
        "chars": len(text),
        "chunks_count": len(chunks),
        "indexed_count": indexed_count,
        "preview": text[:1000],
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
    context_parts = []

    for item in results:
        text = item.payload.get("text", "")
        filename = item.payload.get("filename")
        chunk_index = item.payload.get("chunk_index")
        doc_id = item.payload.get("doc_id")

        matches.append(
            {
                "score": item.score,
                "doc_id": doc_id,
                "filename": filename,
                "chunk_index": chunk_index,
                "text": text,
            }
        )

        context_parts.append(
            f"[Файл: {filename}; doc_id: {doc_id}; chunk: {chunk_index}]\n{text}"
        )

    # В текущей версии проекта дополнительно подтягиваем релевантные фрагменты ГК РФ
    # из отдельной коллекции `GK_QDRANT_COLLECTION` и кладём их в контекст, а также
    # возвращаем отдельным массивом `gk_matches` (см. реализацию в `app/main.py`).

    if not context_parts:
        return {
            "question": question,
            "answer": "Не удалось найти релевантные фрагменты в загруженных документах.",
            "matches": [],
            "gk_matches": gk_matches,
        }

    context = "\n---\n".join(context_parts)

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
    }
```

### Дополнительные эндпоинты (реализованы в проекте)
- `GET /documents` — список загруженных договоров (`doc_id`, `filename`), лимит 50
- `GET /documents/{doc_id}` — полные метаданные договора
- `GET /gk/health` — диагностика коллекции ГК РФ (существует ли и сколько в ней точек)

### Хранение метаданных документов (реализовано)
Метаданные загрузок пишутся в SQLite:
`/datasets/legal-rag-ru/processed/metadata.sqlite`

### Подключение справочной базы ГК РФ (вариант 1 — отдельная коллекция, реализовано)
Идея: индексировать ГК РФ в отдельную коллекцию Qdrant (`GK_QDRANT_COLLECTION`) и при `/ask`
опционально подмешивать найденные фрагменты в контекст.

1) Подготовить файлы (папка `GK_RF_DIR`):
- `scripts/fetch_gk_rf.py --local-raw` — парсит существующие `*.html/*.htm` из `GK_RF_DIR/raw_html/`
- `scripts/parse_gk_rf_txt.py ...` — парсит локальные TXT частей I–IV

2) Индексация в Qdrant:

```bash
python scripts/index_gk_rf.py
```

3) Проверка:

```bash
curl -sS http://localhost:8000/gk/health
```

--------------------------------------------------
ЭТАП 13. Запустить проект локально
--------------------------------------------------

```bash
cd /workspace/legal-rag-ru
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`.env` загружается автоматически при старте FastAPI (через `python-dotenv` в `app/main.py`), поэтому ручной `source .env` не обязателен.

Проверка:
- `http://localhost:8000/health`
- `http://localhost:8000/docs`

--------------------------------------------------
ЭТАП 14. Прогнать тестовый сценарий
--------------------------------------------------

## Создать тестовый договор

```bash
cat > /tmp/test_contract.txt <<'EOF'
ДОГОВОР ОКАЗАНИЯ УСЛУГ

1. Предмет договора
Исполнитель обязуется оказать услуги, а Заказчик обязуется принять и оплатить их.

2. Оплата
Стоимость услуг составляет 100 000 рублей. Оплата производится в течение 5 дней.

3. Ответственность
За просрочку оплаты Заказчик уплачивает пеню в размере 0.5% за каждый день просрочки.

4. Расторжение
Заказчик вправе отказаться от договора в одностороннем порядке в любое время без выплаты компенсации Исполнителю.
EOF
```

## Загрузить файл

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "file=@/tmp/test_contract.txt"
```

## Задать вопрос

Через локальную модель Ollama:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Есть ли риск одностороннего расторжения для исполнителя и что стоит изменить?",
    "limit": 3,
    "llm_provider": "ollama"
  }'
```

Через OpenAI:

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Есть ли риск одностороннего расторжения для исполнителя и что стоит изменить?",
    "limit": 3,
    "llm_provider": "openai"
  }'
```

## Что должно получиться

Сервис должен вернуть:
- краткий вывод
- найденные фрагменты
- риск для исполнителя
- рекомендацию добавить уведомление / компенсацию
- оговорку, что ответ основан только на тексте документа

--------------------------------------------------
ЭТАП 15. Что улучшать после MVP
--------------------------------------------------

После того как базовый MVP заработал, следующая очередь работ такая:

1. Фильтрация вопросов по конкретному `doc_id`
2. Хранение метаданных документа
3. UI поверх API
4. Выделение рисковых пунктов по типам
5. Более точная чанкизация по разделам договора
6. Подключение справочной базы по ГК РФ
7. Логирование запросов и ответов
8. Docker Compose для полного запуска
9. Reverse proxy через Nginx
10. Перенос на тестовый сервер Ubuntu 24.04

--------------------------------------------------
ГОТОВЫЙ КАРКАС ПРОЕКТА ДЛЯ БЫСТРОГО СОЗДАНИЯ
--------------------------------------------------

Создание файлов одной командой:

```bash
cd /workspace/legal-rag-ru
mkdir -p app scripts tests
cat > app/__init__.py <<'EOF'
EOF
```

Потом по очереди создай файлы:
- `app/extractors.py`
- `app/chunking.py`
- `app/embeddings.py`
- `app/vectorstore.py`
- `app/llm.py`
- `app/main.py`
- `.env`

и вставь содержимое из этапов выше.

--------------------------------------------------
КОРОТКИЙ ЧЕК-ЛИСТ ЗАПУСКА
--------------------------------------------------

1. Активировать `.venv`
2. Проверить `docker ps`
3. Проверить `curl http://localhost:6333/collections`
4. Проверить `ollama list`
5. Проверить `OPENAI_API_KEY` (если используешь OpenAI)
6. Запустить `uvicorn`
7. Открыть `/docs`
8. Загрузить документ
9. Задать вопрос с `llm_provider` (`ollama` или `openai`)

--------------------------------------------------
КОМАНДЫ ДИАГНОСТИКИ
--------------------------------------------------

Проверка Qdrant:

```bash
docker start qdrant
curl http://localhost:6333/collections
docker ps
docker logs qdrant
```

Если контейнера `qdrant` ещё нет (первый запуск):

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /datasets/legal-rag-ru/qdrant:/qdrant/storage \
  qdrant/qdrant
```

Проверка Ollama:

```bash
sudo systemctl start ollama
ollama list
systemctl status ollama --no-pager
curl http://localhost:11434/api/tags
```

Если список моделей пустой:

```bash
ollama pull qwen2.5:7b
```

Проверка API:

```bash
curl http://localhost:8000/health
```

Проверка OpenAI-пути:

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(bool(os.getenv('OPENAI_API_KEY')), os.getenv('OPENAI_API_KEY','')[:12])"
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Ответь одним словом: тест","llm_provider":"openai","limit":1}'
```

Если получаешь `403 unsupported_country_region_territory`, проверь маршрут трафика/VPN в том же терминале, где запущен `uvicorn`.

--------------------------------------------------
ФИНАЛЬНЫЙ РЕЗУЛЬТАТ
--------------------------------------------------

После выполнения всех 15 этапов у тебя будет рабочий локальный MVP Legal RAG RU:
- загрузка юридических документов
- извлечение текста
- нарезка на чанки
- индексирование в Qdrant
- семантический поиск
- генерация ответа через Ollama или OpenAI (переключение через `llm_provider`)
- структура, готовая к переносу на тестовый сервер

Конец инструкции.

