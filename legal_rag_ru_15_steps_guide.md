# Legal RAG RU — инструкция по реализации в 15 этапов

Проект: локальный RAG-сервис для разбора юридических документов на Ubuntu 24.04.

**Архитектура (фактическая):**
- код: `/workspace/legal-rag-ru`
- данные: `/datasets/legal-rag-ru` (или пути из `.env`)
- Docker: Qdrant по желанию
- Ollama: локальные LLM

**Цель MVP:** загрузка PDF/DOCX/TXT → извлечение текста → чанки → Qdrant → семантический поиск → контекст в LLM → структурированный ответ с опорой на источники; опционально подмешиваются фрагменты ГК РФ из отдельной коллекции.

**Где смотреть код:** модули в `app/` (`main.py`, `extractors.py`, `chunking.py`, `embeddings.py`, `vectorstore.py`, `llm.py`, `metadata_store.py`), скрипты в `scripts/`. Ниже — описание этапов, **команды терминала** и проверки; без дублирования полных исходников приложения.

---

## Этап 1. Окружение и раскладка каталогов

Установить базовые пакеты и создать каталоги данных.

```bash
sudo apt update
sudo apt install -y git curl python3 python3-venv python3-pip ca-certificates
```

Docker для Qdrant (если ещё не установлен):

```bash
sudo apt install -y docker.io docker-compose-v2
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
```

Каталоги проекта и данных (пути при необходимости замените):

```bash
sudo mkdir -p /workspace/legal-rag-ru
sudo chown -R $USER:$USER /workspace/legal-rag-ru

sudo mkdir -p /datasets/legal-rag-ru/{qdrant,uploads,processed,ollama,logs,gk_rf}
sudo chown -R $USER:$USER /datasets/legal-rag-ru
```

**Проверка:** каталоги существуют, владелец совпадает с пользователем запуска.

---

## Этап 2. Python-окружение и зависимости

```bash
cd /workspace/legal-rag-ru
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn[standard] qdrant-client pydantic python-multipart \
            langchain-text-splitters pypdf docx2txt \
            sentence-transformers requests openai python-dotenv httpx
```

**Проверка:**

```bash
which python
which pip
```

Ожидаемо: `/workspace/legal-rag-ru/.venv/bin/python` и `.../pip`.

---

## Этап 3. Qdrant

Запуск в Docker (том на диск данных, порт 6333):

```bash
docker run -d \
  --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /datasets/legal-rag-ru/qdrant:/qdrant/storage \
  qdrant/qdrant
```

Имя коллекции для договоров — `QDRANT_COLLECTION`; для ГК РФ — `GK_QDRANT_COLLECTION` (в `.env`).

**Проверка:**

```bash
curl -sS http://localhost:6333/collections
docker ps
```

---

## Этап 4. Ollama (локальные модели)

Установка:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Опционально: привязать каталог моделей к `/datasets` (через override systemd, переменная `OLLAMA_MODELS`), затем:

```bash
ollama pull qwen2.5:7b
# или другая модель, например:
# ollama pull llama3.1:8b
```

Активная модель для приложения задаётся в `.env` как `OLLAMA_MODEL` (см. `app/llm.py`).

**Проверка:**

```bash
which ollama
ollama --version
ollama list
curl -sS http://localhost:11434/api/tags
```

---

## Этап 5. Структура проекта

```bash
cd /workspace/legal-rag-ru
mkdir -p app scripts tests
touch app/__init__.py
```

Ожидаемая логика: `app/` — API и логика RAG; `scripts/` — ГК РФ; `tests/` — тесты.

**Проверка:** после клонирования/копирования кода запуск uvicorn (этап 13) не падает на импортах.

---

## Этап 6. Файл `.env`

Переменные подхватываются при старте (`load_dotenv()` в `app/main.py`). Создайте `/workspace/legal-rag-ru/.env` и задайте переменные; **секреты не коммитить**.

Пример структуры (значения подставьте свои):

```env
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=legal_docs_ru
GK_QDRANT_COLLECTION=gk_rf_ru

OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b
OPENAI_MODEL=gpt-5-mini
OPENAI_API_KEY=

EMBED_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2

DATA_ROOT=/datasets/legal-rag-ru
UPLOAD_DIR=/datasets/legal-rag-ru/uploads
PROCESSED_DIR=/datasets/legal-rag-ru/processed
LOG_DIR=/datasets/legal-rag-ru/logs
GK_RF_DIR=/datasets/legal-rag-ru/gk_rf

# GigaChat (если llm_provider=gigachat)
GIGACHAT_AUTHORIZATION_KEY=
GIGACHAT_SCOPE=GIGACHAT_API_PERS
GIGACHAT_BASE_URL=https://gigachat.devices.sberbank.ru/api/v1
GIGACHAT_OAUTH_URL=https://ngw.devices.sberbank.ru:9443/api/v2/oauth
GIGACHAT_MODEL=GigaChat
GIGACHAT_VERIFY_SSL_CERTS=true
GIGACHAT_CA_BUNDLE_FILE=
```

Дополнительно для Ollama в `app/llm.py` могут использоваться `OLLAMA_TEMPERATURE`, `OLLAMA_TOP_P`, `OLLAMA_SEED`.

**Проверка:** после правки `.env` перезапустить uvicorn; при смене модели — `ollama list` содержит `OLLAMA_MODEL`.

---

## Этап 7. Извлечение текста

Реализовано в `app/extractors.py` (PDF, DOCX, TXT).

**Проверка:** через `POST /upload` после запуска API (этап 13–14).

---

## Этап 8. Чанкинг

Реализовано в `app/chunking.py`.

**Проверка:** косвенно — успешная индексация после upload и ответ `/ask`.

---

## Этап 9. Эмбеддинги

Реализовано в `app/embeddings.py`, модель из `EMBED_MODEL`.

**Проверка:** первый успешный `POST /upload` подтянет модель эмбеддингов (может занять время).

---

## Этап 10. Qdrant: индексация и поиск

Реализовано в `app/vectorstore.py`: коллекция договоров, поиск, отдельно ГК РФ и `gk_health`.

**Проверка:** см. этапы 13–15 и curl к `/ask`, `/gk/health`.

---

## Этап 11. LLM: провайдеры и промпт

Реализовано в `app/llm.py`: `ollama`, `openai`, `gigachat`; ссылки на источники `[M#]` / `[G#]`.

Проверка OpenAI-ключа локально (опционально):

```bash
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(bool(os.getenv('OPENAI_API_KEY')))"
```

Если OpenAI возвращает `403 unsupported_country_region_territory`, проверьте VPN/маршрут в том же окружении, где запущен uvicorn.

---

## Этап 12. FastAPI: эндпоинты

Реализовано в `app/main.py`: `/health`, `/upload`, `/ask`, `/documents`, `/documents/{doc_id}`, `/gk/health`; провайдер LLM — поле `llm_provider` в теле `POST /ask`.

---

## Этап 13. Запуск локально

```bash
cd /workspace/legal-rag-ru
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

`.env` подхватывается при старте приложения, отдельный `source .env` не обязателен.

**Проверка:**

```bash
curl -sS http://localhost:8000/health
```

Откройте в браузере: `http://localhost:8000/docs`.

---

## Этап 14. Тестовый сценарий

Создать тестовый договор:

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

Загрузка:

```bash
curl -sS -X POST "http://localhost:8000/upload" \
  -F "file=@/tmp/test_contract.txt"
```

Вопрос (Ollama):

```bash
curl -sS -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Есть ли риск одностороннего расторжения для исполнителя и что стоит изменить?",
    "limit": 3,
    "llm_provider": "ollama"
  }'
```

OpenAI:

```bash
curl -sS -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Есть ли риск одностороннего расторжения для исполнителя и что стоит изменить?",
    "limit": 3,
    "llm_provider": "openai"
  }'
```

---

## Этап 15. ГК РФ как отдельная коллекция

Подготовка данных в `GK_RF_DIR` (см. `scripts/fetch_gk_rf.py`, `scripts/parse_gk_rf_txt.py` при необходимости).

Индексация в Qdrant:

```bash
cd /workspace/legal-rag-ru
source .venv/bin/activate
python scripts/index_gk_rf.py
```

**Проверка:**

```bash
curl -sS http://localhost:8000/gk/health
```

В `POST /ask` используйте `include_gk_rf` и при необходимости `gk_limit`; для GigaChat — `"llm_provider": "gigachat"`.

---

## После MVP: типичные улучшения

1. Фильтрация вопросов по `doc_id`.
2. Расширение метаданных и аудит загрузок.
3. Веб-UI поверх API.
4. Типизация рисков по пунктам договора.
5. Уточнение чанкизации по структуре договора.
6. Расширение справочных баз (не только ГК РФ).
7. Логирование запросов и ответов.
8. Docker Compose для полного запуска.
9. Reverse proxy через Nginx.
10. Перенос на тестовый сервер Ubuntu 24.04.

---

## Диагностика (команды)

Qdrant:

```bash
docker start qdrant
curl -sS http://localhost:6333/collections
docker ps
docker logs qdrant
```

Ollama:

```bash
sudo systemctl start ollama
ollama list
systemctl status ollama --no-pager
curl -sS http://localhost:11434/api/tags
```

API:

```bash
curl -sS http://localhost:8000/health
```

OpenAI-путь:

```bash
curl -sS -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Ответь одним словом: тест","llm_provider":"openai","limit":1}'
```

---

Конец инструкции.
