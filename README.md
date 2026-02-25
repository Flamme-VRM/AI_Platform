# AsylBILIM — AI Platform

> **Асыл Білім** (Kazakh: *Precious Knowledge*) — an AI-powered backend for UNT exam preparation, delivering intelligent tutoring in Kazakh via a REST API and Telegram bot.

Built on **Google Gemini**, **RAG** (Retrieval-Augmented Generation), and **Kazakh Whisper STT**, deployed over **FastAPI** with **Redis** for persistence.

---

## What It Does

Kazakhstani high-school students use AsylBILIM to prepare for the **UNT** (Unified National Testing). The platform:

- Answers questions on **Math, Informatics, and History** in the Kazakh language
- Retrieves relevant context from a hand-crafted **knowledge base** of 60+ UNT study documents before generating answers
- Accepts **voice messages** and transcribes them using a Kazakh-tuned Whisper model
- Manages **multiple named conversations** per user, with full history persistence
- Enforces a **rate limit** of 15 messages / 24 hours via Redis to ensure fair usage

---

## Architecture

```
Client (Flutter App / Telegram)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│                FastAPI  (api_server.py)              │
│                                                      │
│  /api/chat ──► AIService ──► Gemini LLM              │
│                    └──► RAG Pipeline ──► Vector DB   │
│  /api/voice ──► SpeechToTextService (Whisper)        │
│  /api/chats/* ──► CacheService ──► Redis             │
└──────────────────────────────────────────────────────┘
        │                    │
      Redis              SQLite
  (chat history,        (vector store
   metadata,            documents.db)
   rate limits)
```

---

## Project Structure

```
AI_Platform/
├── src/
│   ├── api_server.py              # FastAPI app — all REST endpoints (v2.1.0)
│   ├── bot.py                     # Telegram bot (aiogram 3)
│   ├── config.py                  # Pydantic Settings — loaded from .env
│   ├── handlers/
│   │   └── message_handler.py    # Telegram message routing
│   └── services/
│       ├── ai.py                  # AI generation + RAG pipeline integration
│       ├── cache.py               # Redis client — history, metadata, rate limits
│       ├── speech_to_text.py     # Whisper Kazakh ASR service
│       ├── embeddings.py          # Sentence-Transformers embedding model
│       ├── vector_db.py           # SQLite-backed local vector store
│       ├── improved_rag_service.py # Semantic retrieval (top-K search)
│       ├── chunker.py             # Text chunking for RAG indexing
│       ├── document_loader.py    # Load JSON docs from RAG/
│       ├── indexer.py             # Index pipeline: load → chunk → embed → store
│       └── analytics.py           # Usage tracking
├── RAG/                           # 60+ JSON knowledge-base files
│   ├── math_*.json               # Algebra, geometry, calculus, trigonometry
│   ├── informatics_*.json        # Python, SQL, networking, algorithms
│   └── history_*.json            # Kazakhstan history
├── scripts/
│   └── index_documents.py        # CLI: indexes RAG/ into documents.db
├── tests/
│   ├── test_rag_system.py        # RAG integration tests
│   ├── verify_rag.py             # Retrieval quality checks
│   └── tst.py                    # Misc scripts
├── documents.db                   # SQLite vector database (pre-built)
├── main.py                        # Telegram bot runner
└── requirements.txt
```

---

## REST API

Interactive docs: `http://localhost:8000/docs`

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Service info, version, available routes |
| `GET` | `/health` | Health check — Redis ping + model name |
| `POST` | `/api/chats/new` | Create a new chat conversation |
| `GET` | `/api/chats/{user_id}` | List all chats for a user |
| `DELETE` | `/api/chats/{chat_id}?user_id=` | Delete a chat and all its history |
| `PATCH` | `/api/chats/{chat_id}` | Rename a chat |
| `GET` | `/api/chats/{user_id}/{chat_id}/history` | Get structured message history |
| `POST` | `/api/chat` | Send a text message, receive AI response |
| `POST` | `/api/voice` | Upload audio, receive transcription + AI response |
| `GET` | `/api/status/{user_id}` | Rate limit info: count / remaining / reset |

### Example: Text Chat

```http
POST /api/chat
Content-Type: application/json

{
  "user_id": 123456,
  "chat_id": "uuid-here",
  "message": "Квадрат теңдеуді қалай шешемін?",
  "language": "kk"
}
```

```json
{
  "response": "Квадрат теңдеуді шешу үшін...",
  "message_count": 3,
  "remaining_messages": 12
}
```

---

## How the AI Pipeline Works

```
User message
    │
    ├─ 1. Rate limit check (Redis) ──► 429 if exceeded
    │
    ├─ 2. Load chat history from Redis
    │
    ├─ 3. Embed message with Sentence-Transformers
    │
    ├─ 4. Semantic search → top-K passages from documents.db
    │
    ├─ 5. Build prompt:
    │       [System Prompt] + [Retrieved Context] + [Chat History] + [Message]
    │
    ├─ 6. Generate response via Google Gemini (streaming disabled)
    │
    ├─ 7. Save updated history to Redis (last 50 msgs, TTL: 7 days)
    │
    └─ 8. Update chat metadata (last_message, count, TTL: 30 days)
```

---

## Data Storage (Redis)

All chat state is stored in Redis using msgpack serialization:

| Key | Type | TTL | Purpose |
|-----|------|-----|---------|
| `user_chats:{user_id}` | List | 30 days | Ordered list of chat UUIDs |
| `chat_metadata:{user_id}:{chat_id}` | String | 30 days | Title, dates, last message, count |
| `chat_history:{user_id}:{chat_id}` | String | 7 days | `["User: ...", "AsylBILIM: ..."]` |
| `rate_limit:{user_id}` | String | 24 hours | Message count for current window |

---

## Voice Input (Whisper STT)

The `/api/voice` endpoint:

1. Receives an uploaded audio file (OGG from mobile, or any format)
2. Converts to WAV, resamples to **16kHz mono** if needed (via `pydub`)
3. Runs **Hugging Face Whisper** with `language="kazakh"` and `task="transcribe"`
4. Passes the transcribed text into the standard AI pipeline
5. Returns `recognized_text` + `response` in one call

> CUDA is used automatically when available. Falls back to CPU gracefully.

---

## Knowledge Base (RAG)

The `RAG/` directory contains hand-crafted JSON documents grouped by subject:

| Subject | File Count | Topics |
|---------|-----------|--------|
| **Math** | ~35 | Algebra, geometry, calculus, combinatorics, probability, sequences, trigonometry |
| **Informatics** | ~20 | Python, SQL, HTML/CSS, networking, algorithms, sorting, logic |
| **History** | ~5 | Kazakhstan history, context documents |

Documents are indexed into `documents.db` (SQLite + cosine similarity search). To reindex after adding new files:

```bash
python scripts/index_documents.py
```

---

## Setup

### Prerequisites

- Python 3.8+
- Redis (cloud or local — [Redis Cloud free tier](https://redis.com/try-free/) works)
- NVIDIA GPU recommended for STT (CPU fallback available)
- Google Gemini API key
- Telegram Bot Token (for the bot interface)

### 1. Clone

```bash
git clone https://github.com/Flamme-VRM/AI_Platform.git
cd AI_Platform
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> PyTorch is pinned to CUDA 13.0. Adjust `--index-url` in `requirements.txt` for your CUDA version or use `pip install torch` for CPU-only.

### 3. Configure

Create a `.env` file in the project root (**never committed**):

```env
BOT_TOKEN=your_telegram_bot_token
LLM_API_KEY=your_google_gemini_api_key
MODEL=gemini-3-flash-preview

REDIS_HOST=your_redis_host
REDIS_PORT=15792
REDIS_USERNAME=default
REDIS_PASSWORD=your_redis_password

RATE_LIMIT=15
RATE_WINDOW_HOURS=24

VECTOR_DB_PATH=documents.db
SYSPROMPT_PATH=SYSPROMPT.txt
```

### 4. Index Knowledge Base

```bash
python scripts/index_documents.py
```

### 5. Start REST API

```bash
python -m uvicorn src.api_server:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start Telegram Bot (optional)

```bash
python main.py
```

---

## Rate Limiting

| Setting | Value |
|---------|-------|
| Messages per window | **15** |
| Window duration | **24 hours** |
| Reset mechanism | Rolling from first message |
| Enforcement | Redis key with 24h TTL |

Users receive remaining count in every response. A `429` HTTP status is returned when the limit is exceeded, with `retry_after` in seconds.

---

## Mobile Frontend

This backend powers the **AsylBILIM Flutter mobile app**:
→ [AI_Platform_UI](https://github.com/Flamme-VRM/AI_Platform_UI) | [FullStack_Platform](https://github.com/Flamme-VRM/FullStack_Platform)

---

## Contact

- Telegram: [@Vermeei](https://t.me/Vermeei)
- Created by **Shyngisbek Asylkhan**

---

*Empowering Kazakhstani students to excel in UNT through AI-powered education* 🇰🇿
