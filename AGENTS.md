# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

> **Note**: This file is kept in sync with the current codebase. The Streamlit frontend has been archived (see [legacy/](legacy/)); FastAPI is the sole active entry point. For the authoritative and more detailed reference, see [CLAUDE.md](CLAUDE.md).

## Project Overview

SAR-Sense: an intelligent platform for SAR (Synthetic Aperture Radar) ship detection. Combines a YOLO-based object detection model with a LangChain ReAct agent that has hybrid RAG (vector + BM25 + parent-child + BGE reranker), tool calling, and conversational memory. Includes a user system (JWT auth + RBAC) backed by MySQL.

## Running the App

**Conda environment:** `rag_env_backup` is required. Activate it before any Python command:
```bash
conda activate rag_env_backup
```
There is **no root `requirements.txt`** in this repository; the only requirements file is under `参考项目/` and belongs to the reference project. Runtime API keys are documented in [.env.example](.env.example): copy it to `.env` and fill `DASHSCOPE_API_KEY` for ChatTongyi/DashScope embeddings and `TAVILY_API_KEY` for web search.

**FastAPI frontend (primary and only active frontend):**
```bash
python api_server_fastapi.py
# or on Windows:
start_server.bat
```
Port 5000. Entry point delegates to [api/app.py](api/app.py) (`create_app()`), which mounts static `css/` `js/` `assets/`, registers global exception handlers, and includes routers under `api/routers/`. Swagger at `/docs`.

**Streamlit frontend (ARCHIVED, do not use):** moved to [legacy/streamlit_app.py](legacy/streamlit_app.py) (originally root `app.py`). It is **not runnable** — `utils.conversation_manager` was removed, so it is kept only for historical reference. See [legacy/README.md](legacy/README.md). Do not attempt to fix it; work on the FastAPI main line instead.

## Architecture

### Single Active Frontend (FastAPI + vanilla HTML/CSS/JS)

- Entry [api_server_fastapi.py](api_server_fastapi.py) → [api/app.py](api/app.py) `create_app()` → routers in [api/routers/](api/routers/) (`chat`, `detection`, `knowledge`, `conversations`, `metrics`, `files`, `health`, `pages`).
- Singletons (agent, metrics, vector_store) come from [api/dependencies.py](api/dependencies.py) as lazy module-level singletons; agent + metrics are pre-loaded in a startup background thread.
- Frontend assets: `*.html` + `css/` + `js/` (`chat.js`, `detection.js`, `knowledge.js`, `metrics.js`, `auth.js`).

### Four Functional Pages

1. **SAR检测** — Upload SAR images → YOLO inference → display detection results with bounding boxes
2. **智能体问答** — Streaming chat, thought-chain visualization, file attachments, conversation persistence
3. **知识库管理** — Upload TXT/PDF → semantic/parent-child chunk → embed into ChromaDB vector store
4. **可观测性** — Agent metrics dashboard: tool call frequency, success rates, timeline, loop detection

### Agent Layer ([agent/](agent/))

- **react_agent.py** — `ReactAgent` wrapping LangChain's `create_agent`. Streams via `execute_stream(chat_pack, conversation_id=None, on_step=None)`. Thought-chain steps are pushed via the `on_step` callback (no global state) — callers collect them in request-local lists. `on_step` is optional, so legacy direct calls still work. **There is no global `_thought_chain` dict anymore.**
- **tools/agent_tools.py** — `@tool`-decorated functions: `rag_summarize`, `get_weather`, `get_scene_id`, `get_current_month`, `get_user_location`, `fetch_external_data`, `fill_context_for_report`, `get_sea_state`, `compare_scenes`, `get_scene_trend`, `web_search`, `detect_ships`, `extract_file_content`. `detect_ships` reuses the `get_yolo_model()` singleton from [api/dependencies.py](api/dependencies.py).
- **tools/middleware.py** — LangChain middleware:
  - `monitor_tool` (`@wrap_tool_call`): logs/metrics every tool call. **When the agent calls `fill_context_for_report`, it sets `runtime.context["report"] = True`** — this is the trigger for prompt switching, not `fetch_external_data`.
  - `log_before_model` (`@before_model`): pre-LLM hook for metrics
  - `report_prompt_switch` (`@dynamic_prompt`): on every model step, picks `report_prompt.txt` if `context["report"]` is True else `main_prompt.txt`. The flag is reset to False per user message.
- **metrics_collector.py** — `AgentMetrics` singleton for tool calls, success rates, LLM calls, conversation rounds. In-memory stats (read by `/api/metrics`) + dual-writes each event to MySQL `metric_events` via synchronous pymysql. `start/end_conversation` are called **only** in [api/routers/chat.py](api/routers/chat.py)'s SSE generator, not in `execute_stream`. Not user-isolated yet (all events write `user_id=1`).

### RAG Layer ([rag/](rag/))

- **vector_store.py** — ChromaDB singleton (`get_vector_store_service()`). Loads docs, applies semantic + parent-child chunking (controlled by [config/chroma.yml](config/chroma.yml)), embeds child chunks into Chroma, persists parent chunks to `parent_docstore.json`.
- **hybrid_retriever.py** — `DynamicHybridRetriever`: vector + BM25 ensemble with dynamic weights. BM25 index is pickled to disk keyed by the manifest SHA-256 hash; on load it verifies both the manifest hash and the `preprocess_func` qualname to detect drift, rebuilding if either mismatches.
- **parent_child_retriever.py** — `ParentChildResolver.resolve(child_docs)`: dedup hit child blocks → fetch corresponding parent blocks, preserving rerank order.
- **parent_docstore.py** — JSON-backed parent block store with a `threading.Lock` for thread-safe writes during ingestion.
- **reranker.py** — `BGERerankerService` (bge-reranker-base, CPU). Pure function by design: never mutates inputs, builds new `Document` instances with shallow-copied metadata + `rerank_score`. Default `score_threshold=0.3`; eval passes 0.0.
- **rag_service.py** — `RagSummarizeService.retriever_docs(query)` flow: retrieve children (hybrid) → rerank on children → resolve to parents in rerank order (RR→PC).

### Model Layer ([model/factory.py](model/factory.py))

Factory for `chat_model` (ChatTongyi, deepseek-v3.2) and `embeddings` (DashScopeEmbeddings, text-embedding-v4) — both DashScope. Configured via [config/rag.yml](config/rag.yml). Both are module-level singletons (eager init at import).

### Storage Layer: MySQL + SQLAlchemy 2.0 async (users, conversations, metrics)

User accounts, conversation history, and metric events are stored in **MySQL `sar_sense` via async SQLAlchemy + aiomysql**. Chroma holds vectors; `data/` holds original files; `manifest.json`/`parent_docstore.json` remain the knowledge-base store (not migrated). The knowledge base is **shared across all users**; only conversations are user-isolated.

- **config/db_conf.py** — `async_engine`, `AsyncSessionLocal`, `get_db()` dependency (commit/rollback/close).
- **models/** — shared `Base(DeclarativeBase)` + ORM models: `User`, `Conversation`, `ConversationMessage` (thought_steps as MySQL `JSON` column; `UNIQUE(conversation_id, message_index)`), `MetricEvent`.
- **schemas/** — Pydantic request models (`LoginRequest`, `RegisterRequest`, `CreateConversationRequest`, `AppendMessageRequest`).
- **crud/** — async CRUD functions taking `db: AsyncSession` as first arg. `crud/conversations.py` enforces user isolation (`WHERE user_id=?` on every read/write). `crud/metrics.py` is synchronous pymysql because LangChain middleware calls it from sync code.

### Auth Layer (JWT + RBAC)

- **utils/security.py** — `hash_password`/`verify_password` (bcrypt direct, no passlib) + `create_access_token`/`decode_token` (PyJWT, 7-day expiry).
- **api/auth.py** — `get_current_user(token)` decodes JWT → `{id, username}` (no DB hit per request); `require_admin(user)` checks `username == "admin"` → 403.
- **Frontend** ([js/auth.js](js/auth.js)) — token in `localStorage`, `apiFetch()` wrapper injects `Authorization: Bearer <token>`, 401 → redirect to login.
- **RBAC**: knowledge base `upload`/`delete` are `Depends(require_admin)`; `list` is `Depends(get_current_user)`. Seed user: `admin`/`admin123`.

### Detection ([Detct_prdc/](Detct_prdc/))

- `MBE-Net/weights/best.pt` — Trained YOLO weights for SAR ship detection
- `detect.py` — Standalone batch detection script

### Vendored Ultralytics ([ultralytics/](ultralytics/))

Forked copy of YOLO v8.3.9 with custom NN modules (mamba, cutlass, DCNv2/v3/v4, selective_scan, TransNeXt, etc.). Do not `pip install ultralytics` — Python imports resolve to the local copy. Model configs live in `ultralytics/cfg/models/`.

## Config ([config/](config/))

| File | Purpose |
|------|---------|
| `rag.yml` | Chat model (`deepseek-v3.2`), embedding model (`text-embedding-v4`) |
| `chroma.yml` | Collection name, chunk sizes, semantic/parent-child toggles, separators, `retrieve_k_children`, `retrieve_k_parents` |
| `prompts.yml` | Paths to prompt template files |
| `agent.yml` | External data path, Tavily API key |

All loaded at import time by [utils/config_handler.py](utils/config_handler.py) → exported as `rag_conf`, `chroma_conf`, `prompts_conf`, `agent_conf`.

## Key Technical Details

- **LLM Provider**: Alibaba Cloud DashScope (ChatTongyi / DashScopeEmbeddings)
- **Vector DB**: ChromaDB, persisted locally in `chroma_db/`
- **Parent docstore**: JSON file `parent_docstore.json` (threading.Lock for ingestion writes)
- **Knowledge base dedup**: by SHA-256 (`manifest.json`), **not MD5**
- **Detection Model**: YOLO via vendored ultralytics, loaded once via the `get_yolo_model()` lazy singleton in [api/dependencies.py](api/dependencies.py) (shared by `/api/detect` and the `detect_ships` agent tool)
- **Agent Framework**: LangChain `create_agent` (langgraph-based), streaming via `agent.stream()`
- **Conversation Storage**: MySQL `sar_sense` (async SQLAlchemy + aiomysql) — **not JSON files**
- **Environment**: `HF_ENDPOINT=https://hf-mirror.com` and `KMP_DUPLICATE_LIB_OK=TRUE` set defensively in `rag_service.py` and `reranker.py`
- **Logging**: Daily files in `logs/agent_YYYYMMDD.log`; FastAPI access log to `server.log`

## Important Patterns

- **Prompt switching**: middleware watches `fill_context_for_report` (a stub tool the LLM is told to call when the user asks for a report). On that signal, `@dynamic_prompt` swaps system prompt to `report_prompt.txt` for subsequent steps in the same turn. Per user message, the flag resets to False. **Trigger tool is `fill_context_for_report`, not `fetch_external_data`.**
- **Tool monitoring is decoupled**: tool functions don't know about logging/metrics — `@wrap_tool_call` middleware does it transparently.
- **RAG concurrency safety**: the reranker NEVER writes to its input Documents. BM25 returns shared long-lived Document objects from its index — mutating their `.metadata` in-place would race across concurrent queries. The reranker copies into fresh Documents instead.
- **Thought-chain via callback, not global state**: `execute_stream(on_step=...)` pushes steps to the caller; `chat.py` relays them through an `event_queue` as `thought_step` events (event-driven SSE, not polling).
- **Conversation isolation**: every conversation read/write in `crud/conversations.py` carries `user_id`; unauthorized access returns an empty dict (not 403) to avoid probing.
- **Metrics counted once**: `start/end_conversation` live only in `chat.py`'s SSE generator, never in `execute_stream`.
