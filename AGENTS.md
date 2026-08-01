# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

> **Note**: This file is kept in sync with the current codebase. FastAPI is the sole active entry point.

## Project Overview

SAR-Sense: an intelligent platform for SAR (Synthetic Aperture Radar) ship detection. Combines a YOLO-based object detection model with a LangChain ReAct agent that has hybrid RAG (vector + BM25 + parent-child + BGE reranker), tool calling, and conversational memory. Includes a user system (JWT auth + RBAC) backed by MySQL.

## Running the App

**Conda environment:** `rag_env_backup` is required. Activate it before any Python command:
```bash
conda activate rag_env_backup
```
Install dependencies from the root [requirements.txt](requirements.txt). Runtime configuration is documented in [.env.example](.env.example); copy it to `.env` and fill the required model and external-service settings.

**FastAPI frontend (primary and only active frontend):**
```bash
python api_server_fastapi.py
# or on Windows:
start_server.bat
```
Port 5000. Entry point delegates to [api/app.py](api/app.py) (`create_app()`), which mounts static `css/` `js/` `assets/`, registers global exception handlers, and includes routers under `api/routers/`. Swagger at `/docs`.

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
- **metrics_collector.py** — `AgentMetrics` singleton keeps request-time counters in memory and persists every tool, LLM, and conversation-timing event with the authenticated `user_id` through `MetricsRepository`. `/api/metrics` reads global historical aggregates across all users from MySQL through the same Repository, so dashboard totals survive process restarts. This dashboard is not user-isolated yet. `start/end_conversation` are called **only** in [api/routers/chat.py](api/routers/chat.py)'s SSE generator, not in `execute_stream`.

### RAG Layer ([rag/](rag/))

- **vector_store.py** — ChromaDB singleton (`get_vector_store_service()`). Loads docs, applies semantic + parent-child chunking (controlled by [config/chroma.yml](config/chroma.yml)), embeds versioned child chunks into Chroma, and persists document metadata and parent chunks to MySQL. Uploads activate a complete new generation before cleaning up the old one.
- **hybrid_retriever.py** — `DynamicHybridRetriever`: vector + BM25 ensemble with dynamic weights. Both routes filter by active MySQL chunk ids, so failed or superseded generations cannot be recalled. The BM25 pickle is keyed by a database-derived knowledge fingerprint plus the `preprocess_func` qualname.
- **parent_child_retriever.py** — `ParentChildResolver.resolve(child_docs)`: dedup hit child blocks → fetch corresponding parent blocks, preserving rerank order.
- **repositories/parent_chunk_repository.py** — `ParentChunkRepository` owns runtime SQLAlchemy persistence and batched lookup for `parent_chunks`, avoiding one query per hit.
- **reranker.py** — `BGERerankerService` (bge-reranker-base, CPU). Pure function by design: never mutates inputs, builds new `Document` instances with shallow-copied metadata + `rerank_score`. Default `score_threshold=0.3`; eval passes 0.0.
- **rag_service.py** — `RagSummarizeService.retriever_docs(query)` flow: retrieve children (hybrid) → rerank on children → resolve to parents in rerank order (RR→PC).

### Model Layer ([model/factory.py](model/factory.py))

Factory for the configured chat provider (Ollama or DashScope) and DashScope embeddings. The current [config/rag.yml](config/rag.yml) selects local Ollama for chat and DashScope for embeddings. Both are module-level singletons (eager init at import).

### Storage Layer: MySQL + SQLAlchemy 2.0 (async request data + sync RAG/metrics)

MySQL `sar_sense` is the source of truth for users, conversations, knowledge metadata, parent chunks, ACL, and metric events. FastAPI request CRUD uses async SQLAlchemy + aiomysql; synchronous LangChain/RAG persistence is isolated in `repositories/` and uses SQLAlchemy + PyMySQL through `SyncSessionLocal`. Chroma holds child vectors and `data/.knowledge_versions/<uuid>/` holds immutable, versioned source files. Local `manifest.json` and `parent_docstore.json` files, if present, are obsolete backups and are not required by the application.

- **config/db_conf.py** — `async_engine`, `AsyncSessionLocal`, `get_db()` dependency (commit/rollback/close).
- **models/** — shared `Base(DeclarativeBase)` + ORM models for users, conversations/messages, `KnowledgeDocument`, `ParentChunk`, and `MetricEvent`.
- **schemas/** — Pydantic request models (`LoginRequest`, `RegisterRequest`, `CreateConversationRequest`, `AppendMessageRequest`).
- **repositories/** — synchronous persistence boundary for LangChain/RAG and metrics: `KnowledgeRepository` owns knowledge metadata and ingestion generations, `ParentChunkRepository` owns parent chunks, and `MetricsRepository` owns metric-event persistence and historical aggregation.
- **crud/** — only request-facing async CRUD taking `db: AsyncSession`; `crud/conversations.py` enforces user isolation (`WHERE user_id=?` on every read/write).

### Auth Layer (JWT + RBAC)

- **utils/security.py** — `hash_password`/`verify_password` (bcrypt direct, no passlib) + `create_access_token`/`decode_token` (PyJWT, 7-day expiry).
- **api/auth.py** — `get_current_user(token, db)` decodes the JWT, then reloads the current user from MySQL so role changes take effect on the next request. `require_admin(user)` checks `role == "admin"` and returns 403 otherwise.
- **Frontend** ([js/auth.js](js/auth.js)) — token in `localStorage`, `apiFetch()` wrapper injects `Authorization: Bearer <token>`, 401 → redirect to login.
- **RBAC**: knowledge upload/delete/ACL changes and user-role management require `admin`; list/download/retrieval enforce each document's `allowed_roles`. The seed administrator is configured through `ADMIN_USERNAME` and `ADMIN_PASSWORD`; production startup rejects an empty password.

### Detection ([Detct_prdc/](Detct_prdc/))

- `MBE-Net/weights/best.pt` — Trained YOLO weights for SAR ship detection
- `detect.py` — Standalone batch detection script

### Vendored Ultralytics ([ultralytics/](ultralytics/))

Forked copy of YOLO v8.3.9 with custom NN modules (mamba, cutlass, DCNv2/v3/v4, selective_scan, TransNeXt, etc.). Do not `pip install ultralytics` — Python imports resolve to the local copy. Model configs live in `ultralytics/cfg/models/`.

## Config ([config/](config/))

| File | Purpose |
|------|---------|
| `rag.yml` | Chat provider/model, embedding model, reranker, and MinerU endpoint |
| `chroma.yml` | Collection name, chunk sizes, semantic/parent-child toggles, separators, `retrieve_k_children`, `retrieve_k_parents` |
| `prompts.yml` | Paths to prompt template files |
| `agent.yml` | External data path, Tavily API key |

All loaded at import time by [utils/config_handler.py](utils/config_handler.py) → exported as `rag_conf`, `chroma_conf`, `prompts_conf`, `agent_conf`.

## Key Technical Details

- **LLM Provider**: configurable Ollama or DashScope chat model; DashScope embeddings
- **Vector DB**: ChromaDB, persisted locally in `chroma_db/`
- **Parent storage**: MySQL `parent_chunks`
- **Knowledge base metadata/dedup**: MySQL `knowledge_documents`, keyed by SHA-256-derived document identity, **not MD5**
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
- **Agent concurrency is bounded**: synchronous Agent runs are submitted to one shared `AgentExecutor` thread pool instead of creating an unbounded thread per request. SSE still consumes request-local `asyncio.Queue` events, so client disconnects do not cancel the background Agent run.
- **Conversation isolation**: every conversation read/write in `crud/conversations.py` carries `user_id`; unauthorized access returns an empty dict (not 403) to avoid probing.
- **Metrics counted once**: `start/end_conversation` live only in `chat.py`'s SSE generator, never in `execute_stream`.
- **Knowledge updates are generation-based**: source files use immutable version paths; a new document generation becomes active atomically before obsolete Chroma/parent data is cleaned up.
- **Cross-store cleanup is eventual**: MySQL, Chroma, and the filesystem do not share one transaction. A forced stop can leave inactive vectors, parent rows, or version files; active-id filtering prevents recall, but maintenance cleanup may still be required.
- **Metrics reset scope**: MySQL deletion and in-memory reset share a process-wide lock. This is sufficient for the current single-worker deployment; multi-worker deployment requires a database or distributed lock.
