# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAR-Sense: an intelligent platform for SAR (Synthetic Aperture Radar) ship detection. Combines a YOLO-based object detection model with a LangChain ReAct agent that has hybrid RAG (vector + BM25 + parent-child + BGE reranker), tool calling, and conversational memory. Includes a user system (JWT auth + RBAC) backed by MySQL.

## Running the App

**Conda environment:** `rag_env_backup` (required). Activate before any Python command. Dependencies also pinned in [requirements.txt](requirements.txt).

**FastAPI frontend (primary):**
```bash
python api_server_fastapi.py
```
Port 5000. Entry point delegates to [api/app.py](api/app.py) (`create_app()`), which mounts static `css/` `js/` `assets/`, registers global exception handlers, and includes routers under `api/routers/`. Swagger at `/docs`. Windows launcher: `start_server.bat`.

**Prerequisites:** MySQL running on localhost:3306, database `sar_sense` (utf8mb4, user/pass `root/root`). Tables auto-created on startup via `Base.metadata.create_all`. Seed user `admin`/`admin123` auto-created if users table empty.

**Streamlit frontend (legacy, currently broken):** `streamlit run app.py` — port 8501. **Not maintained this phase**: `ConversationManager` was removed and its methods moved to `crud/conversations.py` + `utils/conversation_builder.py`, so [app.py](app.py)'s direct calls will fail. FastAPI is the active frontend.

## Tests & Evaluation

```bash
# Functional tests (parent-child + hybrid retriever)
python tests/test_parent_child.py
python tests/test_parent_child_integration.py
python tests/test_hybrid_retriever.py

# RAG evaluation (50 hand-crafted QAs × 5 ablation pipelines × 3 k values)
python eval/evaluate.py                    # retrieval metrics only
python eval/evaluate.py --with-answers     # also generate RAG answers for external judge LLM
python eval/evaluate.py --limit 5 --with-answers   # smoke test
python eval/aggregate.py                   # merge external judge scores into final report

# Concurrency probes (root-level diagnostic scripts, not part of test suite)
python _bm25_identity_probe.py             # confirms BM25 returns shared Document refs
python _rerank_nomutate_probe.py           # confirms reranker doesn't mutate inputs
python _concurrency_probe.py               # parallel rag_summarize sanity check
```

## Architecture

### Dual Frontend (sharing one backend)

- **Streamlit** ([app.py](app.py)) — sidebar radio nav, inline CSS via `inject_custom_css()`, all session state in Python
- **FastAPI + vanilla HTML/CSS/JS** — entry [api_server_fastapi.py](api_server_fastapi.py) → [api/app.py](api/app.py) `create_app()` → routers in [api/routers/](api/routers/) (`chat`, `detection`, `knowledge`, `conversations`, `metrics`, `files`, `health`, `pages`). Singletons (agent, conv_manager, metrics) come from [api/dependencies.py](api/dependencies.py) and are pre-loaded in a startup background thread.

Both frontends share the same agent/rag/model layers below.

### Four Functional Pages

1. **SAR检测** — Upload → YOLO inference → boxes overlay
2. **智能体问答** — Streaming chat, thought-chain visualization, file attachments, conversation persistence
3. **知识库管理** — Upload TXT/PDF → semantic/parent-child chunk → embed into ChromaDB
4. **可观测性** — Metrics dashboard: tool call frequency, success rates, timeline, loop detection

### Agent Layer ([agent/](agent/))

- **[react_agent.py](agent/react_agent.py)** — `ReactAgent` wrapping LangChain's `create_agent`. Streams via `execute_stream(chat_pack, conversation_id=None, on_step=None)`. Thought-chain steps are pushed via the `on_step` callback (no global state) — callers collect them in request-local lists. `on_step` is optional, so legacy direct calls still work.
- **[tools/agent_tools.py](agent/tools/agent_tools.py)** — `@tool`-decorated functions: `rag_summarize`, `get_weather`, `get_scene_id`, `get_current_month`, `get_user_location`, `fetch_external_data`, `fill_context_for_report`, `get_sea_state`, `compare_scenes`, `get_scene_trend`, `web_search`, `detect_ships`, `extract_file_content`.
- **[tools/middleware.py](agent/tools/middleware.py)** — LangChain middleware:
  - `monitor_tool` (`@wrap_tool_call`): logs/metrics every tool call. **When the agent calls `fill_context_for_report`, it sets `runtime.context["report"] = True`** — this is the trigger for prompt switching, not the fact that any specific data tool ran.
  - `log_before_model` (`@before_model`): pre-LLM hook for metrics
  - `report_prompt_switch` (`@dynamic_prompt`): on every model step, picks `report_prompt.txt` if `context["report"]` is True else `main_prompt.txt`. The flag is reset to False per user message.
- **[metrics_collector.py](agent/metrics_collector.py)** — `AgentMetrics` singleton for tool calls, success rates, LLM calls, conversation rounds. In-memory stats (read by `/api/metrics`) + dual-writes each event to MySQL `metric_events` via synchronous pymysql (LangChain middleware is sync). `start/end_conversation` are called **only** in [api/routers/chat.py](api/routers/chat.py)'s SSE generator, not in `execute_stream` (avoids double-counting). Not user-isolated yet (all events write `user_id=1`).

### RAG Layer ([rag/](rag/))

- **[vector_store.py](rag/vector_store.py)** — ChromaDB singleton (`get_vector_store_service()`). Loads docs, applies semantic + parent-child chunking (controlled by [config/chroma.yml](config/chroma.yml)), embeds child chunks into Chroma, persists parent chunks to `parent_docstore.json`.
- **[hybrid_retriever.py](rag/hybrid_retriever.py)** — `DynamicHybridRetriever`: vector + BM25 ensemble. **Critical invariant**: `BM25Retriever.from_documents(docs)` keeps long-lived references to the `docs` list — every BM25 query returns those same Document objects (NOT fresh copies). Anything mutating their metadata in-place will race across concurrent queries.
- **[parent_child_retriever.py](rag/parent_child_retriever.py)** — `ParentChildResolver.resolve(child_docs)`: dedup hit child blocks → fetch corresponding parent blocks from docstore, preserving the input order. The first child hit per parent wins; its `rerank_score` is propagated to the parent's metadata for downstream sorting/display.
- **[parent_docstore.py](rag/parent_docstore.py)** — JSON-backed parent block store with file lock for thread-safe writes during ingestion.
- **[reranker.py](rag/reranker.py)** — `BGERerankerService` (bge-reranker-base, CPU). **Pure function by design**: never mutates inputs. Builds new `Document` instances with shallow-copied metadata + `rerank_score`. This is intentional concurrency safety — without it, two parallel queries reranking shared BM25 Documents would overwrite each other's scores. Default `score_threshold=0.3` gates low-quality results in production; eval/pipelines pass 0.0 for fairness.
- **[rag_service.py](rag/rag_service.py)** — `RagSummarizeService.retriever_docs(query)` flow (post-refactor): retrieve children (hybrid) → **rerank on children** (small text → CrossEncoder more accurate) → resolve to parents in rerank order. The rerank-on-children-then-resolve-parents order (RR→PC) was validated as ~MRR-equivalent to PC alone with ~65% lower latency vs. the old PC→RR order.

### Model Layer ([model/factory.py](model/factory.py))

Factory for `chat_model` (ChatTongyi, deepseek-v3.2) and `embeddings` (DashScopeEmbeddings, text-embedding-v4) — both DashScope. Configured via [config/rag.yml](config/rag.yml).

### Storage Layer: MySQL + SQLAlchemy 2.0 async (users, conversations, metrics)

User accounts, conversation history, and metric events are stored in **MySQL `sar_sense` via async SQLAlchemy + aiomysql**. Chroma still holds vectors; `data/` still holds original files; `manifest.json`/`parent_docstore.json` remain the knowledge-base store (not migrated). The knowledge base is **shared across all users**; only conversations are user-isolated.

Layered architecture (mirrors the `参考项目/` reference project):
- **[config/db_conf.py](config/db_conf.py)** — `async_engine` (aiomysql, pool_size=10, pool_recycle=3600 — `pool_pre_ping` is off due to an aiomysql ping-signature incompatibility), `AsyncSessionLocal`, `get_db()` dependency (commit/rollback/close).
- **[models/](models/)** — shared `Base(DeclarativeBase)` + 4 ORM models: `User`, `Conversation`, `ConversationMessage` (thought_steps as MySQL `JSON` column; `UNIQUE(conversation_id, message_index)`), `MetricEvent`.
- **[schemas/](schemas/)** — Pydantic request models (`LoginRequest`, `RegisterRequest`, `CreateConversationRequest`, `AppendMessageRequest`). `RegisterRequest.password` enforces `min_length=4`.
- **[crud/](crud/)** — async CRUD functions, each taking `db: AsyncSession` as first arg. `crud/conversations.py` enforces user isolation (`WHERE user_id=?` on every read/write; unauthorized access returns an empty dict, not 403, to avoid probing). `crud/metrics.py` is **synchronous pymysql** (not async) because LangChain middleware calls it from sync code.
- **Business code never touches a `Session` directly** — routes call crud, crud calls ORM.

### Auth Layer (JWT + RBAC)

- **[utils/security.py](utils/security.py)** — `hash_password`/`verify_password` (bcrypt direct, no passlib) + `create_access_token`/`decode_token` (PyJWT, 7-day expiry, `SECRET_KEY` constant — note "change to env for production").
- **[api/auth.py](api/auth.py)** — `oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")`; `get_current_user(token)` decodes JWT → `{id, username}` dict (no DB hit per request); `require_admin(user)` checks `username == "admin"` → 403. Routes: `POST /api/auth/register` (409 on duplicate), `POST /api/auth/login`, `GET /api/auth/me`.
- **Frontend** ([js/auth.js](js/auth.js)) — token in `localStorage`, `apiFetch()` wrapper injects `Authorization: Bearer <token>`, 401 → clear token + redirect to login.html, page guard via `<body data-public>`. All pages include auth.js before their page JS.
- **RBAC**: knowledge base `upload`/`delete` are `Depends(require_admin)`; `list` is `Depends(get_current_user)` (any logged-in user can view). Frontend hides the upload tab + delete buttons for non-admins.
- **Seed user**: `admin`/`admin123` (auto-created on startup if users table empty). Historical conversations are owned by admin.

### Global Exception Handling

[utils/exception_handlers.py](utils/exception_handlers.py) registers 4 handlers (subclass-before-parent): `HTTPException` → `{detail}` passthrough; `IntegrityError` → friendly message ("用户名已存在" etc.); `SQLAlchemyError` → 500 generic; `Exception` → 500 catch-all (no stack leak). **Routes have no try/except** except [api/routers/chat.py](api/routers/chat.py)'s SSE `generate()` — StreamingResponse can't be intercepted once started, so stream-internal exceptions yield an `error` event to the client. Response format unchanged (`{success:True,...}` on success, `{detail:'...'}` on failure) — frontend untouched.

### Detection ([Detct_prdc/](Detct_prdc/))

- `MBE-Net/weights/best.pt` — Trained YOLO weights for SAR ship detection
- `detect.py` — Standalone batch detection script

### Vendored Ultralytics ([ultralytics/](ultralytics/))

**Forked copy** of YOLO v8.3.9 with custom NN modules (mamba, cutlass, DCNv2/v3/v4, selective_scan, TransNeXt, etc.). **Do not `pip install ultralytics`** — Python imports resolve to the local copy. Model configs live in `ultralytics/cfg/models/`.

### Evaluation Module ([eval/](eval/))

Independent, non-invasive RAG evaluation harness — does NOT modify production code. Covered fully in [eval/README.md](eval/README.md). Key files:
- `qa_dataset.json` — 50 hand-crafted QAs across 5 SAR text files
- `pipelines.py` — 5+1 ablation pipelines (`vector_only`, `hybrid_no_pc_no_rr`, `hybrid_pc_no_rr`, `hybrid_no_pc_with_rr`, `full`, `full_v2_rr_then_pc`) all reusing production singletons
- `metrics.py` — `is_hit` (filename match + NFKC-normalized snippet containment), Recall@k, MRR
- `evaluate.py` — runs all pipelines × k ∈ {3,5,10}, optionally generates RAG answers for external Claude judging
- `aggregate.py` — merges externally-judged `scores.jsonl` into final report

## Config ([config/](config/))

| File | Purpose |
|------|---------|
| `rag.yml` | Chat model (`deepseek-v3.2`), embedding model (`text-embedding-v4`) |
| `chroma.yml` | Collection name, chunk sizes, semantic/parent-child toggles, separators, `retrieve_k_children`, `retrieve_k_parents` |
| `prompts.yml` | Paths to prompt template files |
| `agent.yml` | External data path, Tavily API key |

All loaded at import time by [utils/config_handler.py](utils/config_handler.py) → exported as `rag_conf`, `chroma_conf`, `prompts_conf`, `agent_conf`.

## Key Technical Details

- **LLM**: Alibaba Cloud DashScope (ChatTongyi `deepseek-v3.2` / DashScopeEmbeddings)
- **Vector DB**: ChromaDB persisted in `chroma_db/`
- **Parent docstore**: JSON file `parent_docstore.json` (file-locked for ingestion)
- **Detection model**: Local YOLO via vendored ultralytics, loaded once with `@st.cache_resource`
- **Agent framework**: LangChain `create_agent` (langgraph-based), streaming via `agent.stream()`
- **Primary storage**: MySQL `sar_sense` (async SQLAlchemy + aiomysql) for users/conversations/metrics. SQLite migration source `runtime/sar_sense.db` is archived.
- **Knowledge base storage**: `manifest.json` + `parent_docstore.json` (not migrated to MySQL — shared across users, low write concurrency, JSON suffices). Dedup by SHA-256 (`manifest.json`).
- **Environment**: `HF_ENDPOINT=https://hf-mirror.com` and `KMP_DUPLICATE_LIB_OK=TRUE` are set defensively in `rag_service.py` and `reranker.py`
- **Logging**: Daily files in `logs/agent_YYYYMMDD.log`; FastAPI access log to `server.log`

## Important Patterns

- **Prompt switching**: middleware watches `fill_context_for_report` (a stub tool the LLM is told to call when the user asks for a report). On that signal, `@dynamic_prompt` swaps system prompt to `report_prompt.txt` for subsequent steps in the same turn. Per user message, the flag resets to False.
- **Tool monitoring is decoupled**: tool functions don't know about logging/metrics — `@wrap_tool_call` middleware does it transparently.
- **RAG concurrency safety** (subtle, important): the reranker NEVER writes to its input Documents. BM25 returns shared long-lived Document objects from its index — mutating their `.metadata` in-place would race across concurrent queries. The reranker copies (page_content + shallow-copied metadata) into fresh Documents instead. Honor this contract anywhere else in the pipeline that adds metadata to retrieved Documents.
- **Single-pass parent-child resolution**: child blocks come pre-sorted by rerank score; `ParentChildResolver` walks them once and keeps first-seen parent per parent_id, so the parent ordering inherits child rerank ordering for free.
- **Singletons everywhere**: `get_vector_store_service()`, `chat_model`, `embeddings`, the BGE reranker, `AgentMetrics` — all module-level singletons. Eval pipelines reuse them rather than rebuilding indices.
- **Thought-chain via callback, not global state**: `execute_stream(on_step=...)` pushes steps to the caller; `chat.py` relays them through an `event_queue` as `thought_step` events (event-driven SSE, not polling). Request-local lists — concurrent same-conversation requests no longer clobber each other.
- **Conversation isolation**: every conversation read/write in `crud/conversations.py` carries `user_id`; unauthorized access returns an empty dict (not 403) to avoid letting callers probe which conv_ids exist.
- **Metrics counted once**: `start/end_conversation` live only in `chat.py`'s SSE generator, never in `execute_stream` — prevents double-counting when agent is called.

## Things That Look Like Bugs But Aren't

- `fill_context_for_report`'s function body is a no-op stub returning a string — it exists *only* so the LLM has something to call; the side effect (prompt switch) happens in the tool-call middleware.
- The reranker's `score_threshold=0.3` returning `[]` for "low-quality" queries is intentional (keeps the model from hallucinating off bad context). Eval passes 0.0 to bypass this for fair ablation.
- `parent_child_retriever.py`'s `if not any(parent_id) → child_docs[:k]` fallback is for legacy data ingested before parent-child was enabled — it's the graceful path, not a bug.
- `crud/metrics.py` uses synchronous pymysql (not async) despite the rest being async — LangChain middleware is sync, so the metric write can't await an AsyncSession. Short-lived connection per write, failure only warns.
- `chat.py`'s SSE `generate()` keeps internal try/except even after global exception handlers were added — StreamingResponse can't be intercepted once streaming starts, so stream-internal exceptions must yield an `error` event to the client.
- `AgentMetrics` is a global singleton with per-process in-memory counters — `avg_response_time_s` can skew if two conversations run concurrently (the `conversation_start_time` field gets overwritten). Single-worker demo deployment makes this a non-issue; per-conversation timing is a future refactor.
