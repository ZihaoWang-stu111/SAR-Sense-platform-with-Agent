# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SAR-Sense: an intelligent platform for SAR (Synthetic Aperture Radar) ship detection. Combines a YOLO-based object detection model with a LangChain ReAct agent that has RAG capabilities, tool calling, and conversational memory.

## Running the App

**Conda environment:** `rag_env_backup` (required — no requirements.txt exists)

**Streamlit frontend (primary):**
```bash
streamlit run app.py
```
Port 8501. Single-file monolith (~1260 lines). All UI changes go here.

**FastAPI frontend (alternative):**
```bash
python api_server_fastapi.py
```
Port 5000. Serves HTML/CSS/JS frontend + REST API. API docs at `/docs`. Windows launcher: `start_server.bat`.

## Architecture

### Dual Frontend

The project has two complete frontend implementations:
- **Streamlit** (`app.py`) — sidebar radio navigation, inline CSS via `inject_custom_css()`, all session state managed in Python
- **FastAPI + vanilla HTML/CSS/JS** (`api_server_fastapi.py` serving `*.html` + `css/` + `js/`) — multi-page layout with `chat.js`, `detection.js`, `knowledge.js`, `metrics.js`

Both share the same backend agent/rag/model layers.

### Four Pages

1. **SAR检测** — Upload SAR images → YOLO inference → display detection results with bounding boxes
2. **智能体问答** — Chat interface with streaming output, thought-chain visualization, file attachments, conversation persistence
3. **知识库管理** — Upload TXT/PDF → chunk → embed into ChromaDB vector store
4. **可观测性** — Agent metrics dashboard: tool call frequency, success rates, timeline, loop detection

### Agent Layer (`agent/`)

- **react_agent.py** — `ReactAgent` class wrapping LangChain's `create_agent`. Streams via `execute_stream()`. Uses a global `_thought_chain` dict (mutated during streaming, snapshot-copied into message history).
- **tools/agent_tools.py** — All `@tool`-decorated functions: `rag_summarize`, `get_weather`, `get_scene_id`, `get_current_month`, `get_user_location`, `fetch_external_data`, `fill_context_for_report`, `get_sea_state`, `compare_scenes`, `get_scene_trend`, `web_search`, `detect_ships`, `extract_file_content`
- **tools/middleware.py** — LangChain middleware: `monitor_tool` (wraps tool calls for logging/metrics, sets `report` flag), `log_before_model` (pre-LLM hook), `report_prompt_switch` (`@dynamic_prompt` — swaps system prompt between `main_prompt.txt` and `report_prompt.txt` based on context flag)
- **metrics_collector.py** — `AgentMetrics` singleton tracking tool calls, success rates, LLM calls, conversation rounds

### RAG Layer (`rag/`)

- **vector_store.py** — ChromaDB vector store (document loading, chunking, embedding, retrieval)
- **rag_service.py** — `RagSummarizeService` orchestrating retrieval → reranking → LLM summarization
- **hybrid_retriever.py** — Hybrid retrieval combining vector search with other strategies
- **reranker.py** — BGE-based reranker (bge-reranker-base via sentence-transformers)

### Model Layer (`model/`)

- **factory.py** — Factory for ChatTongyi (chat model) and DashScopeEmbeddings. Configured via `config/rag.yml`.

### Detection (`Detct_prdc/`)

- **MBE-Net/weights/best.pt** — Trained YOLO model weights for SAR ship detection
- **detect.py** — Standalone batch detection script

### Vendored Ultralytics

`ultralytics/` is a **forked copy** of Ultralytics YOLO v8.3.9 with custom neural network modules (mamba, cutlass, DCNv2/v3/v4, selective_scan, TransNeXt, etc.). Do not `pip install ultralytics` — use the local version. Model configs are in `ultralytics/cfg/models/`.

## Config (`config/`)

| File | Purpose |
|------|---------|
| `rag.yml` | Chat model (`deepseek-v3.2`), embedding model (`text-embedding-v4`) |
| `chroma.yml` | Collection name, chunk sizes, file type filters, separators |
| `prompts.yml` | Paths to prompt template files |
| `agent.yml` | External data path, Tavily API key |

All configs are loaded at import time by `utils/config_handler.py` → exported as `rag_conf`, `chroma_conf`, `prompts_conf`, `agent_conf`.

## Key Technical Details

- **LLM Provider**: Alibaba Cloud DashScope (ChatTongyi / DashScopeEmbeddings)
- **Vector DB**: ChromaDB, persisted locally in `chroma_db/`
- **Detection Model**: YOLO (via vendored ultralytics), loaded once via `@st.cache_resource`
- **Agent Framework**: LangChain ReAct pattern, streaming via `agent.stream()`
- **Conversation Storage**: JSON files in `conversations/`, one file per conversation
- **Environment**: Requires `HF_ENDPOINT=https://hf-mirror.com` for model downloads (set in `rag_service.py`)
- **Logging**: Daily log files in `logs/agent_YYYYMMDD.log`

## Important Patterns

- The agent's system prompt in `main_prompt.txt` contains strict rules about intent classification (knowledge QA vs data query vs image detection vs report generation)
- `@dynamic_prompt` middleware swaps the system prompt mid-ReAct-loop when `fetch_external_data` is called (report mode), resetting per user message via `context={"report": False}`
- Tool calls are monitored via LangChain middleware (`@wrap_tool_call`), not by modifying tool code
- Knowledge base uses MD5 deduplication (`md5.text`) to avoid re-embedding duplicate documents
