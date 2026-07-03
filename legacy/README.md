# legacy/

本目录存放**已弃用、不再维护**的早期前端实现。

## `streamlit_app.py`（原根目录 `app.py`）

早期 Streamlit 单文件前端。**当前已不可运行**，仅作历史存档保留。

### 为什么归档

- 主线前端已切换为 **FastAPI**（入口 [api_server_fastapi.py](../api_server_fastapi.py) → [api/app.py](../api/app.py)），端口 5000。
- Streamlit 版依赖的 `utils.conversation_manager.ConversationManager` 已被移除（方法迁至 [crud/conversations.py](../crud/conversations.py) + [utils/conversation_builder.py](../utils/conversation_builder.py)），直接 `streamlit run legacy/streamlit_app.py` 会因 `ImportError` 失败。
- 流式问答、思维链、会话持久化、鉴权等能力都只在 FastAPI 主线实现，Streamlit 版未跟进。

### 不要尝试修复

如需双前端展示，应优先在 FastAPI 主线上完善，而非修复此文件。此文件已做最小改动让 `ImportError` 不再硬阻断加载（`ConversationManager` 降级为 `None`），但运行时仍会因缺失 `ConversationManager` 的方法调用而失败——这是有意为之，标记其为历史代码。

## 活跃入口

```bash
conda activate rag_env_backup
python api_server_fastapi.py
# http://localhost:5000
```
