# Changelog

本文件记录 SAR-Sense 项目的工程化改造，方便答辩/面试时回顾。日期为 2026-06-22 ~ 2026-06-23。

## 2026-06-23 工程化优化（减法 + 加固）

### 架构精简
- **删除 `ConversationManager` 空壳类**：6 个方法全是一行转发到 crud，无状态无业务逻辑。删掉类 + `get_conv_manager()` 单例 getter，路由直接调 `crud.conversations`。`build_chat_pack`（有真实摘要压缩逻辑）抽到 [utils/conversation_builder.py](utils/conversation_builder.py) 作为独立函数。对齐参考项目的 `routers → crud → models` 分层。
- **修复 `has_been_compressed` 判断 bug**：从 `"summary_up_to" in conv_data` 改为 `bool(summary)`。新 crud 返回的 dict 永远带 `summary_up_to` key，`in` 判断失效。

### Bug 修复
- **指标重复统计**：`agent/react_agent.py` 的 `execute_stream` 里删掉 `start_conversation`/`end_conversation` 调用——路由 `chat.py` 已经调过一次，agent 内部再调导致 `conversation_rounds` 和 `avg_response_time` 翻倍。指标统计只保留在路由层 SSE generator 的 try/finally。
- **`_thought_chains` 全局状态污染**：删掉模块级全局 dict，`execute_stream` 加 `on_step` 回调参数。调用方在请求局部 list 收集步骤——同 conversation_id 并发不再串台/清零/泄漏。`chat.py` 的 SSE 从"轮询读全局 dict（50ms 延迟）"改成"事件驱动（thought_step 走 event_queue）"，延迟接近 0。Streamlit `app.py` 同步改用 `on_step=local_steps.append`。

### 安全加固
- **文件上传路径穿越修复**：[api/routers/files.py](api/routers/files.py) + [api/routers/detection.py](api/routers/detection.py) 的 `file.filename` 改用 `os.path.basename()`，防 `../../evil.py` 跳出临时目录。知识库路由本来就有 basename，现在三个上传入口统一。
- **知识库 RBAC**：[api/auth.py](api/auth.py) 新增 `require_admin` 依赖；知识库 `upload`/`delete` 限 admin（403），`list` 登录即可。前端 [knowledge.html](knowledge.html) 非 admin 隐藏上传 tab + 删除按钮。

### 全局异常处理
- **新增 [utils/exception_handlers.py](utils/exception_handlers.py)**：4 个处理器（HTTPException / IntegrityError / SQLAlchemyError / Exception 兜底），`register_exception_handlers(app)` 在 `api/app.py` 注册。
- **路由删 20 处 try/except 样板**：6 个路由文件（conversations/knowledge/files/metrics/detection/chat 外层）。`chat.py` 的 SSE `generate()` 内部 try/except 保留——StreamingResponse 已开始后全局处理器接不住流内异常。
- **修复堆栈泄漏**：原来 `detail=str(e)` 把 SQL 错误返回前端，现在系统异常只返回通用提示，详情记日志。IntegrityError 自动转友好提示（"用户名已存在"等）。响应格式不变（`{detail}` 失败 / `{success:True,...}` 成功），前端零改动。

### 文档
- **更新 [CLAUDE.md](CLAUDE.md)**：补充 MySQL 存储层、JWT 认证、RBAC、全局异常处理、on_step 回调等章节；标注 Streamlit 失效、指标并发局限等已知边界。

---

## 2026-06-22 存储层迁移 + 用户系统

### SQLite → MySQL（异步 SQLAlchemy + aiomysql）
- 对齐根目录 `参考项目/` 的分层架构：新建 `config/db_conf.py`（异步 engine + get_db 依赖）、`models/`（4 个 ORM 模型）、`schemas/`（Pydantic 请求模型）、`crud/`（异步 CRUD 函数）。
- 业务表 4 张：`users` / `conversations` / `conversation_messages`（thought_steps 用 MySQL JSON 列，UNIQUE(conversation_id, message_index)）/ `metric_events`。
- 表结构由 `Base.metadata.create_all` 在 startup 自动创建；种子用户 `admin`/`admin123` 自动创建。
- [utils/migrate_sqlite_to_mysql.py](utils/migrate_sqlite_to_mysql.py)：一次性把旧 SQLite（runtime/sar_sense.db）的 1 用户 + 29 对话 + 62 消息 + 3 指标导入 MySQL。
- 删掉旧 `utils/db.py`（同步 sqlite3）+ `utils/migrate_sqlite.py`。
- AgentMetrics 保留内存计数（`/api/metrics` 读内存），额外同步 pymysql 写 `metric_events`（LangChain middleware 是同步的，不能走 async session）。

### 用户注册登录 + 对话隔离
- JWT + Authorization 头认证（PyJWT + bcrypt，无 passlib）。新增 [utils/security.py](utils/security.py)（哈希 + JWT 工具）+ [api/auth.py](api/auth.py)（register/login/me 路由 + `get_current_user` 依赖）。
- `ConversationManager`（当时还在）6 个方法加 `*, user_id` 必填参数 + SQL `WHERE user_id=?`。越权访问返回空 dict 而非 403，避免探测。
- 前端新增 [js/auth.js](js/auth.js)（token 存 localStorage + `apiFetch` 自动带 Authorization 头 + 401 跳登录 + 页面守卫 + navbar 登录态 UI）+ [login.html](login.html)（登录/注册合一，含卡通角色动画）。7 个 HTML 加 auth.js script，chat.js 的 7 处 fetch 改 apiFetch。
- 种子用户 id=1（admin），历史 29 条对话天然归属，零数据迁移。

### 之前已回退的尝试
- 一开始做过一版 MySQL + SQLAlchemy ORM + Repository 层的完整改造，已回退。后续评估后改用更轻的 SQLite，再后来按参考项目方式迁回 MySQL 异步分层架构（即当前版本）。
