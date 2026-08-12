# RAG 引用证据抽屉实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让聊天回答中的 RAG 引用编号和来源条目可点击，并在经过实时文档 ACL 校验后展示对应父块全文。

**Architecture:** 后端新增一个只读证据接口，复用 `ParentChunkRepository.get()`、`get_document_acl()` 和现有 JWT/RBAC，不新增表，也不查询 Chroma。RAG 来源协议只增加可选页码；前端兼容新旧协议，通过聊天区的一次事件委托打开右侧抽屉。

**Tech Stack:** FastAPI、Pydantic、SQLAlchemy、vanilla JavaScript、CSS、pytest/unittest、Node.js 语法检查

---

## 文件边界

- 修改 `schemas/knowledge.py`：定义证据响应模型。
- 修改 `api/routers/knowledge.py`：提供受保护的父块证据接口，并复用现有下载权限判断。
- 修改 `rag/rag_service.py`：在来源行中输出可选页码。
- 修改 `js/chat.js`：兼容新旧来源协议、隐藏内部 ID、处理引用点击和证据抽屉。
- 修改 `templates/chat.html`：增加抽屉 DOM，并更新静态资源版本号。
- 修改 `css/style_v2.css`：增加抽屉和可点击来源样式。
- 修改 `tests/test_knowledge_api_mysql.py`：覆盖证据接口 ACL、active generation 和统一 404。
- 修改 `tests/test_rag_citation_grounding.py`：覆盖来源页码输出。
- 修改 `tests/test_chat_live_status.py`：覆盖来源协议兼容和抽屉前端契约。

不新增 Repository、数据库迁移、前端框架、状态库或 LLM 调用。

---

### Task 1: 增加受 ACL 保护的父块证据接口

**Files:**
- Modify: `schemas/knowledge.py`
- Modify: `api/routers/knowledge.py`
- Test: `tests/test_knowledge_api_mysql.py`

- [ ] **Step 1: 为证据读取写失败测试**

在 `make_doc()` 的默认值中加入当前 generation 的子块：

```python
"chunk_ids": ["parent-1:child:000"],
```

在 `KnowledgeAPIWithMySQLTest` 中增加以下场景：

```python
async def test_evidence_returns_authorized_active_parent(self):
    record = {
        "page_content": "完整父块正文",
        "metadata": {"doc_id": "doc-1", "filename": "paper.pdf", "page": 7},
    }

    async def inline_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch.object(knowledge.parent_chunk_repository, "get", return_value=record),
        patch.object(knowledge, "run_in_threadpool", side_effect=inline_threadpool),
        patch.object(knowledge, "get_document_acl", AsyncMock(return_value=make_doc())),
        patch.object(knowledge, "_document_file_path", return_value="paper.pdf"),
    ):
        response = await knowledge.get_knowledge_evidence(
            "parent-1",
            user={"id": 8, "role": "researcher", "username": "reader"},
            db=SimpleNamespace(),
        )

    assert response.model_dump() == {
        "filename": "paper.pdf",
        "page": 7,
        "content": "完整父块正文",
        "doc_id": "doc-1",
        "download_url": "/api/knowledge/files/doc-1/download",
    }


async def test_evidence_admin_can_read_admin_only_document(self):
    record = {
        "page_content": "管理员证据",
        "metadata": {"doc_id": "doc-1", "page": 0},
    }

    async def inline_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    with (
        patch.object(knowledge.parent_chunk_repository, "get", return_value=record),
        patch.object(knowledge, "run_in_threadpool", side_effect=inline_threadpool),
        patch.object(
            knowledge,
            "get_document_acl",
            AsyncMock(return_value=make_doc(allowed_roles=[])),
        ),
        patch.object(knowledge, "_document_file_path", return_value=None),
    ):
        response = await knowledge.get_knowledge_evidence(
            "parent-1",
            user={"id": 1, "role": "admin", "username": "admin"},
            db=SimpleNamespace(),
        )

    assert response.content == "管理员证据"
    assert response.download_url is None


async def test_evidence_uses_uniform_404_for_unavailable_resources(self):
    cases = [
        (None, None),
        ({"page_content": "x", "metadata": {}}, None),
        (
            {"page_content": "x", "metadata": {"doc_id": "doc-1"}},
            make_doc(status="failed"),
        ),
        (
            {"page_content": "x", "metadata": {"doc_id": "doc-1"}},
            make_doc(allowed_roles=["business"]),
        ),
        (
            {"page_content": "x", "metadata": {"doc_id": "doc-1"}},
            make_doc(chunk_ids=["another-parent:child:000"]),
        ),
    ]

    async def inline_threadpool(func, *args, **kwargs):
        return func(*args, **kwargs)

    for record, document in cases:
        with (
            patch.object(knowledge.parent_chunk_repository, "get", return_value=record),
            patch.object(knowledge, "run_in_threadpool", side_effect=inline_threadpool),
            patch.object(
                knowledge,
                "get_document_acl",
                AsyncMock(return_value=document),
            ),
        ):
            with self.assertRaises(HTTPException) as raised:
                await knowledge.get_knowledge_evidence(
                    "parent-1",
                    user={"id": 8, "role": "researcher", "username": "reader"},
                    db=SimpleNamespace(),
                )
        self.assertEqual(raised.exception.status_code, 404)
        self.assertEqual(raised.exception.detail, "Evidence not found")
```

同时把 `/evidence/{parent_id}` 加入现有 `test_orm_response_routes_declare_pydantic_models` 的响应模型断言。

- [ ] **Step 2: 运行测试并确认失败**

Run:

```powershell
E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_knowledge_api_mysql.py -v
```

Expected: FAIL，提示 `parent_chunk_repository`、`get_knowledge_evidence` 或响应模型尚不存在。

- [ ] **Step 3: 增加响应模型和最小接口实现**

在 `schemas/knowledge.py` 增加：

```python
class KnowledgeEvidenceResponse(BaseModel):
    filename: str
    page: int | None = None
    content: str
    doc_id: str
    download_url: str | None = None
```

在 `api/routers/knowledge.py`：

1. 导入 `ParentChunkRepository` 和 `KnowledgeEvidenceResponse`。
2. 创建模块级 `parent_chunk_repository = ParentChunkRepository()`；构造本身不连接数据库。
3. 增加统一可见性函数，并让现有下载接口复用它：

```python
def _can_read_document(document, user: dict) -> bool:
    return bool(
        document
        and document.status == "active"
        and (
            is_admin(user)
            or user.get("role", "guest") in (document.allowed_roles or [])
        )
    )


def _is_active_parent(document, parent_id: str) -> bool:
    prefix = f"{parent_id}:child:"
    return any(
        chunk_id == parent_id or chunk_id.startswith(prefix)
        for chunk_id in (document.chunk_ids or [])
    )
```

4. 在 `/files/{doc_id}/download` 之前增加接口：

```python
@router.get(
    "/evidence/{parent_id}",
    response_model=KnowledgeEvidenceResponse,
)
async def get_knowledge_evidence(
    parent_id: str,
    user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> KnowledgeEvidenceResponse:
    record = await run_in_threadpool(parent_chunk_repository.get, parent_id)
    metadata = dict((record or {}).get("metadata") or {})
    doc_id = metadata.get("doc_id")
    document = await get_document_acl(db, doc_id) if doc_id else None

    if not _can_read_document(document, user) or not _is_active_parent(document, parent_id):
        raise HTTPException(status_code=404, detail="Evidence not found")

    raw_page = metadata.get("page")
    try:
        page = int(raw_page) if raw_page not in (None, "", "-") else None
    except (TypeError, ValueError):
        page = None

    download_url = None
    if _document_file_path(document) is not None:
        download_url = f"/api/knowledge/files/{document.doc_id}/download"

    return KnowledgeEvidenceResponse(
        filename=document.filename,
        page=page,
        content=record["page_content"],
        doc_id=document.doc_id,
        download_url=download_url,
    )
```

现有 `download_knowledge_file()` 中把重复判断替换成：

```python
if not _can_read_document(document, user):
    raise HTTPException(status_code=404, detail="Document not found")
```

- [ ] **Step 4: 运行接口测试并确认通过**

Run:

```powershell
E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_knowledge_api_mysql.py -v
```

Expected: PASS。

- [ ] **Step 5: 提交后端接口**

```powershell
git add schemas/knowledge.py api/routers/knowledge.py tests/test_knowledge_api_mysql.py
git commit -m "feat(knowledge): 增加受保护的引用证据接口"
```

---

### Task 2: 扩展来源协议并兼容历史消息

**Files:**
- Modify: `rag/rag_service.py`
- Modify: `js/chat.js`
- Test: `tests/test_rag_citation_grounding.py`
- Test: `tests/test_chat_live_status.py`

- [ ] **Step 1: 写来源页码和兼容性失败测试**

在 `tests/test_rag_citation_grounding.py` 中，把带 `page=7` 的现有期望更新为：

```python
assert result == (
    "结论得到内部证据支持[1]\n\n"
    "参考来源：\n"
    "[1] paper.pdf | chunk_id=chunk-1 | page=7 | score=0.8765"
)
```

保留其他无 `page` 文档的旧格式断言，证明页码确实可选。

在 `tests/test_chat_live_status.py::test_only_structured_rag_sources_are_folded` 中使用以下 cases：

```python
cases = [
    ("[1] paper.pdf | chunk_id=abc | page=7 | score=0.9876", True),
    ("[1] paper.pdf | chunk_id=abc | score=0.9876", True),
    ("[1] 普通正文", False),
    ("[1] paper.pdf | score=0.9876", False),
    ("[1] paper.pdf | chunk_id=abc", False),
]
```

- [ ] **Step 2: 运行两个测试并确认失败**

```powershell
E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_rag_citation_grounding.py tests/test_chat_live_status.py -v
```

Expected: 至少页码来源断言和新格式解析失败。

- [ ] **Step 3: 后端只在有页码时追加字段**

在 `RagSummarizeService.rag_summarize()` 的来源构造处替换为：

```python
source = f"[{i}] {filename} | chunk_id={chunk_id}"
if page not in (None, "", "-"):
    source += f" | page={page}"
source += f" | score={score}"
sources.append(source)
```

- [ ] **Step 4: 前端解析器兼容新旧格式**

在 `js/chat.js` 修改正则：

```javascript
const citationSourceLinePattern = /^\[(\d+)\]\s*([^|]+)\s*\|\s*chunk_id=(\S+)(?:\s*\|\s*page=(\S+))?\s*\|\s*score=(\S+)$/;
```

`hasStructuredCitationSource()` 改为检查第 3、5 组：

```javascript
return Boolean(sourceMatch?.[3] && sourceMatch?.[5]);
```

单次和多次来源解析都改成：

```javascript
{
  index: parseInt(m[1]),
  filename: (m[2] || '').trim(),
  parentId: m[3] || '',
  page: m[4] && m[4] !== '-' ? m[4] : null,
  score: m[5] || '-'
}
```

本步骤只调整数据解析，暂不改视觉输出。

- [ ] **Step 5: 运行测试和 JavaScript 语法检查**

```powershell
E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_rag_citation_grounding.py tests/test_chat_live_status.py -v
node --check js/chat.js
```

Expected: 全部 PASS，Node 退出码为 0。

- [ ] **Step 6: 提交来源协议**

```powershell
git add rag/rag_service.py js/chat.js tests/test_rag_citation_grounding.py tests/test_chat_live_status.py
git commit -m "feat(rag): 为引用来源补充页码"
```

---

### Task 3: 实现可点击引用和右侧证据抽屉

**Files:**
- Modify: `templates/chat.html`
- Modify: `js/chat.js`
- Modify: `css/style_v2.css`
- Test: `tests/test_chat_live_status.py`

- [ ] **Step 1: 写前端契约失败测试**

在 `tests/test_chat_live_status.py` 增加：

```python
def test_evidence_drawer_markup_and_safe_rendering(self):
    html = Path("templates/chat.html").read_text(encoding="utf-8")
    css = Path("css/style_v2.css").read_text(encoding="utf-8")

    for element_id in (
        "evidenceDrawer",
        "evidenceDrawerClose",
        "evidenceDrawerContent",
        "evidenceDrawerDownload",
    ):
        self.assertIn(f'id="{element_id}"', html)

    self.assertIn("function openEvidenceDrawer", self.source)
    self.assertIn("content.textContent = data.content", self.source)
    self.assertIn("/api/knowledge/evidence/", self.source)
    self.assertIn(".evidence-drawer", css)
    self.assertIn(".evidence-drawer::backdrop", css)
    self.assertIn("@media (max-width: 640px)", css)


def test_citations_hide_internal_ids_and_use_one_delegated_handler(self):
    self.assertIn('data-parent-id="${encodeURIComponent', self.source)
    self.assertNotIn('chunk: ${s.chunkId}', self.source)
    self.assertEqual(self.source.count("initCitationClickHandlers(chatMessages)"), 1)
    self.assertNotIn("initCitationClickHandlers(contentDiv)", self.source)
```

- [ ] **Step 2: 运行测试并确认失败**

```powershell
E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_chat_live_status.py -v
```

Expected: FAIL，提示抽屉 DOM、函数和样式不存在。

- [ ] **Step 3: 在聊天模板增加一个抽屉**

在 `templates/chat.html` 的聊天 section 后、脚本前增加一个原生 `dialog`。原生元素负责焦点隔离和 `Esc` 关闭，避免手写一套模态框状态：

```html
<dialog
  class="evidence-drawer"
  id="evidenceDrawer"
  aria-labelledby="evidenceDrawerTitle"
>
  <header class="evidence-drawer-header">
    <div>
      <p class="evidence-drawer-eyebrow">引用证据</p>
      <h3 id="evidenceDrawerTitle">证据详情</h3>
      <p id="evidenceDrawerMeta" class="evidence-drawer-meta"></p>
    </div>
    <button class="btn btn-icon" id="evidenceDrawerClose" type="button" aria-label="关闭证据详情" autofocus>
      <svg class="icon icon-sm" aria-hidden="true"><use href="/assets/icons.svg#x"/></svg>
    </button>
  </header>
  <div class="evidence-drawer-body" id="evidenceDrawerBody" aria-busy="false">
    <p class="evidence-drawer-status" id="evidenceDrawerStatus" role="status" aria-live="polite"></p>
    <article class="evidence-drawer-content" id="evidenceDrawerContent"></article>
  </div>
  <footer class="evidence-drawer-footer">
    <button class="btn btn-primary btn-sm" id="evidenceDrawerDownload" type="button" hidden>查看原文件</button>
  </footer>
</dialog>
```

把资源版本改为：

```html
<link rel="stylesheet" href="css/style_v2.css?v=20260812-1">
<script src="js/chat.js?v=20260812-1"></script>
```

- [ ] **Step 4: 让来源和组内引用携带父块信息**

在 `js/chat.js` 增加轻量格式化函数：

```javascript
function citationDataAttributes(source) {
  return `data-parent-id="${encodeURIComponent(source.parentId)}" ` +
    `data-filename="${encodeURIComponent(source.filename)}" ` +
    `data-page="${encodeURIComponent(source.page || '')}" ` +
    `data-score="${encodeURIComponent(source.score || '')}"`;
}

function formatCitationScore(value) {
  const score = Number(value);
  if (Number.isFinite(score) && score >= 0 && score <= 1) {
    return `${(score * 100).toFixed(1)}%`;
  }
  return value && value !== '-' ? String(value) : '未知';
}

function decorateCitationBody(text, sources, displayIndexes = null) {
  let decorated = text || '';
  decorated = decorated.replace(/\[(\d+)\]/g, (raw, number) => {
    const index = parseInt(number);
    const source = sources.find(item => item.index === index);
    if (!source) return raw;
    const displayIndex = displayIndexes?.get(index) || index;
    return `<button type="button" class="citation-ref" ${citationDataAttributes(source)} ` +
      `aria-label="查看第 ${displayIndex} 条引用证据">[${displayIndex}]</button>`;
  });
  return decorated.replace(/《(.+?)》/g, '<span class="citation-filename">📄 $1</span>');
}
```

单次 RAG 用该函数替换局部 `decorateCitationBody`；多次 RAG 的每个 `group.toolBody` 使用自己的 `group.sources` 调用它，避免不同检索批次的 `[1]` 串到错误来源。

来源条目改为按钮，并隐藏内部 ID：

```javascript
const pageText = source.page ? `第 ${escapeHtml(source.page)} 页 · ` : '';
return `<button type="button" class="citation-source-item" ${citationDataAttributes(source)}>
  <span class="citation-badge">[${displayIndex}]</span>
  <span class="citation-name">📄 ${escapeHtml(source.filename)}</span>
  <span class="citation-meta">${pageText}相关度 ${escapeHtml(formatCitationScore(source.score))}</span>
  <span class="citation-view">查看证据</span>
</button>`;
```

单次和多次来源列表共用同样的显示规则。

- [ ] **Step 5: 收敛事件监听并实现抽屉生命周期**

在 `initChat()` 取得 `chatMessages` 后只调用一次：

```javascript
initCitationClickHandlers(chatMessages);
initEvidenceDrawer();
```

删除 `updateLastMessage()` 和 `renderMessages()` 中重复的 `initCitationClickHandlers(...)`。

把点击委托改为：

```javascript
function initCitationClickHandlers(container) {
  container.addEventListener('click', (event) => {
    const trigger = event.target.closest('.citation-ref, .citation-source-item');
    if (!trigger || !container.contains(trigger) || !trigger.dataset.parentId) return;
    event.preventDefault();
    openEvidenceDrawer({
      parentId: decodeURIComponent(trigger.dataset.parentId),
      filename: decodeURIComponent(trigger.dataset.filename || ''),
      page: decodeURIComponent(trigger.dataset.page || ''),
      score: decodeURIComponent(trigger.dataset.score || ''),
      trigger,
    });
  });
}
```

在 `js/chat.js` 增加抽屉状态与函数：

```javascript
let evidenceRequestSerial = 0;
let evidenceLastTrigger = null;

function initEvidenceDrawer() {
  const drawer = document.getElementById('evidenceDrawer');
  document.getElementById('evidenceDrawerClose')?.addEventListener('click', closeEvidenceDrawer);
  document.getElementById('evidenceDrawerDownload')?.addEventListener('click', downloadEvidenceFile);
  drawer?.addEventListener('click', (event) => {
    const rect = drawer.getBoundingClientRect();
    const inside = event.clientX >= rect.left && event.clientX <= rect.right
      && event.clientY >= rect.top && event.clientY <= rect.bottom;
    if (!inside) drawer.close();
  });
  drawer?.addEventListener('close', () => {
    evidenceRequestSerial += 1;
    evidenceLastTrigger?.focus();
    evidenceLastTrigger = null;
  });
}

async function openEvidenceDrawer({ parentId, filename, page, score, trigger }) {
  const drawer = document.getElementById('evidenceDrawer');
  const body = document.getElementById('evidenceDrawerBody');
  const title = document.getElementById('evidenceDrawerTitle');
  const meta = document.getElementById('evidenceDrawerMeta');
  const status = document.getElementById('evidenceDrawerStatus');
  const content = document.getElementById('evidenceDrawerContent');
  const download = document.getElementById('evidenceDrawerDownload');
  if (!drawer || !parentId) return;

  const requestSerial = ++evidenceRequestSerial;
  evidenceLastTrigger = trigger || null;
  if (!drawer.open) drawer.showModal();
  body.setAttribute('aria-busy', 'true');
  title.textContent = filename || '证据详情';
  meta.textContent = [page ? `第 ${page} 页` : '', score ? `相关度 ${formatCitationScore(score)}` : '']
    .filter(Boolean)
    .join(' · ');
  status.textContent = '正在读取证据...';
  content.textContent = '';
  download.hidden = true;
  download.dataset.url = '';
  download.dataset.filename = '';

  try {
    const response = await apiFetch(
      `${API_BASE}/api/knowledge/evidence/${encodeURIComponent(parentId)}`
    );
    if (!response.ok) throw new Error('evidence unavailable');
    const data = await response.json();
    if (requestSerial !== evidenceRequestSerial) return;

    title.textContent = data.filename || filename || '证据详情';
    meta.textContent = [
      data.page !== null && data.page !== undefined ? `第 ${data.page} 页` : '',
      score ? `相关度 ${formatCitationScore(score)}` : '',
    ].filter(Boolean).join(' · ');
    status.textContent = '';
    content.textContent = data.content || '';
    download.hidden = !data.download_url;
    download.dataset.url = data.download_url || '';
    download.dataset.filename = data.filename || 'document';
  } catch (error) {
    if (requestSerial !== evidenceRequestSerial) return;
    status.textContent = '证据不存在或当前无权访问';
    content.textContent = '';
  } finally {
    if (requestSerial === evidenceRequestSerial) body.setAttribute('aria-busy', 'false');
  }
}

function closeEvidenceDrawer() {
  const drawer = document.getElementById('evidenceDrawer');
  if (drawer?.open) drawer.close();
}

async function downloadEvidenceFile(event) {
  const button = event.currentTarget;
  if (!button.dataset.url) return;
  const response = await apiFetch(button.dataset.url);
  if (!response.ok) return;
  const objectUrl = URL.createObjectURL(await response.blob());
  const link = document.createElement('a');
  link.href = objectUrl;
  link.download = button.dataset.filename || 'document';
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(objectUrl);
}
```

- [ ] **Step 6: 增加抽屉和来源按钮样式**

在 `css/style_v2.css` 的 citation 区域增加：

```css
.citation-ref,
.citation-source-item {
  border: 0;
  font: inherit;
}

.citation-source-item {
  width: 100%;
  color: inherit;
  background: transparent;
  text-align: left;
  cursor: pointer;
}

.citation-view {
  color: var(--accent-primary);
  font-weight: 600;
  flex-shrink: 0;
}

.evidence-drawer {
  position: fixed;
  inset: 0 0 0 auto;
  margin: 0 0 0 auto;
  z-index: 1200;
  width: min(540px, 92vw);
  height: 100dvh;
  max-height: 100dvh;
  padding: 0;
  color: var(--text-primary);
  display: none;
  grid-template-rows: auto minmax(0, 1fr) auto;
  background: var(--bg-primary);
  border: 0;
  border-left: 1px solid var(--border-color);
}

.evidence-drawer[open] {
  display: grid;
  animation: evidence-drawer-in 180ms ease-out;
}

.evidence-drawer::backdrop {
  background: rgba(0, 0, 0, 0.56);
}

@keyframes evidence-drawer-in {
  from { transform: translateX(100%); }
  to { transform: translateX(0); }
}

.evidence-drawer-header,
.evidence-drawer-footer {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 16px;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-color);
}

.evidence-drawer-footer {
  border-top: 1px solid var(--border-color);
  border-bottom: 0;
}

.evidence-drawer-body {
  min-height: 0;
  overflow-y: auto;
  padding: 24px;
}

.evidence-drawer-content {
  color: var(--text-secondary);
  line-height: 1.75;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
}

.evidence-drawer-status,
.evidence-drawer-meta,
.evidence-drawer-eyebrow {
  color: var(--text-tertiary);
}

@media (max-width: 640px) {
  .evidence-drawer {
    width: 100vw;
  }

  .citation-source-item {
    align-items: flex-start;
    flex-wrap: wrap;
  }

  .citation-meta {
    margin-left: 0;
  }
}
```

补充 `prefers-reduced-motion`：关闭抽屉动画。

- [ ] **Step 7: 运行前端契约测试和语法检查**

```powershell
E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_chat_live_status.py -v
node --check js/chat.js
```

Expected: 全部 PASS。

- [ ] **Step 8: 提交前端抽屉**

```powershell
git add templates/chat.html js/chat.js css/style_v2.css tests/test_chat_live_status.py
git commit -m "feat(chat): 支持查看引用证据原文"
```

---

### Task 4: 全链路回归与浏览器验收

**Files:**
- No production file changes expected

- [ ] **Step 1: 运行后端与前端相关测试**

```powershell
E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_knowledge_api_mysql.py tests/test_rag_citation_grounding.py tests/test_chat_live_status.py tests/test_rag_acl_retrieval.py -v
```

Expected: 全部 PASS。

- [ ] **Step 2: 运行基础静态检查**

```powershell
E:\Anaconda\envs\rag_env_backup\python.exe -m compileall api rag schemas repositories
node --check js/chat.js
git diff --check
```

Expected: compileall 和 Node 退出码为 0，`git diff --check` 无输出。

- [ ] **Step 3: 启动本地服务并做桌面/移动端验收**

```powershell
E:\Anaconda\envs\rag_env_backup\python.exe api_server_fastapi.py
```

在浏览器验证：

1. 用 researcher 提问一个能生成 `[1]` 引用的本地知识库问题。
2. 点击回答中的 `[1]`，确认右侧抽屉显示同一来源的完整父块。
3. 点击参考来源条目，确认打开同一父块。
4. 确认来源列表没有显示 `chunk_id`，页码和相关度可读。
5. 确认“查看原文件”通过认证下载。
6. 以 390×844 视口确认抽屉不溢出、正文可滚动、关闭按钮可操作。
7. admin 撤销 researcher 对该文档的权限，再点击历史引用，确认显示统一无权提示。

- [ ] **Step 4: 检查提交和工作区**

```powershell
git log -4 --oneline
git status --short
```

Expected: 有三个功能提交，工作区干净；若浏览器验收发现问题，只针对问题补一个小修复提交。

---

## 实施顺序与提交结果

预期提交：

```text
feat(knowledge): 增加受保护的引用证据接口
feat(rag): 为引用来源补充页码
feat(chat): 支持查看引用证据原文
```

该顺序保证每一步都可独立验证：先建立安全读取边界，再扩展来源协议，最后开放 UI 点击入口。
