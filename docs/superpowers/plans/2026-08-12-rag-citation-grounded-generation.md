# RAG 引用约束式生成 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** 在不增加模型调用的前提下，让 RAG 只输出有合法证据编号支持的结论，并允许部分回答或明确拒答。

**Architecture:** 保留现有混合召回、BGE 重排、父块回表和来源列表。RagSummarizeService 将父块包装成明确的 [证据n] 单元，现有生成模型输出内部引用 [[EVIDENCE:n]]；一个纯函数负责校验编号、处理 [[INSUFFICIENT]] 并转换为用户可读的 [n]。

**Tech Stack:** Python 3.10、LangChain、pytest/unittest、现有 PromptTemplate 与 RAG 服务。

---

## 文件结构

- 新增 tests/test_rag_citation_grounding.py：引用校验与上下文标签回归测试。
- 修改 rag/rag_service.py：证据标签组装和确定性引用校验。
- 修改 prompts/rag_summarize.txt：直接证据、部分回答、证据不足及引用协议。
- 修改 tests/test_prompt_contracts.py：锁定 RAG 与主 Agent 的提示词契约。
- 修改 prompts/main_prompt.txt：禁止主 Agent 补写 RAG 未支持的事实。

现有前端、数据库、Chroma、Retriever 和 Reranker 均不属于本计划。

---

### Task 1: 引用校验与证据上下文

**Files:**
- Create: tests/test_rag_citation_grounding.py
- Modify: rag/rag_service.py

- [ ] **Step 1: 编写失败测试**

创建 tests/test_rag_citation_grounding.py：

    from langchain_core.documents import Document

    from rag.rag_service import (
        GROUNDING_FALLBACK,
        RagSummarizeService,
        render_grounded_answer,
    )


    class FakeChain:
        def __init__(self, answer):
            self.answer = answer
            self.payload = None

        def invoke(self, payload):
            self.payload = payload
            return self.answer


    def test_valid_evidence_markers_are_rendered():
        answer = render_grounded_answer(
            "SSDD 上提升 2.5 个百分点 [[EVIDENCE:1]]。",
            source_count=2,
        )
        assert answer == "SSDD 上提升 2.5 个百分点 [1]。"


    def test_missing_or_invalid_evidence_fails_closed():
        assert render_grounded_answer("模型给出一个结论。", 2) == GROUNDING_FALLBACK
        assert render_grounded_answer(
            "错误结论 [[EVIDENCE:3]]。", 2
        ) == GROUNDING_FALLBACK
        assert render_grounded_answer(
            "错误格式 [[EVIDENCE:abc]]。", 2
        ) == GROUNDING_FALLBACK


    def test_insufficient_marker_returns_explanation_without_citation():
        answer = render_grounded_answer(
            "[[INSUFFICIENT]] 当前资料没有明确给出 HRSID 对比数据。",
            source_count=2,
        )
        assert answer == "当前资料没有明确给出 HRSID 对比数据。"


    def test_rag_context_uses_explicit_evidence_blocks():
        docs = [
            Document(
                page_content="SFQ-Det 的 mAP@0.5 为 98.2%。",
                metadata={
                    "filename": "SFQ-Det.pdf",
                    "chunk_id": "parent-3",
                    "parent_index": 3,
                    "match_child_id": "child-8",
                    "page": 3,
                    "rerank_score": 0.95,
                },
            )
        ]
        service = object.__new__(RagSummarizeService)
        service.retriever_docs = lambda query, allowed_doc_ids=None: docs
        service.chain = FakeChain("提升结论 [[EVIDENCE:1]]。")

        result = service.rag_summarize("提升多少？")

        context = service.chain.payload["context"]
        assert "[证据1]" in context
        assert "正文：\nSFQ-Det 的 mAP@0.5 为 98.2%。" in context
        assert "来源：SFQ-Det.pdf" in context
        assert "[[EVIDENCE:" not in result
        assert "提升结论 [1]。" in result
        assert "[1] SFQ-Det.pdf | chunk_id=parent-3 | score=0.9500" in result

- [ ] **Step 2: 运行测试并确认失败**

Run:

    E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_rag_citation_grounding.py -q

Expected: FAIL，提示 render_grounded_answer、GROUNDING_FALLBACK 尚不存在，或上下文缺少 [证据1]。

- [ ] **Step 3: 实现最小引用校验**

在 rag/rag_service.py 增加：

    import re

    GROUNDING_FALLBACK = "当前资料不足以直接支持可靠结论，无法基于知识库回答。"
    _INSUFFICIENT_MARKER = "[[INSUFFICIENT]]"
    _EVIDENCE_TOKEN_RE = re.compile(r"\[\[EVIDENCE:([^\]]+)\]\]")


    def render_grounded_answer(answer: str, source_count: int) -> str:
        """校验证据编号并把内部引用转换成用户可读编号。"""
        answer = (answer or "").strip()
        if answer.startswith(_INSUFFICIENT_MARKER):
            explanation = answer[len(_INSUFFICIENT_MARKER):].strip()
            return explanation or GROUNDING_FALLBACK

        raw_ids = _EVIDENCE_TOKEN_RE.findall(answer)
        if not raw_ids or any(not value.isdigit() for value in raw_ids):
            return GROUNDING_FALLBACK

        evidence_ids = [int(value) for value in raw_ids]
        if any(value < 1 or value > source_count for value in evidence_ids):
            return GROUNDING_FALLBACK

        rendered = _EVIDENCE_TOKEN_RE.sub(
            lambda match: f"[{int(match.group(1))}]",
            answer,
        )
        if "[[EVIDENCE:" in rendered:
            return GROUNDING_FALLBACK
        return rendered

- [ ] **Step 4: 调整 rag_summarize 上下文**

把现有 context += 组装改为：

    context_blocks = []
    sources = []
    for i, doc in enumerate(docs, 1):
        filename = doc.metadata.get("filename", "未知文档")
        chunk_id = doc.metadata.get("chunk_id", "-")
        chunk_index = doc.metadata.get(
            "parent_index", doc.metadata.get("chunk_index", "-")
        )
        match_child = doc.metadata.get("match_child_id", "-")
        page = doc.metadata.get("page", "-")
        score = doc.metadata.get("rerank_score", "-")
        if isinstance(score, float):
            score = f"{score:.4f}"

        source_line = (
            f"来源：{filename} | chunk_id={chunk_id} | "
            f"parent_index={chunk_index} | match_child={match_child} "
            f"| page={page} | score={score}"
        )
        table_id = doc.metadata.get("table_id", "")
        if table_id:
            source_line += f" | {table_id}"

        context_blocks.append(
            f"[证据{i}]\n正文：\n{doc.page_content}\n{source_line}"
        )
        sources.append(
            f"[{i}] {filename} | chunk_id={chunk_id} | score={score}"
        )

    context = "\n\n".join(context_blocks)

保留原有“截掉模型自行生成的参考来源列表”逻辑，然后执行：

    answer = render_grounded_answer(answer, source_count=len(docs))
    return f"{answer}\n\n参考来源：\n" + "\n".join(sources)

- [ ] **Step 5: 运行测试并确认通过**

Run:

    E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_rag_citation_grounding.py -q

Expected: 4 passed。

- [ ] **Step 6: 提交**

    git add rag/rag_service.py tests/test_rag_citation_grounding.py
    git commit -m "feat(rag): 校验回答证据引用"

---

### Task 2: RAG Prompt 与主 Agent 约束

**Files:**
- Modify: tests/test_prompt_contracts.py
- Modify: prompts/rag_summarize.txt
- Modify: prompts/main_prompt.txt

- [ ] **Step 1: 编写提示词契约失败测试**

在 tests/test_prompt_contracts.py 增加：

    class RagPromptContractTests(unittest.TestCase):
        def test_rag_prompt_defines_grounding_contract(self):
            prompt = _prompt("rag_summarize.txt")
            for requirement in (
                "直接证据",
                "背景信息",
                "信息缺失",
                "部分回答",
                "[[EVIDENCE:n]]",
                "[[INSUFFICIENT]]",
            ):
                self.assertIn(requirement, prompt)

        def test_main_agent_preserves_rag_limits_and_citations(self):
            prompt = _prompt("main_prompt.txt")
            self.assertRegex(
                prompt,
                r"rag_summarize[^\n]*信息缺失[^\n]*不得.*补全",
            )
            self.assertRegex(
                prompt,
                r"rag_summarize[^\n]*证据引用[^\n]*不得新增.*事实",
            )

- [ ] **Step 2: 运行测试并确认失败**

Run:

    E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_prompt_contracts.py::RagPromptContractTests -q

Expected: FAIL，新规则尚未出现在 Prompt 中。

- [ ] **Step 3: 修改 prompts/rag_summarize.txt**

用以下规则替换原来只要求《文件名》的引用规则，并保留表格完整性要求：

    8. 证据判断：先区分直接证据、背景信息和信息缺失。只有明确包含所需结论、数值、关系或表格行的内容才是直接证据；主题相关但不能直接回答问题的内容只能作为背景信息，不得据此推断答案。
    9. 部分回答：资料只能回答部分问题时，只回答有直接证据的部分，并明确说明其余信息缺失，不得把整道题直接判为无答案。
    10. 强制引用：每个关键事实、数值、比较关系和原因结论后必须写 [[EVIDENCE:n]]，其中 n 必须对应输入中的 [证据n]。不得引用未直接支持该结论的证据。
    11. 证据不足：完全没有直接证据时，以 [[INSUFFICIENT]] 开头，随后说明当前资料没有明确说明什么；不得输出任何未经直接证据支持的结论。
    12. 不要自行输出参考来源列表，系统会根据合法证据编号自动追加来源。

- [ ] **Step 4: 修改 prompts/main_prompt.txt**

在“工具执行规则”中加入：

    7. rag_summarize 已明确指出信息缺失或只能部分回答时，必须保留该边界，不得使用模型自身知识补全；用户限制只能使用本地知识库时不得联网补充。
    8. 基于 rag_summarize 回答时必须保留其证据引用，不得新增工具结果中没有的事实。

- [ ] **Step 5: 运行测试并确认通过**

Run:

    E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_prompt_contracts.py -q

Expected: 全部通过。

- [ ] **Step 6: 提交**

    git add prompts/rag_summarize.txt prompts/main_prompt.txt tests/test_prompt_contracts.py
    git commit -m "refactor(prompt): 约束RAG结论引用证据"

---

### Task 3: 回归验证

**Files:**
- Verify only; no production changes expected.

- [ ] **Step 1: 运行 RAG 核心测试**

    E:\Anaconda\envs\rag_env_backup\python.exe -m pytest tests/test_rag_citation_grounding.py tests/test_prompt_contracts.py tests/test_parent_child.py tests/test_hybrid_retriever.py -q

Expected: 全部通过。

- [ ] **Step 2: 运行语法和 Diff 检查**

    E:\Anaconda\envs\rag_env_backup\python.exe -m compileall rag
    git diff --check

Expected: 均返回退出码 0。

- [ ] **Step 3: 手工验证**

问题：

    SFQ-Det 在 SSDD 和 HRSID 上分别比 YOLOv8 提升多少 mAP，并解释原因？只能使用本地知识库。

Expected:

- SSDD 数值后带合法 [n]。
- HRSID 缺少数据时明确说明无法计算。
- 原因结论只在有直接证据时输出并带 [n]。
- 不调用 web_search。
- 页面不出现内部控制标记。

- [ ] **Step 4: 检查提交边界**

    git status --short
    git log --oneline -n 4

Expected: 本计划产生两个独立代码提交；此前未提交的前端折叠修复仍保持独立。
