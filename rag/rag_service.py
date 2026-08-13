import os
import re
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from rag.vector_store import get_vector_store_service
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from rag.reranker import BGERerankerService
from rag.parent_child_retriever import ParentChildResolver
from utils.config_handler import chroma_conf
from utils.logger_handler import logger


GROUNDING_FALLBACK = "当前资料不足以直接支持可靠结论，无法基于知识库回答。"
CITATION_VALIDATION_FALLBACK = "已检索到相关资料，但回答引用校验未通过，请稍后重试。"
_EVIDENCE_PATTERN = re.compile(r"(?<!\[)\[\[EVIDENCE:([0-9]+)\]\](?!\])")
_REPAIRABLE_EVIDENCE_PATTERN = re.compile(
    r"[\[【]+\s*EVIDENCE\s*[:：]\s*(?P<index>[0-9]+)\s*[\]】]+",
    re.IGNORECASE,
)
_SUSPICIOUS_EVIDENCE_PATTERN = re.compile(
    r"(?:[\[【]+\s*EVIDENCE|EVIDENCE\s*[:：])",
    re.IGNORECASE,
)
_INSUFFICIENT_MARKER = "[[INSUFFICIENT]]"


def _normalize_evidence_markers(answer: str) -> str:
    """修正常见的括号、空格、大小写和全角冒号差异。"""
    def replace(match):
        index = match.group("index").lstrip("0") or "0"
        return f"[[EVIDENCE:{index}]]"

    return _REPAIRABLE_EVIDENCE_PATTERN.sub(replace, answer)


def _citation_validation_failed(reason: str) -> str:
    logger.warning(f"RAG 回答引用校验失败: {reason}")
    return CITATION_VALIDATION_FALLBACK


def render_grounded_answer(answer, source_count):
    """宽容修复引用格式，严格校验编号，再转换为前端引用。"""
    if not isinstance(answer, str) or type(source_count) is not int or source_count < 0:
        return _citation_validation_failed("输入类型或来源数量无效")

    normalized_answer = _normalize_evidence_markers(answer.strip())

    # 证据不足标记只用于模型与后端通信，不展示给用户。
    is_insufficient = normalized_answer.startswith(_INSUFFICIENT_MARKER)
    if is_insufficient:
        normalized_answer = normalized_answer[len(_INSUFFICIENT_MARKER):].strip()
        if not normalized_answer:
            return GROUNDING_FALLBACK

    citation_matches = list(_EVIDENCE_PATTERN.finditer(normalized_answer))
    if not is_insufficient and not citation_matches:
        return _citation_validation_failed("正常回答没有证据引用")

    # 移除合法引用后若还有 EVIDENCE 字样，说明存在无法安全修复的伪引用。
    answer_without_citations = _EVIDENCE_PATTERN.sub("", normalized_answer)
    if _SUSPICIOUS_EVIDENCE_PATTERN.search(answer_without_citations):
        return _citation_validation_failed("存在无法修复的证据标记")

    try:
        citation_ids = [int(match.group(1)) for match in citation_matches]
    except ValueError:
        return _citation_validation_failed("证据编号不是有效整数")

    if any(not 1 <= citation_id <= source_count for citation_id in citation_ids):
        return _citation_validation_failed("证据编号超出本次来源范围")

    return _EVIDENCE_PATTERN.sub(
        lambda match: f"[{match.group(1)}]",
        normalized_answer,
    )


class RagSummarizeService:
    def __init__(self):
        self.vector_store = get_vector_store_service()

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model

        self.reranker = BGERerankerService()
        self.chain = self._init_chain()
        self.final_k = chroma_conf.get("retrieve_k_parents", 3)
        self.parent_resolver = None
        if self.vector_store.parent_child_enabled and self.vector_store.parent_docstore:
            self.parent_resolver = ParentChildResolver(
                parent_docstore=self.vector_store.parent_docstore,
                top_k_parents=self.final_k,
            )

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    @staticmethod
    def _filter_docs(docs, allowed_doc_ids=None):
        if allowed_doc_ids is None:
            return docs
        allowed = set(allowed_doc_ids)
        if not allowed:
            return []
        return [doc for doc in docs if (doc.metadata or {}).get("doc_id") in allowed]

    def retriever_docs(self, query, allowed_doc_ids=None):
        # 子块召回 → 先在子块上重排(只打分不截断) → 再回表聚合成父块
        # rerank 提前到子块层：CrossEncoder 对短文本判分更准，并让"选哪些父块"由相关性决定
        # 表格入库时已存 markdown（_compose_table 转），rerank 天然高分，无需 boost / doc 推断
        if allowed_doc_ids is not None and not allowed_doc_ids:
            return []

        candidate_docs = self.vector_store.retrieve(query, allowed_doc_ids=allowed_doc_ids)

        scored_children = self._filter_docs(self.reranker.rerank(query, candidate_docs), allowed_doc_ids)
        if not scored_children:
            return []

        if self.parent_resolver:
            # resolve 按 rerank 顺序去重，父块继承子块相关性排序，并截断到 final_k
            return self.parent_resolver.resolve(scored_children, allowed_doc_ids=allowed_doc_ids)

        return self._filter_docs(scored_children[:self.final_k], allowed_doc_ids)


    def rag_summarize(self, query, allowed_doc_ids=None):
        docs = self.retriever_docs(query, allowed_doc_ids=allowed_doc_ids)

        if not docs:
            return "知识库中未检索到与该问题相关的可靠资料，无法基于知识库回答。"

        context = ""
        sources = []
        for i, doc in enumerate(docs, 1):
            filename = doc.metadata.get('filename', '未知文档')
            chunk_id = doc.metadata.get('chunk_id', '-')
            chunk_index = doc.metadata.get('parent_index', doc.metadata.get('chunk_index', '-'))
            match_child = doc.metadata.get('match_child_id', '-')
            page = doc.metadata.get('page', '-')
            score = doc.metadata.get('rerank_score', '-')
            if isinstance(score, float):
                score = f"{score:.4f}"
            context += f"[证据{i}]\n正文：\n{doc.page_content}\n"
            ctx_line = (
                f"来源：{filename} | chunk_id={chunk_id} | parent_index={chunk_index} "
                f"| match_child={match_child} | page={page} | score={score}"
            )
            table_id = doc.metadata.get('table_id', '')
            if table_id:
                ctx_line += f" | {table_id}"
            context += ctx_line + "\n\n"
            source = f"[{i}] {filename} | chunk_id={chunk_id}"
            if page not in (None, "", "-"):
                source += f" | page={page}"
            sources.append(f"{source} | score={score}")

        answer = self.chain.invoke(
            {"input": query,
             "context": context}
        )
        if isinstance(answer, str):
            for marker in ("参考来源：", "参考来源:"):
                if marker in answer:
                    answer = answer.split(marker, 1)[0].strip()
                    break
        answer = render_grounded_answer(answer, len(sources))

        return f"{answer}\n\n参考来源：\n" + "\n".join(sources)


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.retriever_docs("SA-WDP 为什么能处理斑点噪声和尺度变化？"))
