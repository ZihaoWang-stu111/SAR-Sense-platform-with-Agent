"""MinerU HTTP 客户端：把 PDF 解析成结构化 Document 列表，供入库链路使用。

设计要点：
- 调 mineru-api 的 /file_parse，同时要 return_md（fallback）和 return_content_list（结构化）。
- content_list 按 item type 重组：table/equation 整块作一个 Document（原子，下游不切），
  text 类攒着交给 _semantic_split。table 的 page_content 必须带 caption——表格 <td> 全是
  数字短词，embedding 质量差，caption 关键词是 BM25+向量的召回命脉。
- content_list 解析失败 → 退回 md_content 单 Document（不打 mineru_structured → 走原链路）。
- 失败只 raise，不吞错。兜底（回退 PyPDFLoader）由 utils/file_handler.pdf_loader 做。
- 配置三级回退（同 rag/reranker.py 模式）：env → rag.yml → 默认值。
"""
import json
import os
import re

import requests
from langchain_core.documents import Document

from utils.logger_handler import logger

# 丢弃的 MinerU item type：页眉页脚页码是噪声，image 本期不处理（return_images=false）。
_DROP_TYPES = {"header", "footer", "page_number", "image"}


def _mineru_url() -> str:
    """mineru-api 地址：env > rag.yml > 默认本地。"""
    from utils.config_handler import rag_conf
    return os.getenv("MINERU_API_URL") or rag_conf.get("mineru_api_url") or "http://127.0.0.1:8000"


def _mineru_enabled() -> bool:
    """是否启用 MinerU：env 显式设置则按 env，否则 rag.yml 默认 true。"""
    val = os.getenv("MINERU_ENABLED")
    if val is None:
        from utils.config_handler import rag_conf
        return bool(rag_conf.get("mineru_enabled", True))
    return val.strip().lower() in ("1", "true", "yes", "on")


def _timeout() -> float:
    """请求超时秒数，大 PDF 可调大。默认 600（CPU pipeline 解析 2.8MB 论文约 150s，留 4x 余量）。"""
    try:
        return float(os.getenv("MINERU_TIMEOUT", "600"))
    except ValueError:
        return 600.0


def _table_html_to_markdown(html: str) -> str:
    """<table>...</table> → markdown 表格。
    简单正则版：忽略 rowspan/colspan（markdown 不支持合并单元格），每 <td>/<th> 一格。
    对比表（纯 <tr><td>）转出来对齐；消融表（有 rowspan）结构乱但数据都在。
    入库存 markdown 而非 HTML：rerank/embedding 对 markdown 打分远高于 HTML（实测 0.57 vs 0.02），
    表格自然进 final，不需要 boost。"""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    if not rows:
        return html
    md_rows = []
    for row in rows:
        cells = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        cells = [re.sub(r"<[^>]+>", "", c).strip() for c in cells]
        if cells:
            md_rows.append("| " + " | ".join(cells) + " |")
    if not md_rows:
        return html
    ncols = md_rows[0].count("|") - 1
    if ncols <= 0:
        return html
    sep = "| " + " | ".join(["---"] * ncols) + " |"
    return md_rows[0] + "\n" + sep + "\n" + "\n".join(md_rows[1:])


def _compose_table(item: dict, file_path: str) -> str:
    """table item → 算法名 + caption + markdown表格 + footnote。

    算法名取文件名 stem（去扩展名），拼在最前。消融表/对比表的 caption 常不含算法名
    （如 "RESULTS OF ABLATION EXPERIMENTS"，用 Model A/B/C 而非 SFQ-Det），
    导致 query "SFQ-Det 的消融表" 匹配不上表格内容 → rerank 低分被挤出 final。
    拼上 stem 后表格自带算法名，rerank/embedding 能命中 query 里的算法名。
    caption 必带（召回命脉），stem 是补充。
    """
    parts = []
    stem = os.path.splitext(os.path.basename(file_path))[0]
    if stem:
        parts.append(stem)
    cap = item.get("table_caption") or []
    if cap:
        parts.append(" ".join(str(c).strip() for c in cap if str(c).strip()))
    body = item.get("table_body") or ""
    if body:
        parts.append(_table_html_to_markdown(body))   # HTML → markdown，rerank/embedding 打分高 25 倍
    fn = item.get("table_footnote") or []
    if fn:
        parts.append(" ".join(str(f).strip() for f in fn if str(f).strip()))
    return "\n\n".join(parts)


def _docs_from_content_list(content_list: list, file_path: str) -> list[Document]:
    """content_list（已解析的 list）→ 结构化 Document 列表。"""
    docs = []
    for item in content_list:
        if not isinstance(item, dict):
            continue
        mtype = item.get("type", "text")
        if mtype in _DROP_TYPES:
            continue
        page = item.get("page_idx")

        if mtype == "table":
            docs.append(Document(
                page_content=_compose_table(item, file_path),
                metadata={
                    "mineru_type": "table",
                    "page": page,
                    "table_caption": item.get("table_caption") or [],
                    "table_footnote": item.get("table_footnote") or [],
                    "mineru_structured": True,
                    "source": file_path,
                },
            ))
        elif mtype == "equation":
            text = (item.get("text") or "").strip()
            if not text:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "mineru_type": "equation",
                    "page": page,
                    "mineru_structured": True,
                    "source": file_path,
                },
            ))
        else:  # text / page_footnote / code / list → 当正文
            text = (item.get("text") or "").strip()
            if not text:
                continue
            meta = {
                "mineru_type": "text",
                "page": page,
                "mineru_structured": True,
                "source": file_path,
            }
            if item.get("text_level") is not None:
                meta["text_level"] = item["text_level"]
            docs.append(Document(page_content=text, metadata=meta))
    return docs


def parse_pdf_to_documents(file_path: str) -> list[Document]:
    """POST PDF 给 MinerU /file_parse，返回结构化 list[Document]（table/equation 整块，text 段落）。

    content_list 解析失败 → 退回 md_content 单 Document（走原链路）。
    任何失败都 raise，由调用方兜底。返回值保证非空。
    """
    url = f"{_mineru_url()}/file_parse"
    with open(file_path, "rb") as f:
        files = {"files": (os.path.basename(file_path), f, "application/pdf")}
        # backend=pipeline 强制 CPU（默认 hybrid-engine 需 GPU）。
        # return_content_list=true 拿结构化块；return_md=true 留作 fallback。
        data = {
            "backend": "pipeline",
            "parse_method": "auto",
            "return_md": "true",
            "return_content_list": "true",
            "return_images": "false",
            "response_format_zip": "false",
        }
        resp = requests.post(url, files=files, data=data, timeout=_timeout())

    if resp.status_code != 200:
        raise RuntimeError(f"MinerU HTTP {resp.status_code}: {resp.text[:300]}")

    body = resp.json()
    results = body.get("results") or {}
    if not results:
        raise RuntimeError("MinerU 返回 results 为空")

    stem = os.path.splitext(os.path.basename(file_path))[0]
    entry = results.get(stem) or next(iter(results.values()))
    if not isinstance(entry, dict):
        raise RuntimeError("MinerU results entry 非对象")

    # 优先 content_list 结构化；失败退回 md_content 单 Document（不打 mineru_structured → 原链路）
    cl_raw = entry.get("content_list")
    if cl_raw:
        try:
            cl = json.loads(cl_raw) if isinstance(cl_raw, str) else cl_raw
            docs = _docs_from_content_list(cl, file_path)
            if docs:
                logger.info(
                    f"MinerU 结构化解析成功: {file_path} ({len(docs)} 个结构块, "
                    f"table×{sum(1 for d in docs if d.metadata.get('mineru_type')=='table')}, "
                    f"eq×{sum(1 for d in docs if d.metadata.get('mineru_type')=='equation')})"
                )
                return docs
            logger.warning(f"MinerU content_list 重组为空，回退 md_content: {file_path}")
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"MinerU content_list 解析失败，回退 md_content: {file_path} - {e}")

    md = entry.get("md_content") or ""
    if not md.strip():
        raise RuntimeError("MinerU 返回 md_content 为空且 content_list 不可用")
    logger.info(f"MinerU 解析成功(fallback md): {file_path} (markdown {len(md)} 字符)")
    # 不打 mineru_structured → 下游走原 _semantic_split 链路，和 PyPDFLoader 一致。
    return [Document(page_content=md, metadata={"source": file_path})]


if __name__ == "__main__":
    # ponytail: 自检——拿一篇真 PDF 跑通，验证结构化分流不出错。需要 mineru 容器在跑。
    import sys
    if len(sys.argv) < 2:
        print("用法: python -m utils.mineru_client <pdf路径>")
        sys.exit(1)
    docs = parse_pdf_to_documents(sys.argv[1])
    by_type = {}
    for d in docs:
        t = d.metadata.get("mineru_type", "?")
        by_type[t] = by_type.get(t, 0) + 1
    print(f"总块数 {len(docs)}，按类型: {by_type}")
    # 抽样：第一个 table 的前 200 字符
    for d in docs:
        if d.metadata.get("mineru_type") == "table":
            print(f"\n首个 table (page={d.metadata.get('page')}, id待 vector_store 生成):")
            print(d.page_content[:200])
            break
