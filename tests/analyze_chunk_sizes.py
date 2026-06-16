"""Analyze knowledge base file types and chunk size fit."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.path_tool import get_abs_path
from utils.config_handler import chroma_conf
from utils.file_handler import text_loader, pdf_loader, listdir_with_allowed_type
from langchain_text_splitters import RecursiveCharacterTextSplitter


def analyze():
    data_path = get_abs_path(chroma_conf["data_path"])
    files = listdir_with_allowed_type(data_path, tuple(chroma_conf["allow_knowledge_file_type"]))

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chroma_conf.get("child_chunk_size", 120),
        chunk_overlap=chroma_conf.get("child_chunk_overlap", 30),
        separators=chroma_conf["separators"],
        length_function=len,
    )

    print("=" * 72)
    print("知识库文件分析")
    print("=" * 72)
    print(f"当前子块: size={chroma_conf.get('child_chunk_size')} overlap={chroma_conf.get('child_chunk_overlap')}")
    print(f"当前父块(语义): min={chroma_conf.get('semantic_min_chunk_size')} max={chroma_conf.get('semantic_max_chunk_size')}")
    print()

    for path in sorted(files):
        name = os.path.basename(path)
        ext = name.rsplit(".", 1)[-1]
        size_kb = os.path.getsize(path) / 1024

        if ext == "txt":
            docs = text_loader(path)
        else:
            docs = pdf_loader(path)

        total_chars = sum(len(d.page_content) for d in docs)
        pages = len(docs)

        # simulate child split on full text (fixed parent proxy)
        all_text = "\n".join(d.page_content for d in docs)
        child_est = len(child_splitter.split_text(all_text))

        avg_page = total_chars // max(pages, 1)
        print(f"[{ext.upper():3}] {name}")
        print(f"      文件大小: {size_kb:.1f} KB | 页/段: {pages} | 总字符: {total_chars:,} | 均页: {avg_page}")
        print(f"      若全用 child={chroma_conf.get('child_chunk_size')} 固定切 → 约 {child_est} 子块")
        if total_chars < 8000:
            preview = all_text[:200].replace("\n", " ")
            print(f"      内容特征: 结构化中文知识文档")
            print(f"      预览: {preview}...")
        else:
            sample = all_text[:300].replace("\n", " ")
            has_en = sum(1 for c in all_text[:2000] if c.isascii() and c.isalpha()) > 200
            print(f"      内容特征: {'中英混合学术论文' if has_en else '长文档'}")
            print(f"      预览: {sample}...")
        print()


if __name__ == "__main__":
    analyze()
