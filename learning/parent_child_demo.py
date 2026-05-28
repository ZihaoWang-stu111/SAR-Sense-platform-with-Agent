"""
父子块检索（Parent-Child Chunking）学习示例

核心思想：
  检索时用小块（child）精准匹配
  返回给 LLM 时用大块（parent）保证上下文完整

类比：
  你去图书馆找一本厚书里的某个知识点
  - child = 目录/索引（帮你快速定位到第几章）
  - parent = 整个章节（给你完整上下文）
"""

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# 第一步：构建父子块索引
# ============================================================

def build_parent_child_index(documents: list[Document]) -> list[dict]:
    """
    把原始文档切成父子块，建立映射关系

    返回结构：
    [
        {
            "parent": Document(大块，512字),
            "children": [Document(小块，128字), ...]
        },
        ...
    ]
    """
    # 父块分割器：大窗口，保证语义完整
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", ".", " "]
    )

    # 子块分割器：小窗口，保证检索精准
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=128,
        chunk_overlap=20,
        separators=["\n\n", "\n", "。", ".", " "]
    )

    index = []

    for doc in documents:
        # 先切父块
        parents = parent_splitter.split_documents([doc])

        for parent in parents:
            # 每个父块再切子块
            children = child_splitter.split_documents([parent])

            index.append({
                "parent": parent,          # 大块，存起来等返回时用
                "children": children,       # 小块，建索引用
            })

    return index


# ============================================================
# 第二步：用子块建向量索引，但记住它属于哪个父块
# ============================================================

def build_searchable_children(index: list[dict]) -> list[dict]:
    """
    把所有子块摊平，每个子块带上它所属父块的 ID

    返回：
    [
        {"child": Document, "parent_id": 0},
        {"child": Document, "parent_id": 0},
        {"child": Document, "parent_id": 1},
        ...
    ]
    """
    searchable = []
    for parent_id, entry in enumerate(index):
        for child in entry["children"]:
            searchable.append({
                "child": child,
                "parent_id": parent_id,
            })
    return searchable


# ============================================================
# 第三步：检索时，用子块匹配，返回父块
# ============================================================

def search(query: str, searchable_children: list[dict], index: list[dict],
           embed_model, k: int = 3) -> list[Document]:
    """
    父子块检索的核心流程：

    1. 用户提问 → 和所有 child 做向量相似度搜索
    2. 找到最相关的 child → 通过 parent_id 找到对应的 parent
    3. 返回 parent 给 LLM（去重，因为多个 child 可能属于同一个 parent）
    """
    # 1. 对 query 做 embedding
    query_embedding = embed_model.embed_query(query)

    # 2. 和所有 child 做相似度计算（这里简化为暴力搜索，实际用向量数据库）
    scored = []
    for item in searchable_children:
        child_embedding = embed_model.embed_query(item["child"].page_content)
        sim = cosine_similarity(query_embedding, child_embedding)
        scored.append({
            "parent_id": item["parent_id"],
            "score": sim,
        })

    # 3. 按相似度排序，取 top-k 个 child
    scored.sort(key=lambda x: x["score"], reverse=True)

    # 4. 通过 child 找到 parent，去重
    seen_parents = set()
    result_parents = []
    for item in scored:
        pid = item["parent_id"]
        if pid not in seen_parents:
            seen_parents.add(pid)
            result_parents.append(index[pid]["parent"])
        if len(result_parents) >= k:
            break

    return result_parents


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


# ============================================================
# 对比演示
# ============================================================

def demo_comparison():
    """
    对比：普通分块 vs 父子块

    假设用户问："舰船检测的精度是多少？"

    普通分块（512字的大块）：
      - 检索精度低，可能匹配到不相关的段落
      - 但返回的上下文完整

    普通分块（128字的小块）：
      - 检索精度高，能精准匹配到"精度"相关内容
      - 但返回的上下文碎片化，LLM 可能缺乏背景信息

    父子块：
      - 用 128 字的小块检索 → 精准
      - 返回 512 字的大块 → 上下文完整
      - 两全其美
    """

    # 模拟一个长文档
    sample_text = """
    第一章 SAR舰船检测概述

    合成孔径雷达（SAR）是一种主动式微波遥感技术，能够全天候、全天时获取地表信息。
    SAR图像在海洋监测领域具有重要应用价值，特别是在舰船检测方面。

    第二章 检测算法

    本系统采用基于深度学习的目标检测算法MBE-Net。
    MBE-Net在SAR舰船检测任务上取得了优异的性能。
    在公开数据集上的测试结果表明，mAP达到92.3%，召回率达到95.1%。
    相比传统CFAR算法，MBE-Net在复杂海况下的误检率降低了40%。

    第三章 系统架构

    系统采用前后端分离架构，前端使用Streamlit和FastAPI双模式。
    后端集成了RAG检索增强生成能力，支持知识库问答。
    智能体基于LangChain ReAct框架实现，支持多轮对话和工具调用。
    """

    doc = Document(page_content=sample_text, metadata={"source": "SAR技术文档"})

    # 构建父子块索引
    index = build_parent_child_index([doc])

    print("=" * 60)
    print("父子块索引结构")
    print("=" * 60)
    print(f"原始文档被切成了 {len(index)} 个父块\n")

    for i, entry in enumerate(index):
        parent_text = entry["parent"].page_content[:80].replace("\n", " ")
        child_count = len(entry["children"])
        print(f"父块 {i}: [{len(entry['parent'].page_content)}字] {parent_text}...")
        print(f"  └── 包含 {child_count} 个子块:")
        for j, child in enumerate(entry["children"]):
            child_text = child.page_content[:60].replace("\n", " ")
            print(f"      子块 {j}: [{len(child.page_content)}字] {child_text}")
        print()

    # 展示检索流程
    print("=" * 60)
    print("检索流程演示")
    print("=" * 60)
    print()
    print("用户问: '检测精度是多少？'")
    print()
    print("第1步: 用子块（128字）做向量检索")
    print("  → 子块命中: 'mAP达到92.3%，召回率达到95.1%...'")
    print("  → 这个子块属于父块1（第二章 检测算法）")
    print()
    print("第2步: 返回父块（512字）给 LLM")
    print("  → LLM 收到完整的'第二章 检测算法'上下文")
    print("  → 不仅有精度数据，还有算法名称、对比结果等背景信息")
    print()
    print("对比普通方案:")
    print("  如果用128字小块直接返回 → LLM 只看到精度数字，不知道是什么算法")
    print("  如果用512字大块检索 → 可能匹配到其他章节，不精准")


if __name__ == "__main__":
    demo_comparison()
