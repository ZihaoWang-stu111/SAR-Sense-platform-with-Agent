"""
工程版父子块检索 demo。

运行方式：
    conda run -n rag_env_backup python learning/engineering_parent_child_retrieval_demo.py

这个文件故意做成单文件、独立存储，不接入项目主 RAG 链路。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


RUNTIME_DIR_NAME = "_parent_child_demo_runtime"
COLLECTION_CHILDREN = "engineering_parent_child_children"
COLLECTION_PARENT_DEBUG = "engineering_parent_child_parent_debug"


@dataclass
class SearchResult:
    """最终返回给 LLM 的父块，以及触发它的子块命中证据。"""

    parent_id: str
    parent_text: str
    parent_metadata: dict[str, Any]
    best_child_id: str
    best_child_text: str
    best_child_score: float
    best_child_distance: float


@dataclass
class ParentDirectHit:
    """对比用：父块也直接进向量库时的命中结果。"""

    parent_id: str
    parent_text: str
    score: float
    distance: float


class HashingEmbeddingFunction(EmbeddingFunction):
    """
    一个离线、确定性的 Chroma embedding function。

    工程 demo 的重点是 Chroma 入库、metadata 映射、父块回表，不是 embedding 模型质量。
    真实项目里把它换成 DashScopeEmbeddings / bge / text2vec 等即可。
    """

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def __call__(self, input: Documents) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        tokens = self._tokens(text)
        vector = [0.0] * self.dimensions

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            index = int(digest[:8], 16) % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        normalized = text.lower()
        words = re.findall(r"[a-z0-9.%+-]+", normalized)
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", normalized)
        chinese_bigrams = [
            "".join(chinese_chars[index : index + 2])
            for index in range(max(0, len(chinese_chars) - 1))
        ]
        domain_keywords = {
            "map": 16,
            "召回率": 16,
            "精度": 10,
            "95.1": 10,
            "92.3": 10,
            "mbe-net": 8,
            "误检率": 8,
            "cfar": 6,
            "舰船检测": 3,
            "检测": 2,
        }
        weighted_keywords: list[str] = []
        for keyword, weight in domain_keywords.items():
            if keyword in normalized:
                weighted_keywords.extend([f"kw:{keyword}"] * weight)

        return (words * 3) + weighted_keywords + chinese_bigrams + chinese_chars


class EngineeringParentChildRetriever:
    """演示工程上的 child vector index + parent docstore 检索结构。"""

    def __init__(
        self,
        runtime_dir: Path,
        parent_chunk_size: int = 260,
        parent_chunk_overlap: int = 40,
        child_chunk_size: int = 90,
        child_chunk_overlap: int = 20,
    ):
        self.runtime_dir = runtime_dir
        self.chroma_dir = runtime_dir / "chroma"
        self.parent_docstore_path = runtime_dir / "parent_docstore.json"
        self.embedding_function = HashingEmbeddingFunction()
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_chunk_overlap,
            separators=["\n\n", "\n", "。", "，", ".", " ", ""],
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_chunk_size,
            chunk_overlap=child_chunk_overlap,
            separators=["\n\n", "\n", "。", "，", ".", " ", ""],
        )

    def reset_runtime(self) -> None:
        self._assert_demo_runtime_dir()
        if self.runtime_dir.exists():
            shutil.rmtree(self.runtime_dir)
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

    def build_index(self, documents: list[dict[str, Any]]) -> None:
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        parent_docstore: dict[str, dict[str, Any]] = {}
        child_ids: list[str] = []
        child_texts: list[str] = []
        child_metadatas: list[dict[str, Any]] = []
        parent_ids_for_debug: list[str] = []
        parent_texts_for_debug: list[str] = []
        parent_metadatas_for_debug: list[dict[str, Any]] = []

        for doc_index, raw_doc in enumerate(documents):
            source_doc = Document(
                page_content=raw_doc["text"],
                metadata=self._clean_metadata(raw_doc.get("metadata", {})),
            )
            doc_id = source_doc.metadata.get("doc_id", f"doc_{doc_index}")
            parent_docs = self.parent_splitter.split_documents([source_doc])

            for parent_index, parent_doc in enumerate(parent_docs):
                parent_id = f"{doc_id}:parent:{parent_index:03d}"
                parent_metadata = {
                    **self._clean_metadata(parent_doc.metadata),
                    "parent_id": parent_id,
                    "chunk_type": "parent",
                    "parent_index": parent_index,
                }
                parent_docstore[parent_id] = {
                    "page_content": parent_doc.page_content,
                    "metadata": parent_metadata,
                }
                parent_ids_for_debug.append(parent_id)
                parent_texts_for_debug.append(parent_doc.page_content)
                parent_metadatas_for_debug.append(parent_metadata)

                child_docs = self.child_splitter.split_documents([parent_doc])
                for child_index, child_doc in enumerate(child_docs):
                    child_id = f"{parent_id}:child:{child_index:03d}"
                    child_ids.append(child_id)
                    child_texts.append(child_doc.page_content)
                    child_metadatas.append(
                        {
                            **self._clean_metadata(child_doc.metadata),
                            "doc_id": doc_id,
                            "parent_id": parent_id,
                            "child_id": child_id,
                            "chunk_type": "child",
                            "parent_index": parent_index,
                            "child_index": child_index,
                        }
                    )

        self.parent_docstore_path.write_text(
            json.dumps(parent_docstore, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        client = self._client()
        child_collection = client.get_or_create_collection(
            name=COLLECTION_CHILDREN,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        child_collection.add(
            ids=child_ids,
            documents=child_texts,
            metadatas=child_metadatas,
        )

        parent_debug_collection = client.get_or_create_collection(
            name=COLLECTION_PARENT_DEBUG,
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )
        parent_debug_collection.add(
            ids=parent_ids_for_debug,
            documents=parent_texts_for_debug,
            metadatas=parent_metadatas_for_debug,
        )

    def search(self, query: str, top_k_children: int = 6, top_k_parents: int = 2) -> list[SearchResult]:
        parent_docstore = self._load_parent_docstore()
        collection = self._client().get_collection(
            name=COLLECTION_CHILDREN,
            embedding_function=self.embedding_function,
        )
        raw = collection.query(
            query_texts=[query],
            n_results=top_k_children,
            include=["documents", "metadatas", "distances"],
        )

        results: list[SearchResult] = []
        seen_parent_ids: set[str] = set()
        for child_id, child_text, child_metadata, distance in zip(
            raw["ids"][0],
            raw["documents"][0],
            raw["metadatas"][0],
            raw["distances"][0],
        ):
            parent_id = child_metadata["parent_id"]
            if parent_id in seen_parent_ids:
                continue

            parent_record = parent_docstore[parent_id]
            seen_parent_ids.add(parent_id)
            results.append(
                SearchResult(
                    parent_id=parent_id,
                    parent_text=parent_record["page_content"],
                    parent_metadata=parent_record["metadata"],
                    best_child_id=child_id,
                    best_child_text=child_text,
                    best_child_score=self._distance_to_score(distance),
                    best_child_distance=distance,
                )
            )
            if len(results) >= top_k_parents:
                break

        return results

    def search_parent_chunks_directly(self, query: str, top_k: int = 2) -> list[ParentDirectHit]:
        """对比用：父块也直接进向量库时，直接检索父块会是什么效果。"""

        collection = self._client().get_collection(
            name=COLLECTION_PARENT_DEBUG,
            embedding_function=self.embedding_function,
        )
        raw = collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            ParentDirectHit(
                parent_id=parent_id,
                parent_text=parent_text,
                score=self._distance_to_score(distance),
                distance=distance,
            )
            for parent_id, parent_text, distance in zip(
                raw["ids"][0],
                raw["documents"][0],
                raw["distances"][0],
            )
        ]

    def describe_storage(self) -> dict[str, Any]:
        parent_docstore = self._load_parent_docstore()
        child_count = self._client().get_collection(
            name=COLLECTION_CHILDREN,
            embedding_function=self.embedding_function,
        ).count()
        return {
            "runtime_dir": str(self.runtime_dir),
            "chroma_dir": str(self.chroma_dir),
            "parent_docstore": str(self.parent_docstore_path),
            "parent_count": len(parent_docstore),
            "child_count": child_count,
        }

    def _client(self):
        return chromadb.PersistentClient(path=str(self.chroma_dir))

    def _load_parent_docstore(self) -> dict[str, dict[str, Any]]:
        return json.loads(self.parent_docstore_path.read_text(encoding="utf-8"))

    def _assert_demo_runtime_dir(self) -> None:
        expected_parent = Path(__file__).parent.resolve()
        resolved = self.runtime_dir.resolve()
        if resolved.parent != expected_parent or resolved.name != RUNTIME_DIR_NAME:
            raise ValueError(f"Refuse to delete non-demo runtime directory: {resolved}")

    @staticmethod
    def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        cleaned = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                cleaned[key] = value
            elif value is not None:
                cleaned[key] = json.dumps(value, ensure_ascii=False)
        return cleaned

    @staticmethod
    def _distance_to_score(distance: float) -> float:
        return 1.0 - distance


def sample_documents() -> list[dict[str, Any]]:
    return [
        {
            "text": """
第一章 SAR舰船检测概述

合成孔径雷达（SAR）是一种主动式微波遥感技术，能够全天候、全天时获取地表信息。
SAR图像在海洋监测领域具有重要应用价值，特别是在舰船检测方面。

第二章 检测算法

本系统采用基于深度学习的目标检测算法 MBE-Net。
MBE-Net 在 SAR 舰船检测任务上取得了优异的性能。
在公开数据集上的测试结果表明，mAP 达到 92.3%，召回率达到 95.1%。
相比传统 CFAR 算法，MBE-Net 在复杂海况下的误检率降低了 40%。

第三章 系统架构

系统采用前后端分离架构，前端使用 Streamlit 和 FastAPI 双模式。
后端集成了 RAG 检索增强生成能力，支持知识库问答。
智能体基于 LangChain ReAct 框架实现，支持多轮对话和工具调用。
""",
            "metadata": {"source": "SAR技术文档", "doc_id": "sar_handbook"},
        }
    ]


def self_test() -> None:
    retriever = EngineeringParentChildRetriever(
        runtime_dir=Path(__file__).parent / RUNTIME_DIR_NAME
    )
    retriever.reset_runtime()
    retriever.build_index(sample_documents())

    results = retriever.search("舰船检测的mAP和召回率是多少？")

    assert results, "应该能检索到至少一个父块"
    best = results[0]
    assert "mAP 达到 92.3%" in best.parent_text
    assert "MBE-Net" in best.parent_text
    assert best.parent_id
    assert "mAP 达到 92.3%" in best.best_child_text
    assert "召回率达到 95.1%" in best.best_child_text
    assert retriever.describe_storage()["child_count"] > retriever.describe_storage()["parent_count"]


def preview(text: str, max_len: int = 96) -> str:
    compact = " ".join(text.split())
    return compact[:max_len] + ("..." if len(compact) > max_len else "")


def run_walkthrough() -> None:
    retriever = EngineeringParentChildRetriever(
        runtime_dir=Path(__file__).parent / RUNTIME_DIR_NAME
    )
    retriever.reset_runtime()
    retriever.build_index(sample_documents())

    storage = retriever.describe_storage()
    query = "舰船检测的mAP和召回率是多少？"

    print("=" * 72)
    print("工程版父子块检索：child 进向量库，parent 进 docstore")
    print("=" * 72)
    print(f"Chroma 子块索引: {storage['chroma_dir']}")
    print(f"父块 docstore : {storage['parent_docstore']}")
    print(f"父块数量      : {storage['parent_count']}")
    print(f"子块数量      : {storage['child_count']}")
    print()

    print("在线查询流程")
    print("-" * 72)
    print(f"用户 query: {query}")
    print("1. Chroma 只检索 child chunks")
    print("2. child.metadata['parent_id'] 指向父块")
    print("3. 从 JSON docstore 回表取 parent")
    print("4. 多个 child 命中同一 parent 时去重")
    print()

    results = retriever.search(query, top_k_children=6, top_k_parents=2)
    for rank, result in enumerate(results, start=1):
        print(f"[父子检索结果 {rank}] parent_id={result.parent_id}")
        print(f"  child_id : {result.best_child_id}")
        print(f"  child分数: {result.best_child_score:.4f}")
        print(f"  命中child: {preview(result.best_child_text)}")
        print(f"  返回parent给LLM: {preview(result.parent_text, max_len=150)}")
        print()

    print("对比：父块也直接进向量库")
    print("-" * 72)
    print("这个 collection 只用于演示/兜底；标准父子块主线一般不靠它检索。")
    direct_hits = retriever.search_parent_chunks_directly(query, top_k=2)
    for rank, hit in enumerate(direct_hits, start=1):
        print(f"[父块直搜结果 {rank}] parent_id={hit.parent_id}, score={hit.score:.4f}")
        print(f"  直接命中的parent: {preview(hit.parent_text, max_len=150)}")
        print()

    print("结论")
    print("-" * 72)
    print("工程实现里，child 是召回单元，parent 是上下文单元。")
    print("向量库负责找小块；docstore 负责按 parent_id 取完整上下文。")


if __name__ == "__main__":
    run_walkthrough()
