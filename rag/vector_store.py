from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
import os
from utils.logger_handler import logger
from utils.file_handler import text_loader, pdf_loader, listdir_with_allowed_type, \
    get_file_hash, load_manifest, save_manifest, check_file_status, update_manifest_entry
from rag.hybrid_retriever import DynamicHybridRetriever

class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"])

        )

        self.pdf_splitter = self._build_spliter(chunk_size=chroma_conf.get("pdf_chunk_size", chroma_conf["chunk_size"]),
                                                chunk_overlap=chroma_conf.get("pdf_chunk_overlap", chroma_conf["chunk_overlap"]))
        self.txt_splitter = self._build_spliter(chunk_size=chroma_conf.get("txt_chunk_size", chroma_conf["chunk_size"]),
                                                chunk_overlap=chroma_conf.get("txt_chunk_overlap", chroma_conf["chunk_overlap"]))
        self.default_splitter = self._build_spliter(chunk_size=chroma_conf.get("chunk_size"),
                                                    chunk_overlap=chroma_conf.get("chunk_overlap"))

        self.hybrid_engine = DynamicHybridRetriever(
            vector_store=self.vector_store,
            k=15
        )

        # 语义断点分块配置
        self.semantic_enabled = chroma_conf.get("semantic_chunking_enabled", True)
        self.semantic_threshold = chroma_conf.get("semantic_threshold", 0.5)
        self.semantic_min_size = chroma_conf.get("semantic_min_chunk_size", 100)
        self.semantic_max_size = chroma_conf.get("semantic_max_chunk_size", 800)

        # Manifest 注册表（替代原 md5.text）
        self.manifest_path = get_abs_path(chroma_conf["manifest_store"])
        self.manifest = load_manifest(self.manifest_path)

    def get_retriever(self, query: str):
        return self.hybrid_engine.get_retriever(query)

    def _build_spliter(self, chunk_size, chunk_overlap):
        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=chroma_conf["separators"],
            length_function=len
        )
    def _get_splitter(self, read_path: str) -> RecursiveCharacterTextSplitter:
        """根据文件类型选择更合适的切块器。"""
        if read_path.endswith(".txt"):
            return self.txt_splitter
        if read_path.endswith(".pdf"):
            return self.pdf_splitter
        return self.default_splitter

    @staticmethod
    def _cosine_sim(a, b):
        """计算两个向量的余弦相似度（纯 Python，无需 numpy）。"""
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _semantic_split(self, documents):
        """基于相邻句子 embedding 相似度的语义断点分块。"""
        # 第1步：切分为句子级单元
        sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.semantic_min_size,
            chunk_overlap=0,
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", "：", ":"]
        )
        sentences = sentence_splitter.split_documents(documents)
        if len(sentences) <= 1:
            return documents

        # 第2步：批量计算 embedding
        texts = [d.page_content for d in sentences]
        try:
            embeddings = embed_model.embed_documents(texts)
        except Exception as e:
            logger.warning(f"语义分块嵌入失败，回退到固定分块: {e}")
            return None

        # 第3步：计算相邻句子相似度，找到语义断点
        boundaries = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_sim(embeddings[i], embeddings[i + 1])
            if sim < self.semantic_threshold:
                boundaries.append(i)

        # 第4步：按断点合并句子为 chunk（chunk_index 由 _enrich_chunks 统一分配）
        chunks = []
        start = 0
        all_ends = boundaries + [len(sentences) - 1]

        for end in all_ends:
            segment = [sentences[j].page_content for j in range(start, end + 1)]
            merged = "\n".join(segment)

            base_meta = sentences[start].metadata if sentences[start].metadata else {}
            if len(merged) <= self.semantic_max_size:
                chunks.append(Document(page_content=merged, metadata=base_meta))
            else:
                # 超长段落在句子边界做子切分
                sub_texts = []
                sub_len = 0
                for j in range(start, end + 1):
                    seg = sentences[j].page_content
                    if sub_len + len(seg) > self.semantic_max_size and sub_texts:
                        chunks.append(Document(page_content="\n".join(sub_texts), metadata=base_meta))
                        sub_texts = [seg]
                        sub_len = len(seg)
                    else:
                        sub_texts.append(seg)
                        sub_len += len(seg)
                if sub_texts:
                    chunks.append(Document(page_content="\n".join(sub_texts), metadata=base_meta))

            start = end + 1

        # 第5步：合并过小的 chunk 到相邻 chunk（保留首个 chunk 的 metadata）
        final_chunks = []
        pending = None
        for ch in chunks:
            if len(ch.page_content) < self.semantic_min_size:
                if pending:
                    pending = Document(
                        page_content=pending.page_content + "\n" + ch.page_content,
                        metadata=pending.metadata
                    )
                else:
                    pending = ch
            else:
                if pending:
                    merged_text = pending.page_content + "\n" + ch.page_content
                    # 保留较大 chunk 的 metadata（含 page 等有用字段）
                    merged_meta = ch.metadata if ch.metadata else pending.metadata
                    final_chunks.append(Document(page_content=merged_text, metadata=merged_meta))
                    pending = None
                else:
                    final_chunks.append(ch)
        if pending and final_chunks:
            final_chunks[-1] = Document(
                page_content=final_chunks[-1].page_content + "\n" + pending.page_content,
                metadata=final_chunks[-1].metadata
            )
        elif pending:
            final_chunks.append(pending)

        result = final_chunks if final_chunks else sentences
        logger.info(
            f"语义分块完成: {len(sentences)}个句子 → {len(boundaries)}个断点 → {len(result)}个chunk"
        )
        return result

    def _enrich_chunks(self, chunks, doc_id, file_hash, file_path, file_type):
        """为 chunk 批量注入完整 metadata（doc_id / chunk_id / file_hash / source / filename / file_type / chunk_index）
           并生成稳定 chunk_id 序列"""
        filename = os.path.basename(file_path)
        enriched_chunks = []
        chunk_ids = []

        for i, chunk in enumerate(chunks):
            cid = f"{doc_id}_{i}"
            meta = {
                "doc_id": doc_id,
                "chunk_id": cid,
                "file_hash": file_hash,
                "source": file_path,
                "filename": filename,
                "file_type": file_type,
                "chunk_index": i,
            }
            # PDF 文档如果有 page 信息，保留
            if "page" in chunk.metadata:
                meta["page"] = chunk.metadata["page"]

            enriched_chunks.append(Document(
                page_content=chunk.page_content,
                metadata=meta
            ))
            chunk_ids.append(cid)

        return enriched_chunks, chunk_ids

    def delete_document(self, filename, delete_file=False):
        """按文件名删除该文档的所有向量及 manifest 记录。

        delete_file=True 时同步删除 data/ 下的原始文件，避免下次入库又被扫描回来。
        """
        entry = self.manifest.get(filename)
        if not entry:
            logger.warning(f"manifest 中未找到 {filename}")
            return 0

        chunk_ids = entry.get("chunk_ids", [])
        if chunk_ids:
            self.vector_store.delete(ids=chunk_ids)
            logger.info(f"已删除 {filename} 的 {len(chunk_ids)} 个向量")

        if delete_file:
            data_dir = os.path.abspath(get_abs_path(chroma_conf["data_path"]))
            file_path = os.path.abspath(os.path.join(data_dir, filename))
            if file_path.startswith(data_dir + os.sep) and os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"已删除原始文件 {file_path}")

        del self.manifest[filename]
        save_manifest(self.manifest_path, self.manifest)
        self.hybrid_engine.rebuild_bm25()
        return len(chunk_ids)

    def delete_document_by_doc_id(self, doc_id, delete_file=False):
        """按 doc_id 删除文档（比文件名更稳定，避免重名/改名/乱码问题）"""
        for fname, entry in self.manifest.items():
            if entry.get("doc_id") == doc_id:
                return self.delete_document(fname, delete_file=delete_file)
        logger.warning(f"manifest 中未找到 doc_id={doc_id}")
        return 0

    def load_document(self):
        """
        加载文件，存入向量库（manifest 模式去重 + 稳定 chunk ID）
        :return: (new_count, updated_count, skipped_count, removed_count)
        """
        def get_file_documents(read_path):
            if read_path.endswith("txt"):
                return text_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allow_files_path = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"])
        )

        new_count = 0
        updated_count = 0
        skipped_count = 0
        duplicate_count = 0

        for path in allow_files_path:
            filename = os.path.basename(path)
            file_hash = get_file_hash(path)
            file_type = path.rsplit(".", 1)[-1] if "." in path else "unknown"

            if file_hash is None:
                logger.error(f"文件 {filename} hash 计算失败，跳过")
                continue

            # 判断文件状态：NEW / UPDATED / SAME / DUPLICATE
            status = check_file_status(self.manifest, filename, file_hash)

            if status == "SAME":
                logger.info(f"文件 {filename} 未变更，跳过")
                skipped_count += 1
                continue

            if status == "DUPLICATE":
                # 同内容文件已以另一个名字入库，跳过不重复入库
                logger.info(f"文件 {filename} 内容与其他文件重复，跳过入库")
                duplicate_count += 1
                skipped_count += 1
                continue

            # UPDATED: 文件内容变了 → 删旧向量，重新入库
            if status == "UPDATED":
                logger.info(f"文件 {filename} 已更新，删除旧向量后重新入库")
                self.delete_document(filename)

            # 生成 doc_id
            doc_id = file_hash[:16]

            try:
                documents = get_file_documents(path)
                if not documents:
                    logger.error(f"文件 {filename} 内容为空，跳过")
                    continue

                # 分块
                chunk_method = "semantic"
                split_document = None
                if self.semantic_enabled:
                    split_document = self._semantic_split(documents)
                if split_document is None:
                    split_document = self._get_splitter(path).split_documents(documents)
                    chunk_method = "fixed"
                if not split_document:
                    logger.error(f"文件 {filename} 分片后内容为空，跳过")
                    continue

                # 注入 metadata + 生成稳定 ID
                enriched_chunks, chunk_ids = self._enrich_chunks(
                    split_document, doc_id, file_hash, path, file_type
                )

                # 写入 ChromaDB（带稳定 ID → 不会产生重复向量）
                self.vector_store.add_documents(
                    enriched_chunks, ids=chunk_ids
                )

                # 更新 manifest
                update_manifest_entry(
                    self.manifest, filename, doc_id, file_hash,
                    chunk_count=len(enriched_chunks),
                    chunk_ids=chunk_ids,
                    chunk_method=chunk_method,
                    file_type=file_type
                )
                # 计数：NEW 只算 new_count，UPDATED 只算 updated_count
                if status == "NEW":
                    new_count += 1
                elif status == "UPDATED":
                    updated_count += 1
                logger.info(
                    f"文件 {filename} 入库成功 [{status}] "
                    f"({chunk_method}, {len(enriched_chunks)} chunks, doc_id={doc_id})"
                )

            except Exception as e:
                logger.error(f"文件 {filename} 入库失败: {str(e)}", exc_info=True)
                continue

        # 清理：manifest 中的文件已物理删除 → 移除残留记录
        current_filenames = {os.path.basename(p) for p in allow_files_path}
        stale_files = [f for f in self.manifest if f not in current_filenames]
        removed_count = 0
        for stale in stale_files:
            logger.info(f"文件 {stale} 已不存在，清理残留向量")
            self.delete_document(stale)
            removed_count += 1

        # 持久化 manifest
        save_manifest(self.manifest_path, self.manifest)

        if new_count > 0 or updated_count > 0 or removed_count > 0:
            self.hybrid_engine.rebuild_bm25()

        return new_count, updated_count, skipped_count, removed_count





if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    test_query = "石头"
    retriever = vs.get_retriever(test_query)
    res = retriever.invoke(test_query)
    print(res)

