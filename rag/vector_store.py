from langchain_chroma import Chroma
from langchain_core.documents import Document
from utils.config_handler import chroma_conf
from model.factory import chat_model, embed_model
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.path_tool import get_abs_path
import os
from utils.logger_handler import logger
from utils.file_handler import text_loader, pdf_loader, listdir_with_allowed_type, get_file_md5_hex
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

        # 第4步：按断点合并句子为 chunk
        chunks = []
        start = 0
        all_ends = boundaries + [len(sentences) - 1]

        for end in all_ends:
            segment = [sentences[j].page_content for j in range(start, end + 1)]
            merged = "\n".join(segment)

            if len(merged) <= self.semantic_max_size:
                chunks.append(Document(page_content=merged))
            else:
                # 超长段落在句子边界做子切分
                sub_texts = []
                sub_len = 0
                for j in range(start, end + 1):
                    seg = sentences[j].page_content
                    if sub_len + len(seg) > self.semantic_max_size and sub_texts:
                        chunks.append(Document(page_content="\n".join(sub_texts)))
                        sub_texts = [seg]
                        sub_len = len(seg)
                    else:
                        sub_texts.append(seg)
                        sub_len += len(seg)
                if sub_texts:
                    chunks.append(Document(page_content="\n".join(sub_texts)))

            start = end + 1

        # 第5步：合并过小的 chunk 到相邻 chunk
        final_chunks = []
        pending = None
        for ch in chunks:
            if len(ch.page_content) < self.semantic_min_size:
                if pending:
                    pending = Document(page_content=pending.page_content + "\n" + ch.page_content)
                else:
                    pending = ch
            else:
                if pending:
                    merged_text = pending.page_content + "\n" + ch.page_content
                    final_chunks.append(Document(page_content=merged_text))
                    pending = None
                else:
                    final_chunks.append(ch)
        if pending and final_chunks:
            final_chunks[-1] = Document(
                page_content=final_chunks[-1].page_content + "\n" + pending.page_content
            )
        elif pending:
            final_chunks.append(pending)

        result = final_chunks if final_chunks else sentences
        logger.info(
            f"语义分块完成: {len(sentences)}个句子 → {len(boundaries)}个断点 → {len(result)}个chunk"
        )
        return result

    def load_document(self):
        """
        加载文件，存入向量库
        :return: (new_count, skipped_count) 实际新入库数量和跳过数量
        """
        def check_md5_hex(md5_for_check):
            if not os.path.exists(get_abs_path(chroma_conf["md5_hex_store"])):
                open(get_abs_path(chroma_conf["md5_hex_store"]), 'w', encoding="UTF-8").close()
                return False
            with open(get_abs_path(chroma_conf["md5_hex_store"]), 'r', encoding="UTF-8") as f:
                for line in f:
                    line = line.strip()
                    if md5_for_check == line:
                        return True
            return False

        def save_md5_hex(md5_for_check):
            with open(get_abs_path(chroma_conf["md5_hex_store"]), 'a', encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

        def get_file_documents(read_path):
            if read_path.endswith("txt"):
                return text_loader(read_path)
            if read_path.endswith("pdf"):
                return pdf_loader(read_path)
            return []

        allow_files_path = listdir_with_allowed_type(get_abs_path(chroma_conf["data_path"]),
                                                     tuple(chroma_conf["allow_knowledge_file_type"]))

        new_count = 0
        skipped_count = 0

        for path in allow_files_path:
            md5_hex = get_file_md5_hex(path)
            if check_md5_hex(md5_hex):
                logger.info(f"加载知识库{path}下的文件已经存在，跳过")
                skipped_count += 1
                continue
            try:
                documents = get_file_documents(path)
                if not documents:
                    logger.error(f"加载知识库{path}文件下内容为空，跳过")
                    continue
                split_document = None
                if self.semantic_enabled:#先尝试语义分块
                    split_document = self._semantic_split(documents)
                if split_document is None:
                    split_document = self._get_splitter(path).split_documents(documents)
                if not split_document:
                    logger.error(f"加载知识库{path}分片后内容为空，跳过")
                    continue
                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)
                new_count += 1
                logger.info(f"加载知识库{path}下的文件成功")
            except Exception as e:
                logger.error(f"加载知识库{path}失败, 错误详情: {str(e)}", exc_info=True)
                continue

        return new_count, skipped_count





if __name__ == '__main__':
    vs = VectorStoreService()
    vs.load_document()
    test_query = "石头"
    retriever = vs.get_retriever(test_query)
    res = retriever.invoke(test_query)
    print(res)

