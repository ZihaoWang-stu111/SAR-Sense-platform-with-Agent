import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from rag.vector_store import get_vector_store_service
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model
from langchain_core.output_parsers import StrOutputParser
from rag.reranker import BGERerankerService
from rag.parent_child_retriever import ParentChildResolver
from utils.config_handler import chroma_conf


class RagSummarizeService:
    def __init__(self):
        self.vector_store = get_vector_store_service()

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model

        self.reranker = BGERerankerService(top_n=3)
        self.chain = self._init_chain()
        self.parent_resolver = None
        if self.vector_store.parent_child_enabled and self.vector_store.parent_docstore:
            self.parent_resolver = ParentChildResolver(
                parent_docstore=self.vector_store.parent_docstore,
                top_k_parents=chroma_conf.get("retrieve_k_parents", 6),
            )

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query):
        candidate_docs = self.vector_store.get_retriever(query).invoke(query)

        if self.parent_resolver:
            candidate_docs = self.parent_resolver.resolve(candidate_docs)

        final_docs = self.reranker.rerank(query, candidate_docs)

        return final_docs


    def rag_summarize(self, query):
        docs = self.retriever_docs(query)

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
            context += f"[{i}] {doc.page_content}\n"
            context += (
                f"来源: {filename} | chunk_id={chunk_id} | parent_index={chunk_index} "
                f"| match_child={match_child} | page={page} | score={score}\n\n"
            )
            sources.append(f"[{i}] {filename} | chunk_id={chunk_id} | score={score}")

        answer = self.chain.invoke(
            {"input": query,
             "context": context}
        ).strip()
        for marker in ("参考来源：", "参考来源:"):
            if marker in answer:
                answer = answer.split(marker, 1)[0].strip()
                break

        return f"{answer}\n\n参考来源：\n" + "\n".join(sources)


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.retriever_docs("我家是复式结构，不仅有楼梯，而且客厅还铺了长毛地毯，另外家里养了三只掉毛的猫。请问这种极端情况，扫地机器人能应付得来吗？如果会迷路或者卡住，该怎么解决？"))
