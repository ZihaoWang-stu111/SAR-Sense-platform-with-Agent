import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ANONYMIZED_TELEMETRY"] = "False"
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model,embed_model
from langchain_core.output_parsers import StrOutputParser
from rag.reranker import BGERerankerService


class RagSummarizeService:
    def __init__(self):
        self.vector_store = VectorStoreService()

        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model

        self.reranker = BGERerankerService(top_n=3)
        self.chain = self._init_chain()

    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    def retriever_docs(self, query):
        candidate_docs = self.vector_store.get_retriever(query).invoke(query)

        final_docs = self.reranker.rerank(query, candidate_docs)

        return final_docs


    def rag_summarize(self, query):
        context = ""
        for i, doc in enumerate(self.retriever_docs(query), 1):
            filename = doc.metadata.get('filename', '未知文档')
            chunk_id = doc.metadata.get('chunk_id', '-')
            chunk_index = doc.metadata.get('chunk_index', '-')
            page = doc.metadata.get('page', '-')
            score = doc.metadata.get('rerank_score', '-')
            if isinstance(score, float):
                score = f"{score:.4f}"
            context += f"[{i}] {doc.page_content}\n"
            context += f"来源: {filename} | chunk_id={chunk_id} | chunk_index={chunk_index} | page={page} | score={score}\n\n"

        return self.chain.invoke(
            {"input": query,
             "context": context}
        )


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.retriever_docs("我家是复式结构，不仅有楼梯，而且客厅还铺了长毛地毯，另外家里养了三只掉毛的猫。请问这种极端情况，扫地机器人能应付得来吗？如果会迷路或者卡住，该怎么解决？"))
