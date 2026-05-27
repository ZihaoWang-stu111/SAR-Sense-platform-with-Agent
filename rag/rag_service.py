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


    def rag_summarize(self,query):
        counter = 0
        context = ""
        for doc in self.retriever_docs(query):
            counter += 1
            source = doc.metadata.get('source', '未知文档')
            context += f"[参考资料{counter}] (来源:{source}): {doc.page_content}\n"

        return self.chain.invoke(
            {"input": query,
             "context": context}
        )


if __name__ == '__main__':
    rag = RagSummarizeService()
    print(rag.retriever_docs("我家是复式结构，不仅有楼梯，而且客厅还铺了长毛地毯，另外家里养了三只掉毛的猫。请问这种极端情况，扫地机器人能应付得来吗？如果会迷路或者卡住，该怎么解决？"))
