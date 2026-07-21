from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_ollama import ChatOllama
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = rag_conf.get("chat_provider", "dashscope")
        if provider == "ollama":
            # 本地 Ollama：chat 切到本地模型，embedding 仍用 DashScope（向量库不重建）
            # 禁止 httpx 继承 Windows 系统代理，避免 localhost 请求被代理后返回 502。
            return ChatOllama(
                model=rag_conf["chat_model_name"],
                base_url=rag_conf.get("ollama_base_url", "http://localhost:11434"),
                reasoning=False,
                num_ctx=8192,
                client_kwargs={"trust_env": False},
            )
        return ChatTongyi(model=rag_conf["chat_model_name"])


class _BatchedDashScopeEmbeddings(DashScopeEmbeddings):
    """对 batch 受限模型按 embed_batch_size 分批嵌入。

    DashScopeEmbeddings 对未登记在内部 BATCH_SIZE 的模型默认按 25 条一批，
    而 qwen3.7-text-embedding 限制单批 ≤ 20，直接用会 400。外层先按
    embed_batch_size 切片，每批再走父类 embed_documents（其内部仍按模型
    BATCH_SIZE 再切），保证单次 API 调用不超过模型限制。
    """

    embed_batch_size: int = 20

    def embed_documents(self, texts):
        embeddings = []
        for i in range(0, len(texts), self.embed_batch_size):
            embeddings.extend(
                super().embed_documents(texts[i : i + self.embed_batch_size])
            )
        return embeddings


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return _BatchedDashScopeEmbeddings(
            model=rag_conf["embedding_model_name"],
            embed_batch_size=rag_conf.get("embedding_batch_size", 20),
        )


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
