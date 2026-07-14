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


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
