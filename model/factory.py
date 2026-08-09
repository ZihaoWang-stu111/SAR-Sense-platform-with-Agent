import os
from abc import ABC, abstractmethod
from typing import Optional

from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel, ChatTongyi
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings.dashscope import embed_with_retry
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from utils.call_governance import (
    DEFAULT_DASHSCOPE_TIMEOUT_S,
    DEFAULT_OPENAI_TIMEOUT_S,
    DEFAULT_OLLAMA_TIMEOUT_S,
    acall_with_retries,
    call_with_retries,
    get_timeout_s,
    resolve_chat_provider_model,
)
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider, model_name = resolve_chat_provider_model()
        if provider == "ollama":
            # 禁止 httpx 继承 Windows 系统代理，避免 localhost 被代理后 502。
            return ChatOllama(
                model=model_name,
                base_url=os.getenv(
                    "OLLAMA_BASE_URL",
                    rag_conf.get("ollama_base_url", "http://localhost:11434"),
                ),
                reasoning=False,
                num_ctx=8192,
                client_kwargs={
                    "trust_env": False,
                    "timeout": get_timeout_s(
                        "OLLAMA_TIMEOUT_S", DEFAULT_OLLAMA_TIMEOUT_S
                    ),
                },
            )
        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
                timeout=get_timeout_s("OPENAI_TIMEOUT_S", DEFAULT_OPENAI_TIMEOUT_S),
                max_retries=0,
                use_responses_api=False,
            )
        if provider == "dashscope":
            # request_timeout → DashScope HTTP；max_retries=1 把重试交给 call_governance
            return ChatTongyi(
                model=model_name,
                max_retries=1,
                model_kwargs={
                    "request_timeout": get_timeout_s(
                        "DASHSCOPE_TIMEOUT_S", DEFAULT_DASHSCOPE_TIMEOUT_S
                    ),
                },
            )
        raise ValueError(f"Unsupported chat provider: {provider}")


class _BatchedDashScopeEmbeddings(DashScopeEmbeddings):
    """qwen3.7-text-embedding 单批 ≤20；外层切片后再走 API。"""

    embed_batch_size: int = 20

    def _embed(self, texts, *, text_type: str):
        timeout = get_timeout_s("DASHSCOPE_TIMEOUT_S", DEFAULT_DASHSCOPE_TIMEOUT_S)
        model_name = getattr(self, "model", "") or ""

        def _one(batch):
            items = embed_with_retry(
                self,
                input=batch,
                text_type=text_type,
                model=self.model,
                request_timeout=timeout,
            )
            return [item["embedding"] for item in items]

        if isinstance(texts, str):
            texts = [texts]
        out = []
        for i in range(0, len(texts), self.embed_batch_size):
            batch = texts[i : i + self.embed_batch_size]
            out.extend(
                call_with_retries(
                    lambda b=batch: _one(b),
                    provider="dashscope",
                    model=model_name,
                )
            )
        return out

    def embed_documents(self, texts):
        return self._embed(texts, text_type="document")

    def embed_query(self, text):
        return self._embed(text, text_type="query")[0]


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return _BatchedDashScopeEmbeddings(
            model=os.getenv(
                "EMBEDDING_MODEL_NAME",
                rag_conf["embedding_model_name"],
            ),
            embed_batch_size=rag_conf.get("embedding_batch_size", 20),
            max_retries=1,  # tenacity 只试 1 次；重试归 call_governance
        )


def _govern_chat_model_methods(model):
    """patch invoke/ainvoke；pydantic 模型需 object.__setattr__。"""
    if model is None:
        return model
    provider, model_name = resolve_chat_provider_model()
    orig_invoke, orig_ainvoke = model.invoke, model.ainvoke

    def invoke(*args, **kwargs):
        return call_with_retries(
            lambda: orig_invoke(*args, **kwargs),
            provider=provider,
            model=model_name,
        )

    async def ainvoke(*args, **kwargs):
        return await acall_with_retries(
            lambda: orig_ainvoke(*args, **kwargs),
            provider=provider,
            model=model_name,
        )

    object.__setattr__(model, "invoke", invoke)
    object.__setattr__(model, "ainvoke", ainvoke)
    return model


chat_model = _govern_chat_model_methods(ChatModelFactory().generator())
embed_model = EmbeddingsFactory().generator()
