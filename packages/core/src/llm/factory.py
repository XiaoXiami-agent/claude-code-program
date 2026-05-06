"""LLM 工厂：统一接口，多提供商。引擎只依赖 LLMAdapter Protocol，不绑定具体模型。"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Protocol

# 消息格式：兼容 OpenAI Chat API
# {"role": "system"|"user"|"assistant", "content": "..."}


class LLMAdapter(Protocol):
    """所有 LLM 提供商需实现的接口。引擎只依赖此 Protocol。"""

    def chat(self, messages: list[dict], **kwargs) -> str:
        """发送消息列表，返回模型回复文本。"""
        ...


# ─── DeepSeek ────────────────────────────────────────

class DeepSeekAdapter:
    """通过 HTTP 调 DeepSeek OpenAI 兼容 API。零额外依赖。"""

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",
        base_url: str = "https://api.deepseek.com",
        temperature: float = 0.8,
        max_tokens: int = 2048,
        timeout: float = 60.0,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout = timeout

    def chat(self, messages: list[dict], **kwargs) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": messages,
            "temperature": kwargs.get("temperature", self._temperature),
            "max_tokens": kwargs.get("max_tokens", self._max_tokens),
            "stream": False,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )

        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                return body["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"DeepSeek API error {e.code}: {error_body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"DeepSeek connection failed: {e.reason}") from e


# ─── Coze ───────────────────────────────────────────

class CozeAdapter:
    """包装已有的 CozeChatModel，统一到 LLMAdapter 接口。"""

    def __init__(self, token: str, bot_id: str, **kwargs) -> None:
        from coze_chat_tool.langchain import CozeChatModel
        self._model = CozeChatModel(token=token, bot_id=bot_id, **kwargs)

    def chat(self, messages: list[dict], **kwargs) -> str:
        # 转换 dict 消息 → LangChain 消息
        from langchain_core.messages import (
            SystemMessage,
            HumanMessage,
            AIMessage,
        )
        lc_messages = []
        for m in messages:
            role = m["role"]
            content = m["content"]
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))

        result = self._model.invoke(lc_messages, **kwargs)
        return result.content if hasattr(result, "content") else str(result)


# ─── 工厂 ───────────────────────────────────────────

def create_llm(provider: str, config: dict | None = None) -> LLMAdapter:
    """
    创建 LLM 适配器。

    provider: "deepseek" | "coze"
    config: 提供商相关配置

    DeepSeek 配置:
        api_key, model, base_url, temperature, max_tokens, timeout

    Coze 配置:
        token, bot_id, base_url(可选), timeout(可选)
    """
    cfg = config or {}

    if provider == "deepseek":
        return DeepSeekAdapter(
            api_key=cfg.get("api_key", ""),
            model=cfg.get("model", "deepseek-chat"),
            base_url=cfg.get("base_url", "https://api.deepseek.com"),
            temperature=cfg.get("temperature", 0.8),
            max_tokens=cfg.get("max_tokens", 2048),
            timeout=cfg.get("timeout", 60.0),
        )

    if provider == "coze":
        return CozeAdapter(
            token=cfg.get("token", ""),
            bot_id=cfg.get("bot_id", ""),
            base_url=cfg.get("base_url", "https://api.coze.cn"),
            timeout=cfg.get("timeout", 30.0),
        )

    raise ValueError(f"Unknown LLM provider: {provider}")
