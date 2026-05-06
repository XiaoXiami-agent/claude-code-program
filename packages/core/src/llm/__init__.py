"""LLM 适配层。所有提供商实现统一接口，引擎不绑定任何具体模型。"""

from .factory import create_llm, LLMAdapter

__all__ = ["create_llm", "LLMAdapter"]
