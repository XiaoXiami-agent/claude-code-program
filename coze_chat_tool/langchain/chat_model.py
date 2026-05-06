from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field, PrivateAttr

from coze_chat_tool.client import CozeChatClient


class CozeChatModel(BaseChatModel):
    """LangChain ChatModel wrapping Coze API."""

    token: str = Field(description="Coze PAT token")
    bot_id: str = Field(description="Coze bot ID")
    base_url: str = Field(default="https://api.coze.cn", description="API base URL")
    timeout: float = Field(default=30.0, description="Request timeout in seconds")
    _client: CozeChatClient = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._client = CozeChatClient(
            token=self.token,
            bot_id=self.bot_id,
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @property
    def _llm_type(self) -> str:
        return "coze"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> ChatResult:
        prompt = self._messages_to_prompt(messages)
        conversation_id = kwargs.get("conversation_id")

        if conversation_id:
            content = self._client.chat_with_conversation(conversation_id, prompt)
        else:
            content = self._client.chat(prompt)

        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs,
    ) -> Iterator[ChatGenerationChunk]:
        from langchain_core.messages import AIMessageChunk

        prompt = self._messages_to_prompt(messages)
        for text_chunk in self._client.chat_stream(prompt):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=text_chunk))
            if run_manager:
                run_manager.on_llm_new_token(text_chunk, chunk=chunk)
            yield chunk

    def _messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        from langchain_core.messages import SystemMessage

        system_parts = []
        human_parts = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                system_parts.append(msg.content)
            elif isinstance(msg, HumanMessage):
                human_parts.append(msg.content)

        result = []
        if system_parts:
            result.append("System: " + "\n".join(system_parts))
        result.extend(human_parts)
        return "\n".join(result) if result else ""
