import json
import time
import uuid
from typing import Iterator

import requests

from coze_chat_tool._http import request, post_stream
from coze_chat_tool.errors import CozeAPIError, CozeNetworkError


class CozeChatClient:
    def __init__(
        self,
        token: str,
        bot_id: str,
        *,
        base_url: str = "https://api.coze.cn",
        user_id: str | None = None,
        timeout: float = 30.0,
    ):
        if not token:
            raise ValueError("Token must not be empty")
        if not bot_id:
            raise ValueError("bot_id must not be empty")

        self.token = token
        self.bot_id = bot_id
        self.base_url = base_url.rstrip("/")
        self.user_id = user_id or uuid.uuid4().hex[:16]
        self.timeout = timeout

        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        })

    # --- Public API ---

    def chat(self, prompt: str) -> str:
        """Non-streaming chat. Returns full assistant reply."""
        return self._chat_and_wait(self._build_payload(prompt, stream=False))

    def chat_stream(self, prompt: str) -> Iterator[str]:
        """Streaming chat. Yields text chunks from the model."""
        payload = self._build_payload(prompt, stream=True)
        url = f"{self.base_url}/v3/chat"
        resp = post_stream(self._session, url, payload, self.timeout)
        yield from self._parse_sse(resp)

    def create_conversation(self) -> str:
        """Create a new conversation. Returns conversation_id."""
        url = f"{self.base_url}/v1/conversation/create"
        body = request(self._session, "POST", url, self.timeout, json={"bot_id": self.bot_id})
        return body["data"]["id"]

    def chat_with_conversation(self, conversation_id: str, prompt: str) -> str:
        """Chat within an existing conversation (multi-turn)."""
        payload = self._build_payload(prompt, stream=False)
        payload["conversation_id"] = conversation_id
        return self._chat_and_wait(payload)

    # --- Internal ---

    def _build_payload(self, prompt: str, *, stream: bool) -> dict:
        return {
            "bot_id": self.bot_id,
            "user_id": self.user_id,
            "stream": stream,
            "auto_save_history": not stream,
            "additional_messages": [
                {"role": "user", "content": prompt, "content_type": "text"}
            ],
        }

    def _chat_and_wait(self, payload: dict) -> str:
        url = f"{self.base_url}/v3/chat"
        body = request(self._session, "POST", url, self.timeout, json=payload)
        data = body["data"]
        self._poll_chat(data["id"], data["conversation_id"])
        return self._fetch_assistant_reply(data["id"], data["conversation_id"])

    def _poll_chat(self, chat_id: str, conversation_id: str, max_retries: int = 60) -> None:
        url = f"{self.base_url}/v3/chat/retrieve"
        for _ in range(max_retries):
            body = request(
                self._session, "GET", url, self.timeout,
                params={"chat_id": chat_id, "conversation_id": conversation_id},
            )
            status = body["data"]["status"]
            if status == "completed":
                return
            if status == "failed":
                raise CozeAPIError(body.get("msg", "Chat failed"))
            time.sleep(1)
        raise CozeNetworkError("Chat poll timed out after {max_retries} retries")

    def _fetch_assistant_reply(self, chat_id: str, conversation_id: str) -> str:
        url = f"{self.base_url}/v3/chat/message/list"
        body = request(
            self._session, "GET", url, self.timeout,
            params={"chat_id": chat_id, "conversation_id": conversation_id},
        )
        parts = []
        for msg in body.get("data", []):
            if msg.get("role") == "assistant" and msg.get("type") == "answer":
                parts.append(msg.get("content", ""))
        return "\n".join(parts)

    def _parse_sse(self, resp) -> Iterator[str]:
        event_type = ""
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    return
                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if event_type == "conversation.chat.failed":
                    raise CozeAPIError(data.get("msg", "Stream chat failed"))
                if event_type == "conversation.message.delta":
                    yield data.get("content", "")
