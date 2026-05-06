"""短时记忆存储实现。都满足 MemoryStore Protocol，可按环境替换。"""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict

from .types import MemoryItem


class InMemoryMemoryStore:
    """内存存储。用于测试、原型、或不需要持久化的场景。"""

    def __init__(self) -> None:
        self._store: dict[str, list[MemoryItem]] = defaultdict(list)

    def add(self, agent_id: str, item: MemoryItem) -> None:
        self._store[agent_id].append(item)

    def get_recent(self, agent_id: str, n: int) -> list[MemoryItem]:
        items = self._store.get(agent_id, [])
        return items[-n:] if n > 0 else []

    def get_all(self, agent_id: str) -> list[MemoryItem]:
        return list(self._store.get(agent_id, []))


class JSONFileMemoryStore:
    """JSON 文件存储。每个 Agent 一个独立文件，方便手动编辑和移植。"""

    def __init__(self, data_dir: str | Path) -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, agent_id: str) -> Path:
        return self._dir / f"{agent_id}.json"

    def _load(self, agent_id: str) -> list[dict]:
        path = self._path(agent_id)
        if not path.exists():
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return []

    def _save(self, agent_id: str, items: list[MemoryItem]) -> None:
        path = self._path(agent_id)
        data = [
            {
                "id": m.id,
                "timestamp": m.timestamp,
                "content": m.content,
                "tags": m.tags,
            }
            for m in items
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def add(self, agent_id: str, item: MemoryItem) -> None:
        items = self._load(agent_id)
        items.append({
            "id": item.id,
            "timestamp": item.timestamp,
            "content": item.content,
            "tags": item.tags,
        })
        self._save(agent_id, self._dicts_to_items(items))

    def get_recent(self, agent_id: str, n: int) -> list[MemoryItem]:
        items = self._load(agent_id)
        if n <= 0:
            return []
        return self._dicts_to_items(items[-n:])

    def get_all(self, agent_id: str) -> list[MemoryItem]:
        return self._dicts_to_items(self._load(agent_id))

    @staticmethod
    def _dicts_to_items(data: list[dict]) -> list[MemoryItem]:
        return [
            MemoryItem(
                id=d["id"],
                timestamp=d.get("timestamp", ""),
                content=d.get("content", ""),
                tags=d.get("tags", []),
            )
            for d in data
        ]
