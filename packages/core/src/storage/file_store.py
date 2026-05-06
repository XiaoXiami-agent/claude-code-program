"""存储实现：满足 AgentStore / SceneStore Protocol。按环境可替换。"""

from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy

from ..types import AgentData, AgentConfig, AgentState


class JSONFileAgentStore:
    """每个 Agent 存为一个独立 JSON 文件。"""

    def __init__(self, data_dir: str | Path) -> None:
        self._dir = Path(data_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, agent_id: str) -> Path:
        return self._dir / f"{agent_id}.json"

    def save(self, agent_data: AgentData) -> None:
        path = self._path(agent_data.id)
        d = {
            "id": agent_data.id,
            "config": {
                "name": agent_data.config.name,
                "personality": agent_data.config.personality,
                "speaking_style": agent_data.config.speaking_style,
                "background": agent_data.config.background,
            },
            "state": {
                "hp": agent_data.state.hp,
                "mp": agent_data.state.mp,
                "emotion": agent_data.state.emotion,
                "location": agent_data.state.location,
                "relationships": agent_data.state.relationships,
                "inventory": agent_data.state.inventory,
                "buffs": agent_data.state.buffs,
                **agent_data.state.extras,
            },
            "goals": agent_data.goals,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(d, f, ensure_ascii=False, indent=2)

    def load(self, agent_id: str) -> AgentData | None:
        path = self._path(agent_id)
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return self._dict_to_data(d)

    def list_ids(self) -> list[str]:
        return [
            p.stem for p in self._dir.glob("*.json")
            if p.is_file()
        ]

    def delete(self, agent_id: str) -> None:
        path = self._path(agent_id)
        if path.exists():
            path.unlink()

    def save_all(self, agents: list[AgentData]) -> None:
        for a in agents:
            self.save(a)

    def load_all(self) -> list[AgentData]:
        result = []
        for aid in self.list_ids():
            data = self.load(aid)
            if data is not None:
                result.append(data)
        return result

    @staticmethod
    def _dict_to_data(d: dict) -> AgentData:
        cfg = d["config"]
        st = d["state"]
        known = {"hp", "mp", "emotion", "location", "relationships", "inventory", "buffs"}
        extras = {k: v for k, v in st.items() if k not in known}
        return AgentData(
            id=d["id"],
            config=AgentConfig(
                name=cfg["name"],
                personality=cfg["personality"],
                speaking_style=cfg["speaking_style"],
                background=cfg.get("background", ""),
            ),
            state=AgentState(
                hp=st.get("hp", 100),
                mp=st.get("mp", 100),
                emotion=st.get("emotion", "平静"),
                location=st.get("location", ""),
                relationships=st.get("relationships", {}),
                inventory=st.get("inventory", []),
                buffs=st.get("buffs", []),
                extras=extras,
            ),
            goals=d.get("goals", []),
        )


class InMemoryAgentStore:
    """内存 Agent 存储。测试用。"""

    def __init__(self) -> None:
        self._store: dict[str, dict] = {}

    def save(self, agent_data: AgentData) -> None:
        self._store[agent_data.id] = {
            "id": agent_data.id,
            "config": {
                "name": agent_data.config.name,
                "personality": agent_data.config.personality,
                "speaking_style": agent_data.config.speaking_style,
                "background": agent_data.config.background,
            },
            "state": {
                "hp": agent_data.state.hp,
                "mp": agent_data.state.mp,
                "emotion": agent_data.state.emotion,
                "location": agent_data.state.location,
                "relationships": deepcopy(agent_data.state.relationships),
                "inventory": list(agent_data.state.inventory),
                "buffs": list(agent_data.state.buffs),
                **agent_data.state.extras,
            },
            "goals": agent_data.goals,
        }

    def load(self, agent_id: str) -> AgentData | None:
        d = self._store.get(agent_id)
        if d is None:
            return None
        return JSONFileAgentStore._dict_to_data(d)

    def list_ids(self) -> list[str]:
        return list(self._store.keys())

    def delete(self, agent_id: str) -> None:
        self._store.pop(agent_id, None)

    def save_all(self, agents: list[AgentData]) -> None:
        for a in agents:
            self.save(a)

    def load_all(self) -> list[AgentData]:
        return [self.load(aid) for aid in self.list_ids() if self.load(aid) is not None]
