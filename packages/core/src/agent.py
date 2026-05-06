"""Agent 类：角色的数字分身。持有不可变配置 + 可变状态，记忆委托给 MemoryStore。"""

from __future__ import annotations

from .types import (
    AgentConfig,
    AgentState,
    AgentData,
    MemoryItem,
    MemoryStore,
    StateDelta,
)


class Agent:
    """角色 Agent。不感知存储、不感知 LLM，只管理自己的数据和记忆。"""

    def __init__(
        self,
        agent_id: str,
        config: AgentConfig,
        state: AgentState | None = None,
        memory_store: MemoryStore | None = None,
        goals: list[str] | None = None,
    ) -> None:
        self.id = agent_id
        self.config = config
        self.state = state or AgentState()
        self._memory = memory_store
        self.goals = goals or []

    # ── 记忆（委托给 MemoryStore）────────────────────

    def remember(self, item: MemoryItem) -> None:
        if self._memory is not None:
            self._memory.add(self.id, item)

    def recent_memories(self, n: int = 5) -> list[MemoryItem]:
        if self._memory is None:
            return []
        return self._memory.get_recent(self.id, n)

    # ── 状态变更 ─────────────────────────────────────

    def apply_state_changes(self, delta: StateDelta) -> None:
        """就地应用 StateDelta。忽略值为 None 的字段（无变化）。"""
        s = self.state
        s.hp += delta.hp_change
        s.mp += delta.mp_change

        if delta.emotion is not None:
            s.emotion = delta.emotion

        for name, change in delta.relationship_change.items():
            current = s.relationships.get(name, 0)
            s.relationships[name] = current + change

        for key, value in delta.extras_delta.items():
            s.extras[key] = value

    # ── Prompt 上下文 ────────────────────────────────

    def to_prompt_context(self) -> str:
        """将自身信息编译为 Prompt 文本块，供 build_prompt 节点使用。"""
        parts = [
            f"【角色】{self.config.name}",
            f"【性格】{self.config.personality}",
        ]
        if self.config.background:
            parts.append(f"【背景】{self.config.background}")
        parts.append(f"【语言风格】{self.config.speaking_style}")
        parts.append(f"【当前情绪】{self.state.emotion}")
        parts.append(f"【位置】{self.state.location}")

        if self.goals:
            parts.append(f"【当前目标】{'、'.join(self.goals)}")

        if self.state.relationships:
            rels = "，".join(
                f"对{name}好感度{val}"
                for name, val in self.state.relationships.items()
            )
            parts.append(f"【人际关系】{rels}")

        return "\n".join(parts)

    # ── 序列化 ───────────────────────────────────────

    def to_data(self) -> AgentData:
        return AgentData(id=self.id, config=self.config, state=self.state, goals=self.goals)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "config": {
                "name": self.config.name,
                "personality": self.config.personality,
                "speaking_style": self.config.speaking_style,
                "background": self.config.background,
            },
            "state": {
                "hp": self.state.hp,
                "mp": self.state.mp,
                "emotion": self.state.emotion,
                "location": self.state.location,
                "relationships": self.state.relationships,
                "inventory": self.state.inventory,
                "buffs": self.state.buffs,
                **self.state.extras,
            },
            "goals": self.goals,
        }

    @classmethod
    def from_dict(cls, data: dict, memory_store: MemoryStore | None = None) -> Agent:
        cfg = data["config"]
        st = data["state"]

        # 分离 known fields 和 extras
        known = {"hp", "mp", "emotion", "location", "relationships", "inventory", "buffs"}
        extras = {k: v for k, v in st.items() if k not in known}

        state = AgentState(
            hp=st.get("hp", 100),
            mp=st.get("mp", 100),
            emotion=st.get("emotion", "平静"),
            location=st.get("location", ""),
            relationships=st.get("relationships", {}),
            inventory=st.get("inventory", []),
            buffs=st.get("buffs", []),
            extras=extras,
        )

        return cls(
            agent_id=data["id"],
            config=AgentConfig(
                name=cfg["name"],
                personality=cfg["personality"],
                speaking_style=cfg["speaking_style"],
                background=cfg.get("background", ""),
            ),
            state=state,
            memory_store=memory_store,
            goals=data.get("goals", []),
        )
