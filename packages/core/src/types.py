"""核心类型定义。本模块零依赖，仅使用 Python 标准库。"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import StrEnum
from typing import Protocol, runtime_checkable


# ─── 枚举 ───────────────────────────────────────────

class InteractionType(StrEnum):
    DIALOGUE = "dialogue"          # 言语交互
    COMBAT = "combat"              # 武力冲突
    MONOLOGUE = "monologue"        # 内心独白
    FREE = "free"                  # 自由交互


class StopType(StrEnum):
    MANUAL = "manual"              # 手动终止
    FAILURE = "failure"            # 失败（HP归零/情绪崩溃）
    WITHDRAWAL = "withdrawal"      # 避让退出
    MAX_ROUNDS = "max_rounds"      # 回合上限


# ─── 记忆 ───────────────────────────────────────────

@dataclass
class MemoryItem:
    id: str
    timestamp: str                # 故事时间，如 "第12天戌时"
    content: str                  # 自然语言描述
    tags: list[str] = field(default_factory=list)


# ─── 角色状态 ───────────────────────────────────────

@dataclass
class AgentConfig:
    """不可变属性：定义了"这个角色是谁"."""
    name: str
    personality: str              # 性格描述
    speaking_style: str           # 语言风格
    background: str = ""          # 背景简述（可选）


@dataclass
class AgentState:
    """可变状态：交互过程中会被修改的数值。extras 槽用于扩展自定义字段。"""
    hp: int = 100
    mp: int = 100
    emotion: str = "平静"
    location: str = ""
    relationships: dict[str, int] = field(default_factory=dict)  # {角色名: 好感度}
    inventory: list[str] = field(default_factory=list)
    buffs: list[str] = field(default_factory=list)
    extras: dict = field(default_factory=dict)   # 扩展槽：自定义字段加这里

    def get(self, key: str, default=None):
        """优先从已知字段取值，fallback 到 extras。方便加自定义字段后不改调用代码。"""
        known = {
            "hp": self.hp, "mp": self.mp, "emotion": self.emotion,
            "location": self.location,
        }
        if key in known:
            return known[key]
        if key == "relationships":
            return self.relationships
        if key == "inventory":
            return self.inventory
        if key == "buffs":
            return self.buffs
        return self.extras.get(key, default)

    def set(self, key: str, value) -> None:
        known_map = {
            "hp": "hp", "mp": "mp", "emotion": "emotion",
            "location": "location",
        }
        if key in known_map:
            setattr(self, known_map[key], value)
        elif key == "relationships":
            self.relationships = value
        elif key == "inventory":
            self.inventory = value
        elif key == "buffs":
            self.buffs = value
        else:
            self.extras[key] = value


@dataclass
class AgentData:
    """Agent 的完整持久化视图（不含记忆，记忆走 MemoryStore）。"""
    id: str
    config: AgentConfig
    state: AgentState
    goals: list[str] = field(default_factory=list)


# ─── 场景 ───────────────────────────────────────────

@dataclass
class FailureConditions:
    hp_threshold: int | None = None       # HP ≤ 此值判定失败
    emotion_extreme: str | None = None    # 情绪等于此值判定崩溃
    affinity_threshold: int | None = None # 好感度 ≤ 此值判定关系破裂


@dataclass
class SceneConfig:
    location: str = ""
    time: str = ""
    weather: str = ""
    atmosphere: str = ""
    background: str = ""
    max_rounds: int = 10
    failure_conditions: FailureConditions = field(default_factory=FailureConditions)


# ─── 调度 ───────────────────────────────────────────

@dataclass
class StateDelta:
    """单次交互的状态变化。由 StateResolver 解析 LLM 输出产生。"""
    emotion: str | None = None
    relationship_change: dict[str, int] = field(default_factory=dict)  # {角色名: 变化量}
    hp_change: int = 0
    mp_change: int = 0
    exit_intent: bool = False          # 是否表达退出意图（停止条件 3）
    reason: str = ""                   # 变化原因简述
    extras_delta: dict = field(default_factory=dict)  # 扩展字段变化


@dataclass
class TurnResult:
    speaker_id: str
    raw_output: str                   # Agent 原始输出（对话文本）
    delta: StateDelta | None = None   # parse_state 节点填充


@dataclass
class StopReason:
    type: StopType
    agent_id: str = ""
    cause: str = ""


# ─── 存储接口（Protocol）───────────────────────────

@runtime_checkable
class MemoryStore(Protocol):
    """短时记忆存取。与 Agent 解耦，可替换为 InMemory / JSONFile / 未来 IndexedDB。"""

    def add(self, agent_id: str, item: MemoryItem) -> None: ...
    def get_recent(self, agent_id: str, n: int) -> list[MemoryItem]: ...
    def get_all(self, agent_id: str) -> list[MemoryItem]: ...


@runtime_checkable
class AgentStore(Protocol):
    """Agent 持久化。Pool 通过此接口存取，不关心底层存储介质。"""

    def save(self, agent_data: AgentData) -> None: ...
    def load(self, agent_id: str) -> AgentData | None: ...
    def list_ids(self) -> list[str]: ...
    def delete(self, agent_id: str) -> None: ...
    def save_all(self, agents: list[AgentData]) -> None: ...
    def load_all(self) -> list[AgentData]: ...


@runtime_checkable
class SceneStore(Protocol):
    """Scene 持久化。"""

    def save(self, scene_data: dict) -> None: ...
    def load(self, scene_id: str) -> dict | None: ...
    def list_ids(self) -> list[str]: ...
    def delete(self, scene_id: str) -> None: ...


# ─── 工具函数 ───────────────────────────────────────

def state_to_dict(state: AgentState) -> dict:
    """将 AgentState 展平为单一 dict（known fields + extras），方便 Prompt 注入。"""
    d = {
        "hp": state.hp,
        "mp": state.mp,
        "emotion": state.emotion,
        "location": state.location,
        "relationships": state.relationships,
        "inventory": state.inventory,
        "buffs": state.buffs,
    }
    d.update(state.extras)
    return d
