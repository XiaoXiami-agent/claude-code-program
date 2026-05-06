"""Agent Pool：管理所有角色 Agent 的生命周期。"""

from __future__ import annotations

from .types import AgentStore
from .agent import Agent


class AgentPoolError(Exception):
    """Pool 操作异常基类。"""


class DuplicateAgentError(AgentPoolError):
    """注册重复 ID。"""


class AgentNotFoundError(AgentPoolError):
    """操作不存在的 Agent。"""


class AgentPool:
    """Agent 注册表。不绑定存储后端，存储操作通过 AgentStore 委托。"""

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    # ── 基本 CRUD ────────────────────────────────────

    def register(self, agent: Agent) -> None:
        if agent.id in self._agents:
            raise DuplicateAgentError(f"Agent '{agent.id}' 已存在")
        self._agents[agent.id] = agent

    def remove(self, agent_id: str) -> None:
        if agent_id not in self._agents:
            raise AgentNotFoundError(agent_id)
        del self._agents[agent_id]

    def get(self, agent_id: str) -> Agent:
        if agent_id not in self._agents:
            raise AgentNotFoundError(agent_id)
        return self._agents[agent_id]

    def list_all(self) -> list[Agent]:
        return list(self._agents.values())

    def list_ids(self) -> list[str]:
        return list(self._agents.keys())

    def contains(self, agent_id: str) -> bool:
        return agent_id in self._agents

    # ── 状态修改 ─────────────────────────────────────

    def update_state(self, agent_id: str, patch: dict) -> None:
        """部分更新状态。只改 patch 中含有的字段。支持 extras 扩展字段。"""
        agent = self.get(agent_id)
        for key, value in patch.items():
            agent.state.set(key, value)

    def set_goals(self, agent_id: str, goals: list[str]) -> None:
        self.get(agent_id).goals = list(goals)

    # ── 持久化 ───────────────────────────────────────

    def save_all(self, store: AgentStore) -> None:
        store.save_all([a.to_data() for a in self._agents.values()])

    def load_all(self, store: AgentStore, memory_store=None) -> None:
        """从 AgentStore 加载所有 Agent，替换当前 Pool 内容。"""
        data_list = store.load_all()
        self._agents.clear()
        for data in data_list:
            agent = Agent(
                agent_id=data.id,
                config=data.config,
                state=data.state,
                memory_store=memory_store,
                goals=data.goals,
            )
            self._agents[agent.id] = agent

    # ── 导出（给调度器用）────────────────────────────

    def export_snapshots(self, agent_ids: list[str]) -> dict[str, dict]:
        """导出指定 Agent 的状态快照（深拷贝 dict），供调度器创建临时状态。"""
        result = {}
        for aid in agent_ids:
            if aid in self._agents:
                result[aid] = self._agents[aid].to_dict()["state"]
        return result
