"""Prompt 组装。将 Agent 状态 + Scene 上下文 + 对话历史编译为 LLM 输入。"""

from __future__ import annotations

from .types import SceneConfig, TurnResult
from .agent import Agent
from .scene import Scene


class PromptBuilder:
    """构建角色 Agent 的 Prompt。可被子类化以支持环境 Agent 等变体。"""

    def __init__(
        self,
        system_template: str | None = None,
        turn_template: str | None = None,
    ) -> None:
        self._system_tmpl = system_template or self._default_system()
        self._turn_tmpl = turn_template or "{speaker}: {content}"

    # ── 公共 API ─────────────────────────────────────

    def build_messages(
        self,
        agent: Agent,
        scene: Scene,
        dialogue_history: list[TurnResult] | None = None,
        other_participants: list[Agent] | None = None,
        memory_count: int = 5,
    ) -> list[dict]:
        """
        构建完整的消息列表，可直接传给 LLMAdapter.chat()。

        返回格式:
        [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},    # 对话历史 + 当前轮次提示
        ]
        """
        system = self._build_system(agent, scene, other_participants or [], memory_count)
        user = self._build_user(agent, dialogue_history or [])
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def build_state_parse_prompt(
        self,
        agent: Agent,
        raw_output: str,
    ) -> list[dict]:
        """
        构建状态解析的 Prompt（给 StateResolver 用，S3 阶段实现）。
        这里只提供基本结构，具体逻辑见 state_resolver.py。
        """
        system = (
            "你是一个对话分析器。给定角色对话内容，提取状态变化。"
            "仅输出 JSON，不要解释。\n"
            "格式: {\"emotion\": \"...\", \"relationship_change\": {\"角色名\": 数值}, "
            "\"hp_change\": 0, \"exit_intent\": false, \"reason\": \"...\"}"
        )
        user = (
            f"【当前情绪】{agent.state.emotion}\n"
            f"【对话内容】\n{raw_output}"
        )
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    # ── 内部 ─────────────────────────────────────────

    def _build_system(
        self,
        agent: Agent,
        scene: Scene,
        others: list[Agent],
        memory_count: int,
    ) -> str:
        parts = []

        # 角色设定
        parts.append(agent.to_prompt_context())

        # 场景共享信息
        parts.append("")
        parts.append(scene.build_shared_context())

        # 场景中其他角色
        if others:
            parts.append("")
            parts.append("【场景中的其他角色】")
            for other in others:
                rel = agent.state.relationships.get(other.config.name, 0)
                parts.append(
                    f"- {other.config.name}：{other.config.personality}，"
                    f"对你的好感度 {rel}，当前情绪 {other.state.emotion}"
                )

        # 近期记忆
        memories = agent.recent_memories(memory_count)
        if memories:
            parts.append("")
            parts.append("【你的近期记忆】")
            for m in memories:
                parts.append(f"- [{m.timestamp}] {m.content}")

        return "\n".join(parts)

    def _build_user(
        self,
        agent: Agent,
        history: list[TurnResult],
    ) -> str:
        parts = []

        # 对话历史
        if history:
            parts.append("【当前对话历史】")
            for turn in history:
                label = turn.speaker_id
                parts.append(f"{label}: {turn.raw_output}")
            parts.append("")

        # 当前回合提示
        parts.append(
            f"任务：以{agent.config.name}的身份，在当前场景下回应。\n"
            f"输出格式：可附带简短动作描述（用括号标注），然后是对白。\n"
            f"只输出你的回应，不要解释、不要评论、不要说出你是AI。"
        )
        return "\n".join(parts)

    @staticmethod
    def _default_system() -> str:
        return (
            "你是一个小说角色的扮演者。严格遵循角色设定，"
            "根据性格、情绪、人际关系来回应对白。"
        )
