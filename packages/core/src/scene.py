"""Scene 容器：共享时空上下文 + 参与者 + 停止条件配置。"""

from __future__ import annotations

from .types import SceneConfig, FailureConditions


class Scene:
    """场景 = 共享信息 + 参与者列表 + 停止条件。"""

    def __init__(self, scene_id: str, name: str, config: SceneConfig | None = None) -> None:
        self.id = scene_id
        self.name = name
        self.config = config or SceneConfig()
        self._participants: list[str] = []

    # ── 参与者管理 ───────────────────────────────────

    @property
    def participants(self) -> list[str]:
        return list(self._participants)

    def add_participant(self, agent_id: str) -> None:
        if agent_id not in self._participants:
            self._participants.append(agent_id)

    def remove_participant(self, agent_id: str) -> None:
        if agent_id in self._participants:
            self._participants.remove(agent_id)

    def has_participant(self, agent_id: str) -> bool:
        return agent_id in self._participants

    # ── 停止条件 ─────────────────────────────────────

    @property
    def max_rounds(self) -> int:
        return self.config.max_rounds

    @property
    def failure_conditions(self) -> FailureConditions:
        return self.config.failure_conditions

    def set_max_rounds(self, n: int) -> None:
        self.config.max_rounds = n

    def set_failure_conditions(self, fc: FailureConditions) -> None:
        self.config.failure_conditions = fc

    # ── Prompt 上下文 ────────────────────────────────

    def build_shared_context(self) -> str:
        """生成共享上下文文本块，注入给所有参与 Agent 的 Prompt。"""
        c = self.config
        parts = [
            f"【地点】{c.location}",
            f"【时间】{c.time}",
            f"【天气】{c.weather}",
            f"【氛围】{c.atmosphere}",
        ]
        if c.background:
            parts.append(f"【场景描述】{c.background}")
        return "\n".join(parts)

    # ── 序列化 ───────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "config": {
                "location": self.config.location,
                "time": self.config.time,
                "weather": self.config.weather,
                "atmosphere": self.config.atmosphere,
                "background": self.config.background,
                "max_rounds": self.config.max_rounds,
                "failure_conditions": {
                    "hp_threshold": self.config.failure_conditions.hp_threshold,
                    "emotion_extreme": self.config.failure_conditions.emotion_extreme,
                    "affinity_threshold": self.config.failure_conditions.affinity_threshold,
                },
            },
            "participants": self._participants,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Scene:
        cfg = data.get("config", {})
        fc_raw = cfg.get("failure_conditions", {})
        failure_conditions = FailureConditions(
            hp_threshold=fc_raw.get("hp_threshold"),
            emotion_extreme=fc_raw.get("emotion_extreme"),
            affinity_threshold=fc_raw.get("affinity_threshold"),
        )
        config = SceneConfig(
            location=cfg.get("location", ""),
            time=cfg.get("time", ""),
            weather=cfg.get("weather", ""),
            atmosphere=cfg.get("atmosphere", ""),
            background=cfg.get("background", ""),
            max_rounds=cfg.get("max_rounds", 10),
            failure_conditions=failure_conditions,
        )
        scene = cls(scene_id=data["id"], name=data["name"], config=config)
        scene._participants = data.get("participants", [])
        return scene
