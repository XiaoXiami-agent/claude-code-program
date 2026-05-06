"""SimulationState — LangGraph 调度图的状态定义。"""

from __future__ import annotations

from typing import TypedDict


class SimulationState(TypedDict, total=False):
    # ── 输入（不可变） ──
    scene: dict                    # Scene.to_dict()
    participants: list[dict]       # [Agent.to_dict(), ...]

    # ── 运行时 ──
    snapshots: dict[str, dict]     # agent_id -> 临时状态快照
    speaker_order: list[str]       # 发言顺序（agent_id 列表）
    turns: list[dict]              # [{speaker_id, raw_output, delta}]
    current_speaker: str           # 当前发言人 ID
    round: int                     # 当前回合数

    # ── 控制 ──
    stop_reason: dict | None       # {type, agent_id, cause}
    status: str                    # "running" | "finished"

    # ── 内部传递（节点间用 _ 前缀标记） ──
    _prompt: str                   # build_prompt 产出
    _llm_output: str               # call_llm 产出
    _delta: dict | None            # parse_state 产出


def init_state(scene: dict, participants: list[dict]) -> SimulationState:
    """创建初始状态（调度图入口）。深拷贝参与者状态作为临时快照。"""
    import copy
    snapshots = {}
    for p in participants:
        snapshots[p["id"]] = copy.deepcopy(p["state"])

    # 默认发言顺序：按参与者列表顺序
    speaker_order = [p["id"] for p in participants]

    return SimulationState(
        scene=scene,
        participants=participants,
        snapshots=snapshots,
        speaker_order=speaker_order,
        turns=[],
        current_speaker="",
        round=0,
        stop_reason=None,
        status="running",
        _prompt="",
        _llm_output="",
        _delta=None,
    )
