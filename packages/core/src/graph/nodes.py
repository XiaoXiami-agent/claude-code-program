"""调度图节点函数。每个节点接收 state，返回部分更新 dict。"""

from __future__ import annotations

import json
import copy
from langgraph.types import RunnableConfig

from .state import SimulationState


# ─── 节点：选择发言人 ───────────────────────────────

def choose_speaker(state: SimulationState) -> dict:
    """从 speaker_order 中轮询选择下一个发言人，推进回合计数。"""
    order = state["speaker_order"]
    current = state["current_speaker"]
    round_num = state["round"]

    # 找到当前发言人在顺序中的位置
    if current and current in order:
        idx = order.index(current)
        next_idx = idx + 1
    else:
        next_idx = 0

    # 如果一轮结束，回到第一个
    if next_idx >= len(order):
        next_idx = 0
        round_num += 1

    next_speaker = order[next_idx]

    return {
        "current_speaker": next_speaker,
        "round": round_num,
    }


# ─── 节点：组装 Prompt ──────────────────────────────

def build_prompt(state: SimulationState) -> dict:
    """
    为当前发言人构建 Prompt。
    从 state 中的 dict 数据直接组装，避免重建完整 Agent 对象。
    """
    speaker_id = state["current_speaker"]
    scene = state["scene"]
    participants = state["participants"]
    snapshots = state["snapshots"]
    turns = state["turns"]

    # 找到当前发言人数据
    speaker = _find_participant(participants, speaker_id)
    if speaker is None:
        return {"_prompt": ""}

    snap = snapshots.get(speaker_id, speaker.get("state", {}))

    # ── 系统消息 ──
    sys_lines = [
        f"你扮演小说角色：{speaker['config']['name']}",
        f"【性格】{speaker['config']['personality']}",
    ]
    if speaker["config"].get("background"):
        sys_lines.append(f"【背景】{speaker['config']['background']}")
    sys_lines.append(f"【语言风格】{speaker['config']['speaking_style']}")
    sys_lines.append(f"【当前情绪】{snap.get('emotion', '平静')}")
    sys_lines.append(f"【位置】{snap.get('location', '')}")
    if speaker.get("goals"):
        sys_lines.append(f"【当前目标】{'、'.join(speaker['goals'])}")

    rels = snap.get("relationships", {})
    if rels:
        rel_str = "，".join(f"对{k}好感度{v}" for k, v in rels.items())
        sys_lines.append(f"【人际关系】{rel_str}")

    # 场景上下文
    sys_lines.append("")
    sys_lines.append(f"【地点】{scene['config']['location']}")
    sys_lines.append(f"【时间】{scene['config']['time']}")
    sys_lines.append(f"【天气】{scene['config']['weather']}")
    sys_lines.append(f"【氛围】{scene['config']['atmosphere']}")
    if scene["config"].get("background"):
        sys_lines.append(f"【场景描述】{scene['config']['background']}")

    # 其他参与者
    others = [p for p in participants if p["id"] != speaker_id]
    if others:
        sys_lines.append("")
        sys_lines.append("【场景中的其他角色】")
        for o in others:
            o_snap = snapshots.get(o["id"], o.get("state", {}))
            rel_val = rels.get(o["config"]["name"], 0) if rels else 0
            sys_lines.append(
                f"- {o['config']['name']}：{o['config']['personality']}，"
                f"对你的好感度 {rel_val}，当前情绪 {o_snap.get('emotion', '平静')}"
            )

    system_msg = "\n".join(sys_lines)

    # ── 用户消息 ──
    user_lines = []
    if turns:
        user_lines.append("【当前对话历史】")
        for t in turns:
            label = t["speaker_id"]
            user_lines.append(f"{label}: {t['raw_output']}")
        user_lines.append("")

    user_lines.append(
        f"任务：以{speaker['config']['name']}的身份，在当前场景下回应。\n"
        f"输出格式：可附带简短动作描述（用括号标注），然后是对白。\n"
        f"只输出你的回应，不要解释、不要评论、不要说出你是AI。"
    )
    user_msg = "\n".join(user_lines)

    return {
        "_prompt": json.dumps([
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ], ensure_ascii=False)
    }


# ─── 节点：调用 LLM ─────────────────────────────────

def call_llm(state: SimulationState, config: RunnableConfig) -> dict:
    """调 LLM 生成当前发言人的对白。LLM 实例通过 config.configurable.llm 传入。"""
    llm = config.get("configurable", {}).get("llm")
    if llm is None:
        return {"_llm_output": "", "stop_reason": {"type": "manual", "cause": "no_llm"}}

    prompt_json = state.get("_prompt", "")
    if not prompt_json:
        return {"_llm_output": ""}

    messages = json.loads(prompt_json)
    try:
        output = llm.chat(messages)
    except Exception as e:
        return {"_llm_output": f"[ERROR: {e}]"}

    return {"_llm_output": output}


# ─── 节点：解析状态变化 ─────────────────────────────

def parse_state(state: SimulationState, config: RunnableConfig) -> dict:
    """
    二次 LLM 调用：从当前发言中解析状态变化。
    同样通过 config.configurable.llm 传入。
    """
    llm = config.get("configurable", {}).get("llm")
    speaker_id = state["current_speaker"]
    snapshots = state["snapshots"]
    llm_output = state.get("_llm_output", "")

    if not llm_output or llm is None:
        return {"_delta": None}

    snap = snapshots.get(speaker_id, {})

    parse_prompt = [
        {
            "role": "system",
            "content": (
                "你是一个对话分析器。给定角色对话内容，提取状态变化。仅输出 JSON，不要解释。\n"
                '格式: {"emotion":"新情绪或null","relationship_change":{"角色名":数值变化},'
                '"hp_change":0,"exit_intent":false,"reason":"简述"}'
            ),
        },
        {
            "role": "user",
            "content": (
                f"【当前情绪】{snap.get('emotion', '平静')}\n"
                f"【对话内容】\n{llm_output}"
            ),
        },
    ]

    try:
        raw = llm.chat(parse_prompt)
        # 尝试提取 JSON
        delta = _extract_json(raw)
        return {"_delta": delta}
    except Exception:
        return {"_delta": None}


def _extract_json(text: str) -> dict:
    """从 LLM 输出中提取 JSON 对象。"""
    # 去掉可能的 markdown 代码块包裹
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return json.loads(text)


# ─── 节点：应用状态变化 ─────────────────────────────

def apply_delta(state: SimulationState) -> dict:
    """将 parse_state 产出的 delta 应用到临时快照。"""
    speaker_id = state["current_speaker"]
    delta = state.get("_delta")
    speaker_output = state.get("_llm_output", "")

    if delta is None:
        # 无 delta，仍然记录回合
        return {
            "turns": state["turns"] + [{
                "speaker_id": speaker_id,
                "raw_output": speaker_output,
                "delta": None,
            }]
        }

    snapshots = state["snapshots"]
    snap = snapshots.get(speaker_id, {})

    # 应用 HP 变化
    if delta.get("hp_change"):
        snap["hp"] = snap.get("hp", 100) + delta["hp_change"]

    # 应用情绪变化
    if delta.get("emotion") and delta["emotion"] != "null":
        snap["emotion"] = delta["emotion"]

    # 应用好感度变化
    rel_changes = delta.get("relationship_change", {})
    if rel_changes:
        rels = snap.get("relationships", {})
        for name, change in rel_changes.items():
            rels[name] = rels.get(name, 0) + change
        snap["relationships"] = rels

    # 更新快照
    snapshots[speaker_id] = snap

    # 记录回合
    turn = {
        "speaker_id": speaker_id,
        "raw_output": speaker_output,
        "delta": delta,
    }

    return {
        "snapshots": snapshots,
        "turns": state["turns"] + [turn],
    }


# ─── 节点：检查停止条件 ─────────────────────────────

def check_stop(state: SimulationState) -> dict:
    """按优先级检查四种停止条件。"""
    scene = state["scene"]
    config = scene.get("config", {})
    snapshots = state["snapshots"]
    turns = state["turns"]
    round_num = state.get("round", 0)
    max_rounds = config.get("max_rounds", 10)
    failure = config.get("failure_conditions", {})

    # 条件 1: 手动终止（外部中断信号）
    sr = state.get("stop_reason") or {}
    if sr.get("type") == "manual":
        return {}

    # 条件 2: 失败
    hp_threshold = failure.get("hp_threshold")
    emotion_extreme = failure.get("emotion_extreme")
    for agent_id, snap in snapshots.items():
        if hp_threshold is not None and snap.get("hp", 100) <= hp_threshold:
            return {
                "stop_reason": {"type": "failure", "agent_id": agent_id, "cause": "hp_zero"},
                "status": "finished",
            }
        if emotion_extreme and snap.get("emotion") == emotion_extreme:
            return {
                "stop_reason": {"type": "failure", "agent_id": agent_id, "cause": "emotional_breakdown"},
                "status": "finished",
            }

    # 条件 3: 避让退出
    last_turn = turns[-1] if turns else {}
    delta = last_turn.get("delta") or {}
    if delta.get("exit_intent"):
        return {
            "stop_reason": {"type": "withdrawal", "agent_id": last_turn["speaker_id"]},
            "status": "finished",
        }

    # 条件 4: 回合上限（round 从 0 开始，达到 max_rounds 时当前轮次已完成）
    if round_num >= max_rounds:
        return {
            "stop_reason": {"type": "max_rounds"},
            "status": "finished",
        }

    return {}


# ─── 工具函数 ───────────────────────────────────────

def _find_participant(participants: list[dict], agent_id: str) -> dict | None:
    for p in participants:
        if p["id"] == agent_id:
            return p
    return None
