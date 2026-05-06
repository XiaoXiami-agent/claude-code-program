"""调度图构建器。将六个节点组装为 LangGraph StateGraph。"""

from __future__ import annotations

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

from .state import SimulationState, init_state
from . import nodes


def build_scheduler_graph() -> StateGraph:
    """构建并编译调度图。返回的 graph 可直接调用 stream() / invoke()。"""

    graph = StateGraph(SimulationState)

    # 添加节点
    graph.add_node("choose_speaker", nodes.choose_speaker)
    graph.add_node("build_prompt", nodes.build_prompt)
    graph.add_node("call_llm", nodes.call_llm)
    graph.add_node("parse_state", nodes.parse_state)
    graph.add_node("apply_delta", nodes.apply_delta)
    graph.add_node("check_stop", nodes.check_stop)

    # 线性边
    graph.add_edge("choose_speaker", "build_prompt")
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", "parse_state")
    graph.add_edge("parse_state", "apply_delta")
    graph.add_edge("apply_delta", "check_stop")

    # 条件边：停止 → END，否则继续循环
    graph.add_conditional_edges(
        "check_stop",
        lambda state: END if state.get("status") == "finished" else "choose_speaker",
    )

    # 入口
    graph.set_entry_point("choose_speaker")

    # 编译（带内建 checkpoint）
    return graph.compile(checkpointer=InMemorySaver())


def create_initial_state(scene: dict, participants: list[dict]) -> SimulationState:
    """创建初始状态，供 graph.stream() 使用。"""
    return init_state(scene, participants)
