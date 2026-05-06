"""S3 调度图场景测试"""

import json
import copy
from packages.core.src.graph.state import SimulationState, init_state
from packages.core.src.graph.nodes import (
    choose_speaker,
    build_prompt,
    apply_delta,
    check_stop,
)
from packages.core.src.graph.scheduler import build_scheduler_graph, create_initial_state


def _make_base_state(**overrides) -> SimulationState:
    scene = {
        "id": "s1", "name": "test",
        "config": {
            "location": "inn", "time": "night", "weather": "rain",
            "atmosphere": "tense", "max_rounds": 10,
            "failure_conditions": {"hp_threshold": 0, "emotion_extreme": "崩溃"},
        },
    }
    participants = [
        {
            "id": "a1", "config": {"name": "hero", "personality": "bold",
             "speaking_style": "short", "background": "orphan"},
            "state": {"hp": 80, "mp": 80, "emotion": "angry", "location": "inn",
                      "relationships": {"sidekick": 60}, "inventory": [], "buffs": []},
            "goals": ["confront"],
        },
        {
            "id": "a2", "config": {"name": "sidekick", "personality": "timid",
             "speaking_style": "soft", "background": ""},
            "state": {"hp": 90, "mp": 100, "emotion": "worried", "location": "inn",
                      "relationships": {"hero": 80}, "inventory": [], "buffs": []},
            "goals": ["help"],
        },
    ]
    base = init_state(scene, participants)
    base.update(overrides)
    return base


class TestChooseSpeaker:
    def test_first_speaker(self):
        state = _make_base_state()
        result = choose_speaker(state)
        assert result["current_speaker"] == "a1"
        assert result["round"] == 0

    def test_second_speaker(self):
        state = _make_base_state(current_speaker="a1")
        result = choose_speaker(state)
        assert result["current_speaker"] == "a2"

    def test_wraps_around(self):
        state = _make_base_state(current_speaker="a2", round=0)
        result = choose_speaker(state)
        assert result["current_speaker"] == "a1"
        assert result["round"] == 1


class TestBuildPrompt:
    def test_contains_agent_info(self):
        state = _make_base_state(current_speaker="a1")
        result = build_prompt(state)
        msgs = json.loads(result["_prompt"])
        sys_content = msgs[0]["content"]
        assert "hero" in sys_content
        assert "bold" in sys_content

    def test_contains_scene_info(self):
        state = _make_base_state(current_speaker="a1")
        result = build_prompt(state)
        msgs = json.loads(result["_prompt"])
        sys_content = msgs[0]["content"]
        assert "inn" in sys_content
        assert "rain" in sys_content

    def test_contains_other_participants(self):
        state = _make_base_state(current_speaker="a1")
        result = build_prompt(state)
        msgs = json.loads(result["_prompt"])
        sys_content = msgs[0]["content"]
        assert "sidekick" in sys_content

    def test_user_message_contains_task(self):
        state = _make_base_state(current_speaker="a1")
        result = build_prompt(state)
        msgs = json.loads(result["_prompt"])
        usr = msgs[1]["content"]
        assert "hero" in usr

    def test_includes_dialogue_history(self):
        state = _make_base_state(
            current_speaker="a2",
            turns=[
                {"speaker_id": "a1", "raw_output": "Why did you lie?", "delta": None},
            ]
        )
        result = build_prompt(state)
        msgs = json.loads(result["_prompt"])
        usr = msgs[1]["content"]
        assert "Why did you lie?" in usr


class TestApplyDelta:
    def test_applies_hp_change(self):
        state = _make_base_state(current_speaker="a1")
        state["_delta"] = {"hp_change": -20, "emotion": None, "relationship_change": {}}
        state["_llm_output"] = "test response"
        result = apply_delta(state)
        assert result["snapshots"]["a1"]["hp"] == 60

    def test_applies_emotion(self):
        state = _make_base_state(current_speaker="a1")
        state["_delta"] = {"hp_change": 0, "emotion": "sad", "relationship_change": {}}
        state["_llm_output"] = "test"
        result = apply_delta(state)
        assert result["snapshots"]["a1"]["emotion"] == "sad"

    def test_applies_relationship_change(self):
        state = _make_base_state(current_speaker="a1")
        state["_delta"] = {"hp_change": 0, "emotion": None,
                          "relationship_change": {"sidekick": -10}}
        state["_llm_output"] = "test"
        result = apply_delta(state)
        assert result["snapshots"]["a1"]["relationships"]["sidekick"] == 50

    def test_records_turn(self):
        state = _make_base_state(current_speaker="a1")
        state["_delta"] = {"hp_change": 0, "emotion": "calm", "relationship_change": {}}
        state["_llm_output"] = "hello world"
        result = apply_delta(state)
        assert len(result["turns"]) == 1
        assert result["turns"][0]["speaker_id"] == "a1"
        assert result["turns"][0]["raw_output"] == "hello world"

    def test_null_delta_still_records(self):
        state = _make_base_state(current_speaker="a1")
        state["_delta"] = None
        state["_llm_output"] = "no parse"
        result = apply_delta(state)
        assert len(result["turns"]) == 1
        assert result["turns"][0]["delta"] is None


class TestCheckStop:
    def test_no_stop(self):
        state = _make_base_state(round=2)
        result = check_stop(state)
        assert result == {}

    def test_max_rounds(self):
        state = _make_base_state(round=10)
        result = check_stop(state)
        assert result["stop_reason"]["type"] == "max_rounds"

    def test_hp_zero_failure(self):
        state = _make_base_state(round=2)
        state["snapshots"]["a1"]["hp"] = -5
        result = check_stop(state)
        assert result["stop_reason"]["type"] == "failure"
        assert result["stop_reason"]["agent_id"] == "a1"

    def test_emotional_breakdown(self):
        state = _make_base_state(round=2)
        state["snapshots"]["a2"]["emotion"] = "崩溃"
        result = check_stop(state)
        assert result["stop_reason"]["type"] == "failure"
        assert result["stop_reason"]["cause"] == "emotional_breakdown"

    def test_withdrawal(self):
        state = _make_base_state(round=2)
        state["turns"] = [{
            "speaker_id": "a1",
            "raw_output": "I'm leaving...",
            "delta": {"exit_intent": True, "emotion": "done", "relationship_change": {}},
        }]
        result = check_stop(state)
        assert result["stop_reason"]["type"] == "withdrawal"


class TestGraphCompilation:
    def test_build_and_compile(self):
        graph = build_scheduler_graph()
        assert graph is not None

    def test_create_initial_state(self):
        state = create_initial_state(
            scene={"id": "s1", "config": {"location": "x", "time": "x", "weather": "x",
                    "atmosphere": "x", "max_rounds": 5,
                    "failure_conditions": {}}},
            participants=[
                {"id": "a1", "config": {"name": "x", "personality": "x", "speaking_style": "x"},
                 "state": {"hp": 100, "emotion": "calm", "location": "", "relationships": {},
                           "inventory": [], "buffs": []}, "goals": []},
            ],
        )
        assert state["snapshots"]["a1"]["hp"] == 100
        assert state["status"] == "running"

    def test_snapshots_are_deep_copies(self):
        state = create_initial_state(
            scene={"id": "s1", "config": {"location": "x", "time": "x", "weather": "x",
                    "atmosphere": "x", "max_rounds": 5, "failure_conditions": {}}},
            participants=[
                {"id": "a1", "config": {"name": "x", "personality": "x", "speaking_style": "x"},
                 "state": {"hp": 100, "emotion": "calm", "location": "", "relationships": {},
                           "inventory": [], "buffs": []}, "goals": []},
            ],
        )
        state["snapshots"]["a1"]["hp"] = 50
        assert state["snapshots"]["a1"]["hp"] == 50
        # Re-create and verify it's not mutated
        state2 = create_initial_state(
            scene={"id": "s1", "config": {"location": "x", "time": "x", "weather": "x",
                    "atmosphere": "x", "max_rounds": 5, "failure_conditions": {}}},
            participants=[
                {"id": "a1", "config": {"name": "x", "personality": "x", "speaking_style": "x"},
                 "state": {"hp": 100, "emotion": "calm", "location": "", "relationships": {},
                           "inventory": [], "buffs": []}, "goals": []},
            ],
        )
        assert state2["snapshots"]["a1"]["hp"] == 100
