"""M3+M4+M5 场景测试"""

import tempfile, os, json
from packages.core.src.agent import Agent
from packages.core.src.pool import AgentPool, DuplicateAgentError, AgentNotFoundError
from packages.core.src.scene import Scene
from packages.core.src.types import (
    AgentConfig, AgentState, SceneConfig, FailureConditions,
    StateDelta, MemoryItem,
)
from packages.core.src.memory_store import InMemoryMemoryStore
from packages.core.src.storage.file_store import JSONFileAgentStore


def make_agent(id: str, name: str, mem=None) -> Agent:
    return Agent(id,
        AgentConfig(name=name, personality="test_persona", speaking_style="test_style"),
        AgentState(hp=100, emotion="平静"),
        memory_store=mem or InMemoryMemoryStore(),
    )


class TestAgent:
    def test_basic_creation(self):
        a = make_agent("h1", "hero")
        assert a.id == "h1"
        assert a.config.name == "hero"
        assert a.state.hp == 100

    def test_to_prompt_context(self):
        a = make_agent("h1", "hero")
        a.goals = ["save_world"]
        a.state.relationships = {"villain": -50}
        ctx = a.to_prompt_context()
        assert "hero" in ctx
        assert "save_world" in ctx
        assert "villain" in ctx

    def test_apply_state_changes(self):
        a = make_agent("h1", "hero")
        delta = StateDelta(hp_change=-20, emotion="愤怒", relationship_change={"shimei": -10})
        a.apply_state_changes(delta)
        assert a.state.hp == 80
        assert a.state.emotion == "愤怒"
        assert a.state.relationships["shimei"] == -10

    def test_apply_partial_delta(self):
        a = make_agent("h1", "hero")
        a.state.relationships = {"shimei": 50}
        a.apply_state_changes(StateDelta(relationship_change={"shimei": 5}))
        assert a.state.relationships["shimei"] == 55
        assert a.state.hp == 100  # unchanged

    def test_remember_and_recent(self):
        mem = InMemoryMemoryStore()
        a = make_agent("h1", "hero", mem)
        for i in range(10):
            a.remember(MemoryItem(id=str(i), timestamp="D1", content=f"memory_{i}"))
        assert len(a.recent_memories(3)) == 3

    def test_roundtrip_dict(self):
        a = make_agent("h1", "hero")
        a.state.set("extras_key", "extras_val")
        a.goals = ["g1"]
        a2 = Agent.from_dict(a.to_dict())
        assert a2.config.name == "hero"
        assert a2.state.get("extras_key") == "extras_val"
        assert a2.goals == ["g1"]

    def test_dict_without_memory_store(self):
        a = Agent.from_dict({
            "id": "h1",
            "config": {"name": "x", "personality": "x", "speaking_style": "x"},
            "state": {"hp": 50},
        })
        assert a.recent_memories(5) == []


class TestPool:
    def test_register_and_retrieve(self):
        pool = AgentPool()
        pool.register(make_agent("h1", "hero"))
        pool.register(make_agent("h2", "sidekick"))
        assert pool.contains("h1")
        assert len(pool.list_ids()) == 2

    def test_duplicate_raises(self):
        pool = AgentPool()
        pool.register(make_agent("h1", "hero"))
        try:
            pool.register(make_agent("h1", "hero2"))
            assert False, "expected DuplicateAgentError"
        except DuplicateAgentError:
            pass

    def test_not_found_raises(self):
        pool = AgentPool()
        try:
            pool.get("nobody")
            assert False, "expected AgentNotFoundError"
        except AgentNotFoundError:
            pass

    def test_update_state_partial(self):
        pool = AgentPool()
        pool.register(make_agent("h1", "hero"))
        pool.update_state("h1", {"hp": 50, "custom_field": "value"})
        a = pool.get("h1")
        assert a.state.hp == 50
        assert a.state.get("custom_field") == "value"

    def test_remove(self):
        pool = AgentPool()
        pool.register(make_agent("h1", "hero"))
        pool.remove("h1")
        assert not pool.contains("h1")

    def test_set_goals(self):
        pool = AgentPool()
        pool.register(make_agent("h1", "hero"))
        pool.set_goals("h1", ["goal_a", "goal_b"])
        assert pool.get("h1").goals == ["goal_a", "goal_b"]

    def test_save_load_roundtrip(self):
        pool = AgentPool()
        pool.register(make_agent("h1", "hero"))
        pool.register(make_agent("h2", "sidekick"))
        pool.set_goals("h1", ["find_truth"])
        pool.update_state("h1", {"emotion": "angry", "nickname": "wolf"})

        store = InMemoryMemoryStore()  # using InMemoryAgentStore from file_store module
        from packages.core.src.storage.file_store import InMemoryAgentStore
        agent_store = InMemoryAgentStore()
        pool.save_all(agent_store)

        pool2 = AgentPool()
        pool2.load_all(agent_store, memory_store=store)
        assert pool2.contains("h1")
        assert pool2.get("h1").goals == ["find_truth"]
        assert pool2.get("h1").state.emotion == "angry"
        assert pool2.get("h1").state.get("nickname") == "wolf"

    def test_jsonfile_save_load(self):
        pool = AgentPool()
        pool.register(make_agent("h1", "hero"))
        pool.update_state("h1", {"emotion": "sad"})

        with tempfile.TemporaryDirectory() as tmp:
            store = JSONFileAgentStore(os.path.join(tmp, "agents"))
            pool.save_all(store)
            pool2 = AgentPool()
            pool2.load_all(store)
            assert pool2.get("h1").state.emotion == "sad"


class TestScene:
    def test_basic_creation(self):
        s = Scene("s1", "test scene")
        assert s.id == "s1"
        assert s.config.max_rounds == 10

    def test_participants_dedup(self):
        s = Scene("s1", "test")
        s.add_participant("h1")
        s.add_participant("h1")
        s.add_participant("h2")
        assert len(s.participants) == 2

    def test_remove_nonexistent(self):
        s = Scene("s1", "test")
        s.remove_participant("nobody")  # no error
        assert s.participants == []

    def test_shared_context(self):
        s = Scene("s1", "test",
            SceneConfig(location="inn", time="night", weather="rain", atmosphere="tense", background="empty_room"))
        ctx = s.build_shared_context()
        assert "inn" in ctx
        assert "night" in ctx
        assert "rain" in ctx
        assert "empty_room" in ctx

    def test_context_no_background(self):
        s = Scene("s1", "test", SceneConfig(location="field"))
        ctx = s.build_shared_context()
        assert "field" in ctx

    def test_stop_conditions(self):
        s = Scene("s1", "test")
        assert s.max_rounds == 10
        assert s.failure_conditions.hp_threshold is None
        s.set_max_rounds(10)
        s.set_failure_conditions(FailureConditions(hp_threshold=0))
        assert s.failure_conditions.hp_threshold == 0

    def test_roundtrip_dict(self):
        s = Scene("s1", "inn fight",
            SceneConfig(location="inn", max_rounds=6,
                       failure_conditions=FailureConditions(hp_threshold=0, emotion_extreme="崩溃")))
        s.add_participant("h1")
        s.add_participant("h2")
        s2 = Scene.from_dict(s.to_dict())
        assert s2.name == "inn fight"
        assert s2.config.max_rounds == 6
        assert s2.failure_conditions.emotion_extreme == "崩溃"
        assert s2.participants == ["h1", "h2"]
