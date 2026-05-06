"""M1 types.py 场景测试：正常 / 边界 / 异常"""

from packages.core.src.types import (
    AgentConfig, AgentState, AgentData,
    SceneConfig, FailureConditions,
    InteractionType, StopType,
    MemoryItem, StateDelta, TurnResult, StopReason,
    state_to_dict, MemoryStore, AgentStore, SceneStore,
)


class TestAgentState:
    def test_defaults(self):
        s = AgentState()
        assert s.hp == 100
        assert s.emotion == "平静"
        assert s.extras == {}

    def test_get_set_known_fields(self):
        s = AgentState(hp=80, emotion="愤怒")
        assert s.get("hp") == 80
        s.set("hp", 50)
        assert s.hp == 50
        s.set("emotion", "悲伤")
        assert s.emotion == "悲伤"

    def test_extras_dynamic_fields(self):
        s = AgentState()
        s.set("称号", "独行侠")
        s.set("内力属性", "火")
        assert s.get("称号") == "独行侠"
        assert s.get("内力属性") == "火"
        assert s.extras == {"称号": "独行侠", "内力属性": "火"}

    def test_get_missing_key_returns_default(self):
        s = AgentState()
        assert s.get("不存在的字段", 42) == 42

    def test_complex_relationships(self):
        s = AgentState(relationships={"甲": 50, "乙": -30})
        s.set("relationships", {"甲": 60, "乙": -20, "丙": 0})
        assert s.relationships["甲"] == 60
        assert "丙" in s.relationships


class TestSerialization:
    def test_agent_config_roundtrip(self):
        cfg = AgentConfig(name="林逸风", personality="冲动", speaking_style="短句", background="孤儿")
        data = {
            "name": cfg.name, "personality": cfg.personality,
            "speaking_style": cfg.speaking_style, "background": cfg.background,
        }
        cfg2 = AgentConfig(**data)
        assert cfg2.name == "林逸风"
        assert cfg2.background == "孤儿"

    def test_state_to_dict_includes_extras(self):
        s = AgentState(hp=80, emotion="愤怒")
        s.set("称号", "剑客")
        d = state_to_dict(s)
        assert d["hp"] == 80
        assert d["称号"] == "剑客"

    def test_scene_config_defaults(self):
        cfg = SceneConfig()
        assert cfg.max_rounds == 10

    def test_failure_conditions_optional(self):
        fc = FailureConditions()
        assert fc.hp_threshold is None
        assert fc.emotion_extreme is None

    def test_agent_data_goals_default(self):
        ad = AgentData(id="test", config=AgentConfig(name="x", personality="x", speaking_style="x"), state=AgentState())
        assert ad.goals == []


class TestEnums:
    def test_interaction_types(self):
        assert InteractionType.DIALOGUE == "dialogue"
        assert InteractionType.COMBAT == "combat"

    def test_stop_types(self):
        assert StopType.MANUAL == "manual"
        assert StopType.WITHDRAWAL == "withdrawal"

    def test_enum_in_dataclass(self):
        sr = StopReason(type=StopType.MANUAL, agent_id="hero_01")
        assert sr.type == "manual"


class TestProtocols:
    def test_memory_store_protocol(self):
        class Impl:
            def add(self, agent_id, item): pass
            def get_recent(self, agent_id, n): return []
            def get_all(self, agent_id): return []
        assert isinstance(Impl(), MemoryStore)

    def test_agent_store_protocol(self):
        class Impl:
            def save(self, agent_data): pass
            def load(self, agent_id): return None
            def list_ids(self): return []
            def delete(self, agent_id): pass
            def save_all(self, agents): pass
            def load_all(self): return []
        assert isinstance(Impl(), AgentStore)
