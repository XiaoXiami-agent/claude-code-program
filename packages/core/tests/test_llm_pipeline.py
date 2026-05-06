"""S2 LLM pipeline 场景测试"""

import os
from packages.core.src.llm.factory import create_llm, DeepSeekAdapter, CozeAdapter
from packages.core.src.agent import Agent
from packages.core.src.scene import Scene
from packages.core.src.prompt_builder import PromptBuilder
from packages.core.src.types import (
    AgentConfig, AgentState, SceneConfig, TurnResult, MemoryItem,
)
from packages.core.src.memory_store import InMemoryMemoryStore


class TestLLMFactory:
    def test_create_deepseek(self):
        llm = create_llm("deepseek", {"api_key": "test_key"})
        assert isinstance(llm, DeepSeekAdapter)

    def test_deepseek_custom_config(self):
        llm = create_llm("deepseek", {
            "api_key": "k", "model": "deepseek-chat",
            "temperature": 0.5, "max_tokens": 512,
        })
        assert llm._temperature == 0.5
        assert llm._max_tokens == 512

    def test_deepseek_default_config(self):
        llm = create_llm("deepseek", {"api_key": "k"})
        assert llm._model == "deepseek-chat"

    def test_unknown_provider_raises(self):
        try:
            create_llm("unknown")
            assert False, "expected ValueError"
        except ValueError:
            pass


class TestPromptBuilder:
    def _make_setup(self):
        mem = InMemoryMemoryStore()
        agent = Agent("h1",
            AgentConfig(name="hero", personality="bold", speaking_style="short"),
            AgentState(hp=80, emotion="angry", location="inn", relationships={"sidekick": 60}),
            memory_store=mem, goals=["find_truth"])
        scene = Scene("s1", "test_scene",
            SceneConfig(location="inn", time="night", weather="rain", atmosphere="tense"))
        other = Agent("h2",
            AgentConfig(name="sidekick", personality="timid", speaking_style="soft"),
            AgentState(hp=90, emotion="scared"))
        builder = PromptBuilder()
        return builder, agent, scene, other

    def test_basic_messages_structure(self):
        builder, agent, scene, other = self._make_setup()
        msgs = builder.build_messages(agent, scene, [], [other])
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_system_contains_agent_info(self):
        builder, agent, scene, other = self._make_setup()
        msgs = builder.build_messages(agent, scene, [], [other])
        sys = msgs[0]["content"]
        assert "hero" in sys
        assert "bold" in sys
        assert "angry" in sys
        assert "inn" in sys

    def test_system_contains_scene_context(self):
        builder, agent, scene, other = self._make_setup()
        msgs = builder.build_messages(agent, scene, [], [other])
        sys = msgs[0]["content"]
        assert "night" in sys
        assert "rain" in sys
        assert "tense" in sys

    def test_system_contains_other_participants(self):
        builder, agent, scene, other = self._make_setup()
        msgs = builder.build_messages(agent, scene, [], [other])
        sys = msgs[0]["content"]
        assert "sidekick" in sys
        assert "好感度 60" in sys or "60" in sys

    def test_user_contains_self_instruction(self):
        builder, agent, scene, other = self._make_setup()
        msgs = builder.build_messages(agent, scene, [], [other])
        usr = msgs[1]["content"]
        assert "hero" in usr  # instruction mentions self name

    def test_no_other_participants(self):
        builder, agent, scene, _ = self._make_setup()
        msgs = builder.build_messages(agent, scene, [], [])
        sys = msgs[0]["content"]
        assert "其他角色" not in sys

    def test_with_dialogue_history(self):
        builder, agent, scene, other = self._make_setup()
        history = [
            TurnResult(speaker_id="sidekick", raw_output="Why did you lie?"),
            TurnResult(speaker_id="hero", raw_output="I had no choice."),
        ]
        msgs = builder.build_messages(agent, scene, history, [other])
        usr = msgs[1]["content"]
        assert "Why did you lie?" in usr
        assert "I had no choice." in usr

    def test_state_parse_prompt(self):
        builder, agent, scene, _ = self._make_setup()
        msgs = builder.build_state_parse_prompt(agent, "I feel sad...")
        assert msgs[0]["role"] == "system"
        assert "JSON" in msgs[0]["content"]
        assert "I feel sad" in msgs[1]["content"]
