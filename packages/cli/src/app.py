"""Agent Pool CLI — 交互式命令行原型。演示 Agent 创建、Scene 管理、对话模拟全流程。"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# 添加项目根目录到 Python path
_core_src = Path(__file__).parent.parent.parent / "core" / "src"
if str(_core_src) not in sys.path:
    sys.path.insert(0, str(_core_src.parent.parent.parent))

from packages.core.src.agent import Agent
from packages.core.src.pool import AgentPool, DuplicateAgentError, AgentNotFoundError
from packages.core.src.scene import Scene
from packages.core.src.types import (
    AgentConfig, AgentState, SceneConfig, FailureConditions,
    InteractionType, MemoryItem,
)
from packages.core.src.memory_store import InMemoryMemoryStore
from packages.core.src.graph.scheduler import build_scheduler_graph, create_initial_state
from packages.core.src.llm.factory import create_llm


# ─── 全局状态 ───────────────────────────────────────

DATA_DIR = Path("project-data")
AGENTS_DIR = DATA_DIR / "agents"
SCENES_DIR = DATA_DIR / "scenes"

pool = AgentPool()
memory_store = InMemoryMemoryStore()
scenes: dict[str, Scene] = {}
llm = None


# ─── 工具函数 ───────────────────────────────────────

def _init_dirs() -> None:
    AGENTS_DIR.mkdir(parents=True, exist_ok=True)
    SCENES_DIR.mkdir(parents=True, exist_ok=True)


def _input(prompt: str, default: str = "") -> str:
    result = input(prompt).strip()
    return result if result else default


def _confirm(prompt: str) -> bool:
    return _input(f"{prompt} (y/n) ", "n").lower() in ("y", "yes")


# ─── LLM 初始化 ─────────────────────────────────────

def init_llm() -> None:
    global llm
    print("\n=== LLM 初始化 ===")
    print("1. DeepSeek")
    print("2. Coze")
    choice = _input("选择 > ", "1")

    if choice == "1":
        api_key = _input("DeepSeek API Key: ")
        if not api_key:
            print("跳过 LLM 初始化")
            return
        llm = create_llm("deepseek", {"api_key": api_key, "temperature": 0.9})
        print("DeepSeek 已连接")
    elif choice == "2":
        token = _input("Coze Token: ")
        bot_id = _input("Bot ID: ")
        if not token:
            print("跳过 LLM 初始化")
            return
        llm = create_llm("coze", {"token": token, "bot_id": bot_id})
        print("Coze 已连接")


# ─── Agent 管理 ─────────────────────────────────────

def agent_menu() -> None:
    while True:
        print("\n=== Agent 管理 ===")
        print("1. 创建 Agent")
        print("2. 列出所有 Agent")
        print("3. 查看 Agent 详情")
        print("4. 编辑 Agent 状态")
        print("5. 删除 Agent")
        print("6. 查看 Agent 记忆")
        print("0. 返回")

        choice = _input("选择 > ")
        if choice == "1":
            _create_agent()
        elif choice == "2":
            _list_agents()
        elif choice == "3":
            _view_agent()
        elif choice == "4":
            _edit_agent_state()
        elif choice == "5":
            _delete_agent()
        elif choice == "6":
            _view_memories()
        elif choice == "0":
            break


def _create_agent() -> None:
    print("\n--- 创建 Agent ---")
    agent_id = _input("ID (英文): ")
    if not agent_id:
        return
    name = _input("名称: ")
    personality = _input("性格描述: ")
    speaking_style = _input("语言风格: ")
    background = _input("背景 (可选): ", "")

    try:
        agent = Agent(
            agent_id=agent_id,
            config=AgentConfig(
                name=name,
                personality=personality,
                speaking_style=speaking_style,
                background=background,
            ),
            state=AgentState(location="未知"),
            memory_store=memory_store,
        )
        pool.register(agent)
        print(f"Agent '{name}' 创建成功")
    except DuplicateAgentError:
        print(f"ID '{agent_id}' 已存在")


def _list_agents() -> None:
    agents = pool.list_all()
    if not agents:
        print("暂无 Agent")
        return
    print(f"\n共 {len(agents)} 个 Agent:")
    for a in agents:
        print(f"  [{a.id}] {a.config.name} | HP:{a.state.hp} 情绪:{a.state.emotion} 位置:{a.state.location}")


def _view_agent() -> None:
    _list_agents()
    agent_id = _input("输入 Agent ID 查看详情: ")
    try:
        a = pool.get(agent_id)
        print(f"\n=== {a.config.name} ===")
        print(f"ID: {a.id}")
        print(f"性格: {a.config.personality}")
        print(f"语言风格: {a.config.speaking_style}")
        if a.config.background:
            print(f"背景: {a.config.background}")
        print(f"HP: {a.state.hp}  MP: {a.state.mp}")
        print(f"情绪: {a.state.emotion}")
        print(f"位置: {a.state.location}")
        if a.state.relationships:
            print("人际关系:")
            for k, v in a.state.relationships.items():
                print(f"  {k}: {v}")
        if a.goals:
            print(f"目标: {', '.join(a.goals)}")
        if a.state.extras:
            print(f"扩展字段: {a.state.extras}")
    except AgentNotFoundError:
        print(f"Agent '{agent_id}' 不存在")


def _edit_agent_state() -> None:
    _list_agents()
    agent_id = _input("输入 Agent ID 编辑: ")
    try:
        pool.get(agent_id)
        print("输入要修改的字段 (留空跳过):")
        hp = _input("HP: ")
        emotion = _input("情绪: ")
        location = _input("位置: ")
        goals = _input("目标 (逗号分隔): ")

        patch = {}
        if hp:
            patch["hp"] = int(hp)
        if emotion:
            patch["emotion"] = emotion
        if location:
            patch["location"] = location
        if patch:
            pool.update_state(agent_id, patch)
        if goals:
            pool.set_goals(agent_id, [g.strip() for g in goals.split(",")])
        print("已更新")
    except AgentNotFoundError:
        print(f"Agent '{agent_id}' 不存在")


def _delete_agent() -> None:
    _list_agents()
    agent_id = _input("输入要删除的 Agent ID: ")
    try:
        pool.remove(agent_id)
        print(f"Agent '{agent_id}' 已删除")
    except AgentNotFoundError:
        print(f"Agent '{agent_id}' 不存在")


def _view_memories() -> None:
    _list_agents()
    agent_id = _input("输入 Agent ID 查看记忆: ")
    try:
        pool.get(agent_id)
        memories = memory_store.get_recent(agent_id, 20)
        if not memories:
            print("暂无记忆")
            return
        print(f"\n最近 {len(memories)} 条记忆:")
        for m in reversed(memories):
            print(f"  [{m.timestamp}] {m.content}")
    except AgentNotFoundError:
        print(f"Agent '{agent_id}' 不存在")


# ─── Scene 管理 ─────────────────────────────────────

def scene_menu() -> None:
    while True:
        print("\n=== Scene 管理 ===")
        print("1. 创建 Scene")
        print("2. 列出所有 Scene")
        print("3. 管理参与者")
        print("4. 配置停止条件")
        print("5. 删除 Scene")
        print("0. 返回")

        choice = _input("选择 > ")
        if choice == "1":
            _create_scene()
        elif choice == "2":
            _list_scenes()
        elif choice == "3":
            _manage_participants()
        elif choice == "4":
            _configure_stop()
        elif choice == "5":
            _delete_scene()
        elif choice == "0":
            break


def _create_scene() -> None:
    print("\n--- 创建 Scene ---")
    sid = _input("ID (英文): ")
    name = _input("名称: ")
    location = _input("地点: ")
    time = _input("时间: ")
    weather = _input("天气: ")
    atmosphere = _input("氛围: ")
    background = _input("场景描述 (可选): ", "")
    max_rounds = _input("最大回合数 (默认10): ", "10")

    config = SceneConfig(
        location=location, time=time, weather=weather,
        atmosphere=atmosphere, background=background,
        max_rounds=int(max_rounds),
    )
    scene = Scene(scene_id=sid, name=name, config=config)
    scenes[sid] = scene
    print(f"Scene '{name}' 创建成功")

    # 创建后立即引导添加参与者
    if pool.list_all():
        print("\n可用的 Agent:")
        for a in pool.list_all():
            print(f"  [{a.id}] {a.config.name}")
        if _confirm("\n是否现在添加参与者?"):
            while True:
                aid = _input("Agent ID (留空结束): ")
                if not aid:
                    break
                if pool.contains(aid):
                    scene.add_participant(aid)
                    print(f"  已添加 {aid}")
                else:
                    print(f"  Agent '{aid}' 不存在")
        print(f"当前参与者: {scene.participants}")
    else:
        print("提示: 请先创建 Agent，再通过 Scene 菜单管理参与者")


def _list_scenes() -> None:
    if not scenes:
        print("暂无 Scene")
        return
    for s in scenes.values():
        print(f"  [{s.id}] {s.name} | {s.config.location} {s.config.time} | "
              f"参与者:{len(s.participants)} 回合上限:{s.max_rounds}")


def _manage_participants() -> None:
    _list_scenes()
    sid = _input("Scene ID: ")
    scene = scenes.get(sid)
    if not scene:
        print("Scene 不存在")
        return

    while True:
        print(f"\n--- {scene.name} 参与者: {scene.participants} ---")
        print("1. 添加参与者")
        print("2. 移除参与者")
        print("0. 返回")
        choice = _input("> ")
        if choice == "1":
            _list_agents()
            aid = _input("Agent ID: ")
            if pool.contains(aid):
                scene.add_participant(aid)
                print("已添加")
            else:
                print("Agent 不存在")
        elif choice == "2":
            aid = _input("Agent ID: ")
            scene.remove_participant(aid)
            print("已移除")
        elif choice == "0":
            break


def _configure_stop() -> None:
    _list_scenes()
    sid = _input("Scene ID: ")
    scene = scenes.get(sid)
    if not scene:
        return
    print(f"\n--- 停止条件配置: {scene.name} ---")
    print(f"当前回合上限: {scene.max_rounds}")
    mr = _input("新回合上限: ")
    if mr:
        scene.set_max_rounds(int(mr))
    print("失败条件 (留空跳过):")
    hp = _input("  HP阈值 (≤此值判定失败): ")
    emotion = _input("  情绪崩溃值: ")
    fc = FailureConditions(
        hp_threshold=int(hp) if hp else None,
        emotion_extreme=emotion if emotion else None,
    )
    scene.set_failure_conditions(fc)
    print("已更新")


def _delete_scene() -> None:
    _list_scenes()
    sid = _input("Scene ID: ")
    if sid in scenes:
        del scenes[sid]
        print("已删除")


# ─── 运行模拟 ───────────────────────────────────────

def run_simulation() -> None:
    if llm is None:
        print("请先初始化 LLM（主菜单选项 3）")
        return

    if not pool.list_all():
        print("请先创建 Agent（主菜单选项 1）")
        return

    _list_scenes()
    if not scenes:
        print("请先创建 Scene（主菜单选项 2）")
        return

    sid = _input("Scene ID: ")
    scene = scenes.get(sid)
    if not scene:
        print("Scene 不存在")
        return

    # 检查参与者，不足时引导添加
    if len(scene.participants) < 2:
        print(f"当前场景参与者不足 ({len(scene.participants)} 个)，需要至少 2 个。")
        print("可用 Agent:")
        for a in pool.list_all():
            in_scene = "✓" if a.id in scene.participants else " "
            print(f"  [{in_scene}] {a.id} - {a.config.name}")
        print("\n请输入要添加的 Agent ID（留空结束）:")
        while True:
            aid = _input("> ")
            if not aid:
                break
            if pool.contains(aid):
                scene.add_participant(aid)
                print(f"  已添加 {aid}")
            else:
                print(f"  Agent '{aid}' 不存在")
        if len(scene.participants) < 2:
            print("参与者仍不足，无法运行模拟。")
            return

    # 导出参与者数据
    participants = []
    for aid in scene.participants:
        try:
            a = pool.get(aid)
            participants.append(a.to_dict())
        except AgentNotFoundError:
            print(f"Agent '{aid}' 不存在，跳过")

    if len(participants) < 2:
        print("有效参与者不足")
        return

    graph = build_scheduler_graph()
    state = create_initial_state(scene.to_dict(), participants)

    print(f"\n{'='*60}")
    print(f"  场景: {scene.name}")
    print(f"  地点: {scene.config.location} | {scene.config.time} | {scene.config.weather}")
    print(f"  参与者: {', '.join(a['config']['name'] for a in participants)}")
    print(f"{'='*60}")

    for chunk in graph.stream(state, {
        "configurable": {"llm": llm, "thread_id": f"cli_{sid}"}
    }):
        for node, data in chunk.items():
            if data is None:
                continue

            if node == "choose_speaker":
                sp = data["current_speaker"]
                r = data["round"]
                p_name = _find_name(participants, sp)
                print(f"\n── 回合 {r} · {p_name} ──")

            elif node == "call_llm":
                output = data.get("_llm_output", "")
                sp = _find_name(participants, state.get("current_speaker", ""))
                print(f"  {output}")

            elif node == "parse_state":
                delta = data.get("_delta")
                if delta:
                    em = delta.get("emotion", "")
                    rel = delta.get("relationship_change", {})
                    ex = delta.get("exit_intent")
                    parts = [f"情绪→{em}"] if em else []
                    if rel:
                        parts.append(f"好感度变化:{rel}")
                    if ex:
                        parts.append("意图退出")
                    if parts:
                        print(f"  [{', '.join(parts)}]")

            elif node == "check_stop":
                sr = data.get("stop_reason")
                if sr:
                    reason_map = {
                        "max_rounds": "达到回合上限",
                        "failure": f"失败: {sr.get('agent_id')} {sr.get('cause')}",
                        "withdrawal": f"{sr.get('agent_id')} 避让退出",
                        "manual": "手动终止",
                    }
                    print(f"\n  结束: {reason_map.get(sr['type'], sr['type'])}")

    # 保存结果
    turns = state.get("turns", [])
    print(f"\n共 {len(turns)} 轮对话完成，数据已存于内存快照中。")

    # 把快照更新回 Agent
    if _confirm("\n将最终状态写回 Agent?"):
        snapshots = state.get("snapshots", {})
        for aid, snap in snapshots.items():
            try:
                pool.update_state(aid, {
                    k: v for k, v in snap.items()
                    if k in ("hp", "mp", "emotion", "location", "relationships")
                })
            except AgentNotFoundError:
                pass
        # 注入记忆
        for t in turns:
            speaker = t["speaker_id"]
            speaker_name = _find_name(participants, speaker)
            try:
                a = pool.get(speaker)
                a.remember(MemoryItem(
                    id=f"turn_{turns.index(t)}",
                    timestamp=f"场景:{scene.name}",
                    content=f"在{scene.config.location}，{speaker_name}说：{t['raw_output'][:100]}",
                    tags=["simulation"],
                ))
            except AgentNotFoundError:
                pass
        print("状态和记忆已写入")


def _find_name(participants: list[dict], agent_id: str) -> str:
    for p in participants:
        if p["id"] == agent_id:
            return p["config"]["name"]
    return agent_id


# ─── 持久化 ─────────────────────────────────────────

def save_all() -> None:
    _init_dirs()
    # 保存 Agent
    for a in pool.list_all():
        path = AGENTS_DIR / f"{a.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(a.to_dict(), f, ensure_ascii=False, indent=2)
    # 保存 Scene
    for s in scenes.values():
        path = SCENES_DIR / f"{s.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(s.to_dict(), f, ensure_ascii=False, indent=2)
    # 保存记忆
    for a in pool.list_all():
        mem_path = DATA_DIR / "memory" / f"{a.id}.json"
        mem_path.parent.mkdir(parents=True, exist_ok=True)
        items = memory_store.get_all(a.id)
        with open(mem_path, "w", encoding="utf-8") as f:
            json.dump([{"id": m.id, "timestamp": m.timestamp, "content": m.content, "tags": m.tags}
                       for m in items], f, ensure_ascii=False, indent=2)
    print(f"已保存到 {DATA_DIR.absolute()}")


def load_all() -> None:
    _init_dirs()
    # 加载 Agent
    if AGENTS_DIR.exists():
        for f in AGENTS_DIR.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            try:
                agent = Agent.from_dict(data, memory_store=memory_store)
                pool.register(agent)
            except DuplicateAgentError:
                pass
    # 加载 Scene
    if SCENES_DIR.exists():
        for f in SCENES_DIR.glob("*.json"):
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            scenes[data["id"]] = Scene.from_dict(data)
    # 加载记忆
    mem_dir = DATA_DIR / "memory"
    if mem_dir.exists():
        for f in mem_dir.glob("*.json"):
            agent_id = f.stem
            with open(f, "r", encoding="utf-8") as fp:
                items = json.load(fp)
            for item in items:
                memory_store.add(agent_id, MemoryItem(
                    id=item["id"], timestamp=item.get("timestamp", ""),
                    content=item.get("content", ""), tags=item.get("tags", []),
                ))
    print(f"已从 {DATA_DIR.absolute()} 加载 {len(pool.list_ids())} Agent, {len(scenes)} Scene")


# ─── 快速开始 ───────────────────────────────────────

def _quick_start() -> None:
    """引导式创建 Agent → Scene → 运行模拟。"""
    print("\n=== 快速开始 ===\n")

    # Step 1: LLM
    if llm is None:
        print("[1/4] 初始化 LLM...")
        init_llm()
        if llm is None:
            print("LLM 未配置，无法继续。")
            return

    # Step 2: 创建 Agent
    print("\n[2/4] 创建 Agent（至少 2 个）...")
    while len(pool.list_ids()) < 2:
        print(f"\n当前 {len(pool.list_ids())} 个 Agent。请创建：")
        _create_agent()

    # Step 3: 创建 Scene
    print("\n[3/4] 创建 Scene...")
    _create_scene()

    # Step 4: 添加参与者
    if scenes:
        # 找到刚创建的场景（或让用户选）
        if len(scenes) == 1:
            scene = list(scenes.values())[0]
        else:
            _list_scenes()
            sid = _input("选择 Scene ID: ")
            scene = scenes.get(sid)

        if scene:
            print(f"\n为 '{scene.name}' 添加参与者:")
            for a in pool.list_all():
                if a.id not in scene.participants:
                    if _confirm(f"  添加 {a.config.name} ({a.id})?"):
                        scene.add_participant(a.id)

    # Step 5: 运行
    print(f"\n[4/4] 准备就绪：{len(pool.list_ids())} 个 Agent, {len(scenes)} 个 Scene")
    if _confirm("是否立即运行模拟?"):
        run_simulation()


# ─── 主菜单 ─────────────────────────────────────────

def main() -> None:
    print("╔══════════════════════════════╗")
    print("║   Agent Pool CLI — MVP      ║")
    print("║   Novel Writing Assistant   ║")
    print("╚══════════════════════════════╝")

    # 自动初始化 DeepSeek
    global llm
    env_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if env_key:
        try:
            llm = create_llm("deepseek", {"api_key": env_key, "temperature": 0.9})
            print("LLM: DeepSeek (from env)")
        except Exception:
            pass

    # 尝试自动加载已有数据
    if AGENTS_DIR.exists() and any(AGENTS_DIR.glob("*.json")):
        load_all()

    while True:
        print("\n========== 主菜单 ==========")
        print(f"Agents: {len(pool.list_ids())} | Scenes: {len(scenes)} | LLM: {'OK' if llm else 'OFF'}")
        print("1. Agent 管理")
        print("2. Scene 管理")
        print("3. LLM 初始化")
        print("4. 运行模拟")
        print("5. 保存数据")
        print("6. 加载数据")
        print("7. 快速开始 (创建Agent→Scene→模拟)")
        print("0. 退出")

        choice = _input("> ")
        if choice == "1":
            agent_menu()
        elif choice == "2":
            scene_menu()
        elif choice == "3":
            init_llm()
        elif choice == "4":
            run_simulation()
        elif choice == "5":
            save_all()
        elif choice == "6":
            load_all()
        elif choice == "7":
            _quick_start()
        elif choice == "0":
            if _confirm("退出前保存?"):
                save_all()
            print("再见!")
            break


if __name__ == "__main__":
    main()
