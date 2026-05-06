# Agent Pool 开发方案

> 版本：2.0  
> 最后更新：2026-05-05  
> 依赖设计文档：`agent-pool-core.md`

---

## 1. 技术选型

| 层 | 选择 | 理由 |
|----|------|------|
| 语言 | **Python 3.12+** | 复用现有 `CozeChatModel`（LangChain Python 实现）；DeepSeek 完美兼容 LangChain Python；LangGraph Python 比 JS 版更成熟 |
| 包管理 | `uv` 或 `pip` + `pyproject.toml` | 轻量，Python 标准 |
| LLM 调用 | **DeepSeek**（主力）+ **Coze**（备选） | 两个都便宜，见第 2 节详细说明 |
| LLM 框架 | LangChain `BaseChatModel` | CozeChatModel 已基于此实现；DeepSeek 直接用 `ChatOpenAI` 兼容 |
| 对话历史 | LangChain `BaseChatMessageHistory` | 省掉手写队列拼接 |
| 调度引擎 | **LangGraph `StateGraph`**（Python 版） | 声明式状态图，内建 checkpoint + 条件分支 + 流式输出 |
| 角色 Agent | **不用** LangChain Agent/AgentExecutor | 自己写 Python class，与工具调用 Agent 是两个东西 |
| 前端 | React 18 + Zustand + Vite | 轻量，刷新快 |
| 前后端通信 | FastAPI + WebSocket | Python 后端 + TS 前端的标准桥 |
| 存储（MVP） | JSON 文件 | 零依赖，数据透明 |
| 测试 | pytest | Python 标准 |

### 为什么从 TypeScript 换到 Python

原本考虑 TS 全栈，但两个主力 LLM 的情况是：

- **CozeChatModel**：已有的 Python + LangChain 实现（`coze_chat_tool/langchain/chat_model.py`），底层走 Coze Bot REST API。用 TS 就得用 HTTP 裸调 Coze API，多一层维护成本。
- **DeepSeek**：兼容 OpenAI API 格式。Python 用 `ChatOpenAI(base_url="https://api.deepseek.com")` 零配置接入。TS 用 OpenAI SDK 同样可以，但和其他 LangChain 组件衔接不如 Python 顺畅。

核心引擎用 Python，保留 TS 只在前端。分工清晰。

### 为什么调度器用 LangGraph 而不是手写循环

```
手写 for + await + if/else：
- 停止条件判断散落在代码各处
- 中断恢复需要自己序列化/反序列化状态
- 调试时看不到流程走到了哪一步
- 加一个节点（比如加环境描写）要改循环结构

LangGraph StateGraph：
- 每个步骤是一个节点，步骤间流转是边，一目了然
- 内建 checkpoint：中断自动保存，重启从断点继续
- 条件边声明式表达四种停止条件
- 加节点加边就行，不碰原有结构
```

---

## 2. LLM 方案

两个模型都便宜，按场景分工。

### 2.1 DeepSeek（主力）

| 项 | 说明 |
|----|------|
| 接入方式 | `ChatOpenAI` 兼容，设置 `base_url="https://api.deepseek.com"` |
| API Key | 用户自备，环境变量 `DEEPSEEK_API_KEY` |
| 价格 | 极低（约 ¥1/百万 token） |
| 适用场景 | 角色对白生成、状态解析、对话总结 |
| 优势 | 中文能力强，成本最低，API 稳定 |
| 劣势 | 高峰期可能限流 |

```python
from langchain_openai import ChatOpenAI

deepseek = ChatOpenAI(
    model="deepseek-chat",
    api_key="sk-xxx",                    # 或从环境变量读取
    base_url="https://api.deepseek.com",
    temperature=0.8,                     # 创作场景适当提高
)
```

### 2.2 Coze（备选 / 特殊场景）

| 项 | 说明 |
|----|------|
| 接入方式 | 已有 `CozeChatModel`，extends `BaseChatModel` |
| 认证 | Coze PAT（Personal Access Token）+ Bot ID |
| 价格 | 按 Coze 平台计费（也很便宜） |
| 适用场景 | 特定 Bot 调优后的角色对话、需要工作流串联的场景 |
| 优势 | 已封装好，直接 import 用；Bot 端可配置知识库和工作流 |
| 劣势 | 参数（temperature 等）在 Bot 端配置，代码侧不可调；依赖 Coze 平台可用性 |

```python
from coze_chat_tool.langchain import CozeChatModel

coze = CozeChatModel(
    token="pat_xxx",       # Coze Personal Access Token
    bot_id="xxx",          # Bot ID
)
```

### 2.3 使用策略

```
                    ┌─────────────────┐
                    │  调用类型判断     │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              ▼              ▼              ▼
         角色对白生成     状态解析        环境描写
        (大段创作，中文)  (简短JSON抽取)   (风格化描写)
              │              │              │
              ▼              ▼              ▼
          DeepSeek        DeepSeek        DeepSeek
         (temperature    (temperature   (temperature
           0.8~1.0)        0.3)           0.9)
              │              │              │
              └──────────────┼──────────────┘
                             │
                    可切 Coze 作为备选
                   （某 Bot 专门调优时）
```

核心思路：**DeepSeek 打主力，Coze 作为可替换的备选方案**。两个都通过统一的 `BaseChatModel` 接口调用，引擎代码不绑定任何一个。

### 2.4 LLM 工厂

```python
# packages/core/llm/factory.py

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

def create_llm(provider: str, config: dict) -> BaseChatModel:
    if provider == "deepseek":
        return ChatOpenAI(
            model=config.get("model", "deepseek-chat"),
            api_key=config["api_key"],
            base_url="https://api.deepseek.com",
            temperature=config.get("temperature", 0.8),
        )
    elif provider == "coze":
        from coze_chat_tool.langchain import CozeChatModel
        return CozeChatModel(
            token=config["token"],
            bot_id=config["bot_id"],
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

引擎只需要 `BaseChatModel` 接口，不管背后是 DeepSeek 还是 Coze。

---

## 3. 项目结构

```
agent-pool/
├── packages/
│   │
│   ├── core/                          # 核心引擎（纯 Python，零 UI 依赖）
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── types.py               # dataclass：Agent, Scene, StateSnapshot...
│   │   │   ├── agent.py               # Agent class
│   │   │   ├── pool.py                # AgentPool 管理器
│   │   │   ├── scene.py               # Scene 容器
│   │   │   ├── prompt_builder.py      # Prompt 组装
│   │   │   ├── memory_store.py        # 短时记忆（基于 BaseChatMessageHistory）
│   │   │   ├── state_resolver.py      # 状态变化二次解析
│   │   │   ├── graph/                 # LangGraph 调度图
│   │   │   │   ├── __init__.py
│   │   │   │   ├── state.py           # SimulationState
│   │   │   │   ├── nodes.py           # 节点函数
│   │   │   │   └── scheduler.py       # build_graph()
│   │   │   ├── llm/                   # LLM 适配
│   │   │   │   ├── __init__.py
│   │   │   │   └── factory.py         # create_llm(provider, config)
│   │   │   └── storage/
│   │   │       ├── __init__.py
│   │   │       ├── file_store.py      # JSON 文件读写
│   │   │       └── memory_store.py    # 内存存储（测试用）
│   │   ├── tests/
│   │   │   ├── test_agent.py
│   │   │   ├── test_pool.py
│   │   │   ├── test_scene.py
│   │   │   ├── test_scheduler.py      # 四种停止条件全覆盖
│   │   │   └── test_state_resolver.py
│   │   └── pyproject.toml
│   │
│   ├── cli/                           # 命令行原型（第一阶段验证）
│   │   ├── src/
│   │   │   ├── __init__.py
│   │   │   ├── main.py
│   │   │   └── commands/
│   │   │       ├── create_agent.py
│   │   │       ├── create_scene.py
│   │   │       └── run_scene.py
│   │   └── pyproject.toml
│   │
│   └── server/                        # FastAPI 后端（第二阶段，桥接前端）
│       ├── src/
│       │   ├── __init__.py
│       │   ├── main.py                # FastAPI app
│       │   ├── routes/
│       │   │   ├── agents.py
│       │   │   ├── scenes.py
│       │   │   └── simulation.py      # WebSocket 端点，流式推送回合
│       │   └── ws/
│       │       └── simulation_ws.py   # WebSocket handler
│       └── pyproject.toml
│
└── apps/
    └── web/                           # React 前端
        ├── src/
        │   ├── components/
        │   │   ├── AgentCard.tsx
        │   │   ├── AgentEditor.tsx
        │   │   ├── ScenePanel.tsx
        │   │   ├── DialogueView.tsx
        │   │   ├── StateDiff.tsx
        │   │   └── StopConditionConfig.tsx
        │   ├── hooks/
        │   │   ├── useWebSocket.ts
        │   │   ├── useAgentPool.ts
        │   │   └── useScene.ts
        │   ├── store.ts
        │   └── App.tsx
        └── package.json
```

---

## 4. 核心引擎设计

### 4.1 Agent 类

```python
# packages/core/src/agent.py
from langchain_core.chat_history import BaseChatMessageHistory

@dataclass
class Agent:
    id: str
    name: str
    personality: str
    speaking_style: str
    state: AgentState
    short_memory: BaseChatMessageHistory
    goals: list[str]

    def to_prompt_context(self) -> str:
        """将自身状态编译为 Prompt 文本块"""

    def remember(self, item: MemoryItem) -> None:
        """添加记忆（交互确认后调用）"""

    def apply_state_changes(self, changes: StateDelta) -> None:
        """应用状态变化（交互确认后调用）"""
```

### 4.2 Pool 类

```python
class AgentPool:
    def register(self, agent: Agent) -> None: ...
    def remove(self, id: str) -> None: ...
    def get(self, id: str) -> Agent | None: ...
    def list_all(self) -> list[Agent]: ...
    def export(self, ids: list[str]) -> list[AgentData]: ...
    def update_state(self, id: str, patch: dict) -> None: ...
```

### 4.3 Scene 类

```python
@dataclass
class Scene:
    id: str
    name: str
    location: str
    time: str
    weather: str
    atmosphere: str
    background: str
    participants: list[str]
    config: SceneConfig

    def build_shared_context(self) -> str: ...
    def get_other_participants(self, agent_id: str) -> list[AgentBrief]: ...
```

---

## 5. LangGraph 调度图（Python 版）

### 5.1 状态定义

```python
# packages/core/src/graph/state.py
from typing import TypedDict
from langgraph.graph import StateGraph, END

class SimulationState(TypedDict):
    # 输入（不可变）
    scene: dict
    participants: list[dict]

    # 运行时
    snapshots: dict[str, dict]      # agent_id -> 临时状态
    turns: list[dict]               # 已完成回合
    current_speaker: str
    round: int

    # 控制
    stop_reason: dict | None
    status: str                     # "running" | "paused" | "finished"
```

### 5.2 节点列表

```python
# packages/core/src/graph/nodes.py

def choose_speaker(state: SimulationState) -> dict:
    """确定本轮谁发言"""

def build_prompt(state: SimulationState) -> dict:
    """组装 Prompt：Agent 档案 + Scene 上下文 + 对话历史"""

def call_llm(state: SimulationState) -> dict:
    """调 LLM，获取对白/动作。配置中指定 provider"""

def parse_state(state: SimulationState) -> dict:
    """二次调用 LLM 解析状态变化 + exit_intent"""

def apply_delta(state: SimulationState) -> dict:
    """更新临时状态"""

def check_stop(state: SimulationState) -> dict:
    """检查四种停止条件"""
```

### 5.3 图结构

```
              ┌─────────────┐
              │choose_speaker│
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ build_prompt│
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │   call_llm  │──── await LLM ──── 流式推送到 WebSocket
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ parse_state │──── 二次 LLM 调用
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ apply_delta │──── 更新临时状态
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │  check_stop │
              └──────┬──────┘
                     │
          ┌──────────┼──────────┐
          │          │          │          │
      手动终止    失败判定   避让退出    回合上限
          │          │          │          │
          └──────────┴──────────┴──────────┘
                     │
              ┌──────▼──────┐
              │    END      │
              └─────────────┘
```

### 5.4 图构建代码

```python
# packages/core/src/graph/scheduler.py
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from .state import SimulationState
from . import nodes

def build_graph(llm_factory=None) -> StateGraph:
    graph = StateGraph(SimulationState)

    graph.add_node("choose_speaker", nodes.choose_speaker)
    graph.add_node("build_prompt", nodes.build_prompt)
    graph.add_node("call_llm", nodes.call_llm)
    graph.add_node("parse_state", nodes.parse_state)
    graph.add_node("apply_delta", nodes.apply_delta)
    graph.add_node("check_stop", nodes.check_stop)

    graph.add_edge("choose_speaker", "build_prompt")
    graph.add_edge("build_prompt", "call_llm")
    graph.add_edge("call_llm", "parse_state")
    graph.add_edge("parse_state", "apply_delta")
    graph.add_edge("apply_delta", "check_stop")

    # 条件边：命中停止条件 → END，否则继续循环
    graph.add_conditional_edges(
        "check_stop",
        lambda state: END if state["stop_reason"] else "choose_speaker",
    )

    graph.set_entry_point("choose_speaker")

    return graph.compile(checkpointer=MemorySaver())
```

### 5.5 call_llm 节点内部实现

```python
# packages/core/src/graph/nodes.py

def call_llm(state: SimulationState, config: RunnableConfig) -> dict:
    """调用 LLM 生成当前发言人的对白"""
    prompt = state["_prompt"]                # build_prompt 节点产出
    llm = _get_llm(config)                  # 从 config 取 LLM 实例

    response = llm.invoke(prompt)
    content = response.content if hasattr(response, 'content') else str(response)

    return {
        "turns": state["turns"] + [{
            "speaker": state["current_speaker"],
            "raw_output": content,
            "delta": None,                   # parse_state 节点会填充
        }]
    }
```

### 5.6 使用方式

```python
from graph.scheduler import build_graph

graph = build_graph()

initial_state: SimulationState = {
    "scene": scene.to_dict(),
    "participants": pool.export(scene.participants),
    "snapshots": {},
    "turns": [],
    "current_speaker": "",
    "round": 0,
    "stop_reason": None,
    "status": "running",
    "_prompt": "",
}

# 流式执行 → 每个节点完成时推送
for chunk in graph.stream(initial_state, {"configurable": {"thread_id": scene.id}}):
    # chunk 包含当前步骤的输出
    # 通过 WebSocket 推到前端
    yield chunk
```

### 5.7 中断 + 恢复

```python
# 作者暂停 → LangGraph 自动保存 checkpoint
# 作者继续 → 从断点恢复
graph.stream(None, {"configurable": {"thread_id": scene.id}})
```

---

## 6. 四种停止条件实现

```python
# packages/core/src/graph/nodes.py

def check_stop(state: SimulationState) -> dict:
    scene_config = state["scene"]["config"]
    snapshots = state["snapshots"]
    turns = state["turns"]
    round_num = state["round"]

    # 条件 1：手动终止（外部信号）
    if state.get("_manual_stop"):
        return {"stop_reason": {"type": "manual"}, "status": "finished"}

    # 条件 2：失败
    for agent_id, snap in snapshots.items():
        failure = scene_config.get("failure_conditions", {})

        if failure.get("hp_threshold") and snap["hp"] <= failure["hp_threshold"]:
            return {"stop_reason": {"type": "failure", "agent_id": agent_id, "cause": "hp_zero"}, "status": "finished"}

        if failure.get("emotion_extreme") and snap["emotion"] == failure["emotion_extreme"]:
            return {"stop_reason": {"type": "failure", "agent_id": agent_id, "cause": "emotional_breakdown"}, "status": "finished"}

    # 条件 3：避让退出
    last_turn = turns[-1] if turns else None
    if last_turn and last_turn.get("delta", {}).get("exit_intent"):
        return {"stop_reason": {"type": "withdrawal", "agent_id": last_turn["speaker"]}, "status": "finished"}

    # 条件 4：回合上限
    if round_num >= scene_config["max_rounds"]:
        return {"stop_reason": {"type": "max_rounds"}, "status": "finished"}

    return {}
```

---

## 7. 前后端通信

### 7.1 FastAPI 后端

```python
# packages/server/src/main.py

from fastapi import FastAPI, WebSocket
from core.graph.scheduler import build_graph

app = FastAPI()

@app.websocket("/ws/simulation/{scene_id}")
async def simulation_ws(websocket: WebSocket, scene_id: str):
    await websocket.accept()
    graph = build_graph()

    async for chunk in graph.astream(initial_state, {"configurable": {"thread_id": scene_id}}):
        await websocket.send_json(chunk)

    await websocket.close()
```

### 7.2 React 前端消费

```typescript
// apps/web/src/hooks/useWebSocket.ts
const ws = new WebSocket(`ws://localhost:8000/ws/simulation/${sceneId}`)

ws.onmessage = (event) => {
  const chunk = JSON.parse(event.data)
  // 更新对话视图 + 状态面板
  store.appendTurn(chunk)
}
```

---

## 8. LangChain 组件使用清单

| 组件 | 来源 | 用途 |
|------|------|------|
| `CozeChatModel` | 项目已有 `coze_chat_tool/langchain` | Coze Bot API 封装 |
| `ChatOpenAI` | `langchain-openai` | DeepSeek（兼容 OpenAI API） |
| `BaseChatMessageHistory` | `langchain-core` | 短时记忆队列基类 |
| `InMemoryChatMessageHistory` | `langchain-core` | 短时记忆内存实现 |
| `StateGraph` | `langgraph` | 调度图引擎 |
| `MemorySaver` | `langgraph` | 内建 checkpoint |

**明确不用的**：
- `create_react_agent` / `AgentExecutor` — 工具调用 Agent，和角色扮演不在一个范畴
- `ConversationChain` — 太黑盒
- `VectorStoreRetrieverMemory` — MVP 不做向量库

---

## 9. 依赖

```
# packages/core/pyproject.toml
[project]
dependencies = [
  "langchain-core",
  "langchain-openai",    # DeepSeek 通过 ChatOpenAI 接入
  "langgraph",
]

# DeepSeek API key: 环境变量 DEEPSEEK_API_KEY
# Coze: import 本地的 coze_chat_tool
```

---

## 10. 实施顺序（6 步）

| 步 | 内容 | 产出 | 验证 |
|----|------|------|------|
| **S1** 数据结构 | `types.py` + `agent.py` + `pool.py` + `scene.py` + `memory_store.py` | Python dataclass，Agent/Scene 可创建、状态可读写、短时记忆可增删 | pytest 单测 |
| **S2** LLM 通路 | `llm/factory.py` + `prompt_builder.py`，先接 DeepSeek | 单个 Agent 注入场景信息后生成一句对白 | CLI 给角色设定 → 看到符合人设的中文对话 |
| **S3** 状态解析 | `state_resolver.py`，二次 LLM 调用从对白中抽取数值变化 | 对白输入 → `{emotion, relationship_change, exit_intent}` 输出 | pytest 测多种对白场景 |
| **S4** 调度图 | `graph/` 全套：state + nodes + scheduler | 两 Agent 轮流对话、状态解析、四条件停止 | CLI 模拟 3-5 轮 → 四条件全部能触发 |
| **S5** FastAPI + WebSocket | `server/` 后端 + WebSocket 流式推送 | 浏览器能通过 WS 收到实时回合数据 | `websocat` 连上看到 JSON 流 |
| **S6** React 前端 | Agent 卡片 + Scene 面板 + 对话视图 + 停止条件配置 | 全流程可视化 | 浏览器操作全流程 |

S1-S2 可以并行推进（数据结构 + LLM 接入互相不依赖）。

---

## 11. 存储目录

```
project-data/
├── agents/
│   ├── 林逸风.json
│   └── 师妹.json
├── scenes/
│   └── 客栈争吵.json
└── snapshots/
    └── 2026-05-05_150000/
        ├── 林逸风.json
        └── 师妹.json
```

---

## 12. 与完整版系统的关系

- **做的**：Agent Pool、Scene、回合调度、四种停止条件、短时记忆、DeepSeek + Coze 双 LLM、手动触发
- **不做的**：时间轴、章节管理、自定冲突检测、向量记忆、环境 Agent 风格预设、Electron、导出

LangGraph 的图结构给后续扩展留了空间——加环境描写节点只需在 `call_llm` 前后插入节点，条件边不变。
