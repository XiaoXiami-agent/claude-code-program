# S1 数据结构需求文档

> 对应 dev-plan.md 实施步骤 S1

---

## M1: types.py

### 功能边界

定义系统中所有数据的类型、枚举和 Protocol 接口。本模块是纯定义层，零逻辑，零依赖（仅 Python 标准库 + `typing` + `dataclasses`）。

### 输入输出

- 输入：无
- 输出：所有 dataclass、Protocol、Enum 定义

### 接口清单

| 类型 | 说明 |
|------|------|
| `AgentState` | 角色可变状态（HP、情绪、位置、好感度、物品、buff），含 `extras: dict` 扩展槽 |
| `AgentConfig` | 角色不可变属性（姓名、性格描述、语言风格） |
| `AgentData` | Agent 持久化格式 = config + state + memory |
| `MemoryItem` | 单条记忆（id, timestamp, content, tags） |
| `SceneConfig` | 场景配置（地点、时间、天气、氛围、背景、maxRounds、failureConditions） |
| `InteractionType` | 枚举：dialogue / combat / monologue / free |
| `StopReason` | 停止原因（type + agentId + cause） |
| `TurnResult` | 单回合产出（speaker, rawOutput, delta） |
| `StateDelta` | 状态变化（emotion, relationshipChange, hpChange, exitIntent, reason） |
| `MemoryStore` (Protocol) | 记忆存取接口 |
| `AgentStore` (Protocol) | Agent 持久化接口 |
| `SceneStore` (Protocol) | Scene 持久化接口 |

### 异常场景

- AgentState 扩展字段 key 冲突 → 后写入覆盖前写入
- 枚举值传入非法字符串 → 构造时校验

### 验收标准

- [ ] 所有类型可通过 `dataclasses.asdict()` 序列化
- [ ] `AgentState.extras` 支持任意自定义字段
- [ ] Protocol 不引任何具体实现

---

## M2: memory_store.py

### 功能边界

短时记忆的存取抽象。提供 Protocol 定义（在 M1）+ 两个实现：
- `InMemoryMemoryStore`：基于 `list`，测试用
- `JSONFileMemoryStore`：基于 JSON 文件，生产用

### 输入输出

| 方法 | 输入 | 输出 |
|------|------|------|
| `add(agent_id, item)` | agent_id: str, item: MemoryItem | None |
| `get_recent(agent_id, n)` | agent_id: str, n: int | `list[MemoryItem]`（最近 n 条） |
| `get_all(agent_id)` | agent_id: str | `list[MemoryItem]` |

### 异常场景

- JSON 文件不存在 → 返回空列表，不抛异常
- JSON 文件损坏 → 抛 `DataCorruptionError`，附带文件路径
- 并发写入 → MVP 不处理（不做文件锁），假设单用户单进程

### 验收标准

- [ ] InMemory 实现：add + getRecent + getAll 正确
- [ ] JSONFile 实现：数据可跨进程持久化
- [ ] MemoryStore 协议和两个实现解耦

---

## M3: agent.py

### 功能边界

Agent 类：角色的数字分身。持有 Config、State、Memory。不主动发起 I/O（记忆操作通过注入的 MemoryStore）。

### 输入输出

| 方法 | 输入 | 输出 |
|------|------|------|
| `__init__(config, state, memory_store)` | AgentConfig, AgentState, MemoryStore | Agent 实例 |
| `to_prompt_context()` | 无 | str（注入 Prompt 的文本块） |
| `remember(item)` | MemoryItem | None |
| `apply_state_changes(delta)` | StateDelta | None（就地修改 state） |
| `recent_memories(n)` | n: int | `list[MemoryItem]` |
| `to_dict()` / `from_dict()` | — | 序列化/反序列化 |

### 异常场景

- `apply_state_changes` 传入无效字段 → 忽略未知字段，只更新已知字段
- `recent_memories(0)` → 返回空列表
- 从 dict 反序列化缺少必填字段 → 抛 `ValueError`

### 验收标准

- [ ] Agent 创建后可调用所有只读方法
- [ ] `to_prompt_context()` 输出包含性格 + 语言风格 + 当前状态 + 目标
- [ ] `apply_state_changes()` 正确更新 hp、emotion、relationships
- [ ] `to_dict()` / `from_dict()` 往返一致
- [ ] `extras` 自定义字段正确保留

---

## M4: pool.py

### 功能边界

AgentPool：管理所有 Agent 的生命周期。提供注册、移除、查询、状态修改、批量导出。

### 输入输出

| 方法 | 输入 | 输出 |
|------|------|------|
| `register(agent)` | Agent | None |
| `remove(id)` | str | None |
| `get(id)` | str | Agent 或 None |
| `list_all()` | 无 | `list[Agent]` |
| `update_state(id, patch)` | str, dict | None（就地修改） |
| `export(ids)` | `list[str]` | `list[dict]` |
| `save_all(store)` | AgentStore | None |
| `load_all(store)` | AgentStore | None |

### 异常场景

- 注册重复 ID → 抛 `DuplicateAgentError`
- 修改不存在的 Agent 状态 → 抛 `AgentNotFoundError`
- `export` 传入不存在的 ID → 跳过，不抛异常

### 验收标准

- [ ] CRUD 全流程正确
- [ ] `update_state` 支持部分更新（patch 只含要改的字段）
- [ ] `export` / `load_all` 往返一致
- [ ] 自定义状态字段（extras）正确保留

---

## M5: scene.py

### 功能边界

Scene：共享时空容器。持有场景基本信息 + 参与者列表 + 停止条件配置。提供 `build_shared_context()` 供 Prompt 注入。

### 输入输出

| 方法 | 输入 | 输出 |
|------|------|------|
| `__init__(config)` | SceneConfig | Scene 实例 |
| `add_participant(id)` | str | None |
| `remove_participant(id)` | str | None |
| `build_shared_context()` | 无 | str（注入 Prompt 的场景文本块） |
| `to_dict()` / `from_dict()` | — | 序列化/反序列化 |

### 异常场景

- 参与者已经在列表中 → 忽略，不抛异常
- `build_shared_context` 参与者列表为空 → 正常输出场景信息，参与者部分留空
- `failureConditions` 为空 → 不启用失败判定

### 验收标准

- [ ] 创建 Scene 后所有字段可读
- [ ] `build_shared_context()` 包含地点、时间、天气、氛围、背景
- [ ] 参与者增删正确
- [ ] `to_dict()` / `from_dict()` 往返一致
