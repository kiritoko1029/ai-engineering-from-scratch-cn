# LangGraph：有状态图与持久执行

> LangGraph 是 2026 年底层有状态编排的参考。智能体是状态机；节点是函数；边是转换；状态是不可变的，每步后检查点。从任何故障处精确恢复。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 12（工作流模式）
**所需时间：** 约75分钟

## 学习目标

- 描述 LangGraph 的核心模型：带不可变状态、函数节点、条件边和步后检查点的状态机。
- 说出文档强调的四个能力：持久执行、流式传输、人在环、综合记忆。
- 解释 LangGraph 支持的三种编排拓扑：主管、对等（群体）、层级（嵌套子图）。
- 用标准库实现一个带不可变状态、条件边和检查点/恢复周期的状态图。

## 问题所在

智能体和工作流共享一个问题：当一个 40 步的运行在第 38 步失败时，你想从第 38 步恢复，而非重新开始。二等状态模型让运维围绕假设全新运行的库拼凑重试。

LangGraph 的设计答案：状态是一等类型化对象，变更是显式的，每个节点后检查点持久化。恢复是一次 `load_state(session_id)` 调用。

## 概念说明

### 图

一个图由以下定义：

- **状态类型。** 每个节点读取和变更的类型化字典（或 Pydantic 模型）。
- **节点。** 纯函数 `(state) -> state_update`。返回后更新合并到状态。
- **边。** 节点间的条件或直接转换。
- **入口和出口。** `START` 和 `END` 哨兵节点标记边界。

示例：一个带 `classify`、`refund`、`bug`、`sales`、`done` 节点的智能体——一个作为图的路由工作流。

### 持久执行

每个节点返回后，运行时序列化状态并写入检查点器（SQLite、Postgres、Redis、自定义）。在第 N 步失败时，运行时可以 `resume(session_id)` 从第 N+1 步精确状态继续。

LangGraph 文档明确强调了这很重要的生产用户：Klarna、Uber、J.P. Morgan。主张不是图形状；而是图形状加检查点使恢复变得廉价。

### 流式传输

每个节点可以产出部分输出。图向调用者流式传输每节点增量事件，使 UI 在图运行时更新。

### 人在环

在节点间检查和修改状态。实现：在关键节点前暂停，向人工展示状态，接受修改，恢复。检查点器使这很容易，因为状态已经序列化。

### 记忆

短期（运行内——状态中的对话历史）和长期（跨运行——通过检查点器加独立长期存储持久化）。LangGraph 通过工具与外部记忆系统（Mem0、自定义）集成。

### 三种拓扑

1. **主管（Supervisor）。** 中央路由器 LLM 分派给专家子智能体。`langgraph-supervisor` 中的 `create_supervisor()`（尽管 LangChain 团队在 2026 年建议通过工具调用直接实现以获得更多上下文控制）。
2. **群体/对等（Swarm / peer-to-peer）。** 智能体通过共享工具接口直接移交。无中央路由器。
3. **层级（Hierarchical）。** 主管管理子主管，实现为嵌套子图。

### 此模式出错的地方

- **检查点太小。** 仅检查点对话轮次使工具状态和记忆写入不可恢复。完整状态必须可序列化。
- **非确定性节点。** 恢复假设节点输入产生相同的状态更新。随机种子、挂钟时间、外部 API 必须被捕获。
- **条件边过度使用。** 每条边都是条件的图是一个无法推理的状态机。更喜欢带偶尔分支的线性链。

## 开始构建

`code/main.py` 实现了一个标准库有状态图：

- `State`——带 `messages`、`step`、`route`、`output`、`human_approval` 的类型化字典。
- `Node`——接受状态并返回更新字典的可调用对象。
- `StateGraph`——节点 + 边 + 条件边 + 运行 + 恢复。
- `SQLiteCheckpointer`（内存中的伪实现）——每个节点后序列化状态；`load(session_id)` 恢复。
- 一个演示图：classify -> branch(refund / bug / sales) -> 人工门控 -> 发送。

运行它：

```
python3 code/main.py
```

轨迹展示了第一次运行在人工门控处失败，持久化，然后恢复产出最终输出。

## 使用它

- **LangGraph**——参考实现，生产就绪。使用 `create_react_agent`、`create_supervisor` 或构建自己的图。
- **AutoGen v0.4**（第 14 课）——高并发场景的 Actor 模型替代方案。
- **Claude Agent SDK**（第 17 课）——带内置会话存储的托管工具链。
- **自定义**——当你需要精确控制状态形状或检查点器后端时。

## 交付它

`outputs/skill-state-graph.md` 在任何目标运行时中生成 LangGraph 形态的状态图，带检查点和恢复。

## 练习

1. 当分类置信度低于阈值时，从 `classify` 添加到 `end` 的条件边。在人工手动设置 `route` 后恢复运行。
2. 将类 SQLite 伪实现替换为真实的 SQLite 检查点器。衡量每步序列化开销。
3. 实现并行边：两个节点并发运行，通过自定义合并器合并。不可变状态在这里带来什么好处？
4. 阅读 `langgraph-supervisor` 参考。将玩具实现移植到 `create_supervisor`。比较轨迹形态。
5. 添加流式传输：每个节点在运行时产出部分状态。打印到达的增量。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 状态图（State graph） | "智能体作为状态机" | 类型化状态 + 节点 + 边 + 合并器 |
| 检查点器（Checkpointer） | "持久化后端" | 每个节点后序列化状态；实现恢复 |
| 合并器（Reducer） | "状态合并" | 将当前状态与节点更新合并的函数 |
| 条件边（Conditional edge） | "分支" | 由状态函数选择的边 |
| 子图（Subgraph） | "嵌套图" | 在另一个图中作为节点使用的图 |
| 持久执行（Durable execution） | "从故障恢复" | 在最后一个成功的节点以精确状态重启 |
| 主管（Supervisor） | "路由器 LLM" | 专家子智能体的中央分派器 |
| 群体（Swarm） | "P2P 智能体" | 智能体通过共享工具移交；无中央路由器 |

## 延伸阅读

- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)——参考文档
- [langgraph-supervisor 参考](https://reference.langchain.com/python/langgraph/supervisor/)——主管模式 API
- [AutoGen v0.4，Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)——Actor 模型替代方案
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview)——会话存储和子智能体
