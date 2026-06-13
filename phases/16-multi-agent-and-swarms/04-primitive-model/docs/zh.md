# 多智能体原语模型

> 2026 年发布的每一个多智能体框架——AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework——都是四维设计空间中的一个点。四个原语，仅此而已：智能体、移交、共享状态、编排器。本课从零构建它们，在所有四个原语上运行一个玩具系统，然后将每个主流框架映射到相同的坐标轴上，让你能用一段话读懂任何新发布。

**类型：** 学习
**语言：** Python（标准库）
**前置课程：** 第 14 阶段（智能体工程）、第 16 阶段 · 01（为什么需要多智能体）
**时间：** 约 60 分钟

## 问题

每六个月就有一个新的多智能体框架发布。2023 年的 AutoGen。2024 年的 CrewAI。2024 年的 LangGraph 和 OpenAI Swarm。2025 年 4 月的 Google ADK。2026 年 2 月的 Microsoft Agent Framework RC。每份新闻稿都宣称自己是"正确的抽象"。

如果你试图逐个学习它们，你会精疲力竭。API 看起来不同。文档对"智能体"是什么各执一词。一个框架称其共享内存为"黑板"，另一个叫"消息池"，第三个叫"StateGraph"。你开始怀疑这个领域只是在原地打转。

事实并非如此。在营销之下，四个原语是稳定的。学一次，用一段话读懂每个新框架。

## 概念

### 四个原语

1. **智能体（Agent）** — 系统提示词加工具列表。无状态；每次运行都从系统提示词和当前消息历史开始。
2. **移交（Handoff）** — 从一个智能体到另一个智能体的结构化控制转移。机制上，是一个返回新智能体的工具调用或一个遵循条件的图边。
3. **共享状态（Shared state）** — 任何多个智能体可以读取（有时写入）的数据结构。消息池、黑板、键值存储、向量记忆。
4. **编排器（Orchestrator）** — 决定谁下一个发言的角色。选项：显式图（确定性）、LLM 发言选择器（柔性）、上一个发言者的移交调用（OpenAI Swarm），或队列上的调度器（群体架构）。

这就是整个设计空间。每个框架为每个坐标轴选择默认值；其余的只是表面语法。

### 2026 年每个框架如何映射

| 框架 | 智能体 | 移交 | 共享状态 | 编排器 |
|------|--------|------|---------|--------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | 工具返回 Agent | 调用者的问题 | LLM 的下一次移交调用 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的发言选择器 | 消息池 | 选择器函数（LLM 或轮询） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 任务输出链式传递 | 管理者 LLM 或静态顺序 |
| LangGraph | 节点函数 | 图边 + 条件 | `StateGraph` 归约器 | 图，确定性 |
| Microsoft Agent Framework | 智能体 + 编排模式 | 模式特定 | 线程 / 上下文 | 模式特定 |
| Google ADK | 智能体 + A2A 卡片 | A2A 任务 | A2A 工件 | 宿主决定 |

表面差异看起来很大。底层：相同的四个旋钮。

### 为什么这很重要

一旦你看到原语，框架比较就变成了一个简短的检查清单：

- 编排器是信任 LLM 来路由（Swarm）还是在代码中固定路由（LangGraph）？
- 共享状态是全历史（GroupChat）还是投影的（StateGraph 归约器）？
- 智能体能修改彼此的提示词（CrewAI 管理者）还是只能移交（Swarm）？

这三个问题回答了 80% 的"哪个框架适合给定问题"。你不再为"最好的多智能体框架"而购物，而是开始为你真正关心的坐标轴进行设计。

### 无状态洞察

除了共享状态之外，每个原语都是无状态的。智能体是（提示词，工具）的函数。移交是一个函数调用。编排器是一个调度器。**系统中唯一有状态的东西是共享状态。** 所有有趣的 bug 都在那里：记忆投毒（第 15 课）、消息排序、版本控制、写入竞争。

隐藏共享状态的框架（Swarm）将问题推给调用者。集中化的框架（LangGraph checkpoint、AutoGen pool）使其可检查，但将协调成本转移到共享状态实现上。

### 单个原语的解剖

#### 智能体

```
Agent = (system_prompt, tools, model, optional_name)
```

没有记忆。没有状态。两个具有相同系统提示词和工具的智能体是可互换的。所有看起来像每个智能体状态的东西实际上都在共享状态或移交协议中。

#### 移交

```
Handoff = (from_agent, to_agent, reason, payload)
```

三种实现占主导：

- **函数返回** — 工具返回下一个智能体。这是 OpenAI Swarm 模式。智能体在其工具模式中携带路由。
- **图边** — LangGraph。边是声明式的。LLM 产生一个值；条件选择下一个节点。
- **发言选择** — AutoGen GroupChat。选择器函数（有时本身是 LLM 调用）读取池并选择下一个发言者。

#### 共享状态

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

至少是一个消息列表。通常更多：结构化工件（CrewAI 任务输出）、类型化上下文（LangGraph 归约器）、外部记忆（MCP、向量数据库）。

两种拓扑：**全池**（每个智能体看到每条消息）和**投影**（智能体看到角色范围的视图）。全池简单但扩展性差。投影池可扩展但需要预先的模式设计。

#### 编排器

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四种风格：

- **静态** — 图在构建时固定（LangGraph 确定性、CrewAI 顺序式）。
- **LLM 选择** — LLM 读取池并选择下一个发言者（AutoGen、CrewAI 层级式）。
- **移交驱动** — 当前智能体通过调用移交工具来决定（Swarm）。
- **队列驱动** — 工作者从共享队列中拉取；没有显式的下一个发言者（群体架构、Matrix）。

### 框架之间有什么变化

一旦原语确定，剩余的设计决策是：

- **记忆策略** — 临时 vs 持久检查点（LangGraph checkpointer）。
- **安全边界** — 谁可以批准移交（人在回路中）。
- **成本核算** — 每个智能体的 token 预算。
- **可观察性** — 追踪移交、持久化状态以供重放。

所有都可以在原语之上实现。没有一个是新原语。

## 动手构建

`code/main.py` 用约 150 行标准库 Python 实现了四个原语。没有真实的 LLM——每个智能体是一个脚本化策略，重点保持在协调结构上。

文件导出：

- `Agent` — 名称、系统提示词、工具、策略函数的数据类。
- `Handoff` — 返回新智能体的函数。
- `SharedState` — 线程安全的消息池。
- `Orchestrator` — 三种变体：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模拟）。

演示通过所有三种编排器类型运行相同的三智能体流水线（研究 → 写作 → 审查），并在最后打印消息池。你可以看到输出仅在*谁选择下一个*上不同；智能体和共享状态在各次运行中是相同的。

运行：

```
python3 code/main.py
```

预期输出：三次编排器运行，每种模式一次。每次打印最终消息池。如果研究者决定提前完成，移交驱动的运行会到达更少的智能体——这就是 LLM 路由权衡的缩影。

## 应用

`outputs/skill-primitive-mapper.md` 是一个技能，它读取任何多智能体代码库或框架文档并返回四原语映射。在新框架发布时运行它，在深入阅读文档之前获得一段话的理解。

## 交付

在采用新框架之前，为它编写原语映射。如果做不到，说明文档不完整或者框架在发明第五个原语（罕见——检查是否有你没见过的共享状态变体）。

将映射固定在你的架构文档中。当新团队成员加入时，在发送 API 文档之前先发送映射。当框架版本变化时，对比映射，而不是变更日志。

## 练习

1. 用不同的智能体策略运行 `code/main.py` 三次。观察编排器选择如何改变哪些智能体运行。
2. 实现第四种编排器类型：队列驱动的，智能体从共享状态轮询工作。可能发生什么死锁，你如何检测它？
3. 取 LangGraph 快速入门（https://docs.langchain.com/oss/python/langgraph/workflows-agents）并将其重写为四个原语。LangGraph 的哪些抽象是 1:1 映射的，哪些是便利包装器？
4. 阅读 OpenAI Swarm cookbook（https://developers.openai.com/cookbook/examples/orchestrating_agents）。识别 Swarm 使哪个原语最符合人体工学，以及它将哪个推给了调用者。
5. 在此表中找到一个完全隐藏共享状态的框架。解释当智能体需要在移交之间协调而无需重新读取历史时会出什么问题。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 智能体（Agent） | "带工具的 LLM" | 一个 `(system_prompt, tools, model)` 三元组。无状态。 |
| 移交（Handoff） | "控制转移" | 一个结构化调用，指定下一个智能体和可选负载。三种实现：函数返回、图边、发言选择。 |
| 共享状态（Shared state） | "记忆" / "上下文" | 多智能体系统中唯一有状态的部分。消息池或黑板。 |
| 编排器（Orchestrator） | "协调器" | 决定谁下一个运行的角色。静态图、LLM 选择器、移交驱动或队列驱动。 |
| 原语（Primitive） | "抽象" | 每个框架参数化的四个坐标轴之一。不是框架特性。 |
| 消息池（Message pool） | "共享聊天历史" | 全历史共享状态。易于理解，扩展性差。 |
| 投影状态（Projected state） | "范围视图" | 共享状态的角色特定视图。可扩展，需要模式设计。 |
| 发言选择（Speaker selection） | "谁下一个说话" | 编排器模式，函数（通常是 LLM）从群组中选择下一个智能体。 |

## 延伸阅读

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 移交驱动编排的最清晰阐述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + 发言选择是 LLM 选择编排的参考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 图边编排和基于归约器的共享状态
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — 角色-目标-背景智能体，顺序/层级式流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — 微软将 v0.4 转入维护后，活跃的 AutoGen v0.2 线路
