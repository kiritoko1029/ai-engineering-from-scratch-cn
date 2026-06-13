# Agent 框架取舍 —— LangGraph vs CrewAI vs AutoGen vs Agno

> 每个框架都在兜售同一个 demo（研究型 agent 生成一份报告），也都掩盖着同一个 bug（state schema 与编排层之间相互冲突）。挑选那个抽象与你问题形态相匹配的框架；其余的一切都是你要重写两遍的胶水代码。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 11 · 09（Function Calling）、Phase 11 · 16（LangGraph）
**所需时间：** 约45分钟

## 问题所在

你有一个任务，它需要不止一次 LLM 调用。也许它是一个研究工作流（规划、检索、总结、引用）。也许它是一条代码评审流水线（解析 diff、批评、打补丁、验证）。也许它是一个多轮助手，会预订机票、撰写邮件、提交报销单。你挑了一个框架。

三天后，你发现框架的抽象会泄漏。CrewAI 给了你角色，但当「研究员」需要把一份结构化计划交给「写作者」时，它却处处掣肘。AutoGen 给了你 agent 之间的对话，但没有一等公民的 state，于是你的检查点就成了对话日志的一个 pickle。LangGraph 给了你一张 state graph，却逼着你在还不知道 agent 会做什么之前就先给每一次转移命名。Agno 给了你一个单 agent 抽象，但当你试图把它扇出成三个并发 worker 时，它就崩溃尖叫。

解决之道不是「挑选最好的框架」，而是让框架的核心抽象去匹配你问题的形态。本课就来绘制这张地图。

## 概念说明

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

四个框架主导着 2026 年的格局。它们的核心抽象各不相同。

| Framework | 核心抽象 | 最佳适配 | 最差适配 |
|-----------|------------------|----------|----------|
| **LangGraph** | `StateGraph` —— 带类型的 state、节点、条件边、checkpointer。 | 具有显式 state 与 human-in-the-loop 中断的工作流；需要时间回溯调试的生产级 agent。 | 拓扑未知、松散且由角色驱动的头脑风暴。 |
| **CrewAI** | `Crew` —— 角色（目标、背景故事）、任务、流程（顺序式或层级式）。 | 角色扮演或人设驱动、带有简短线性/层级计划的工作流。 | 超出 crew 轮次历史之外的任何有状态需求；复杂的分支。 |
| **AutoGen** | `ConversableAgent` 对 —— 两个或更多 agent 轮流发言，直到满足退出条件。 | 多 agent *对话*（师生、提议者-批评者、执行者-评审者），其中的思考从对话中涌现出来。 | 具有已知 DAG 的确定性工作流；任何需要在重启后保持持久 state 的场景。 |
| **Agno** | `Agent` —— 单个 LLM + 工具 + 记忆，可组合成团队。 | 快速搭建的单 agent 与轻量级团队；强大的多模态能力与内置存储驱动。 | 带有自定义 reducer 的深层、显式分支图。 |

### 「抽象」究竟意味着什么

框架的核心抽象，就是你在白板上推介架构时画出来的那个东西。

- **LangGraph** → 你画的是一张图。节点是步骤，边是转移，且每一处的 state 对象都带类型。心智模型是一台状态机。
- **CrewAI** → 你画的是一张组织架构图。每个角色都有一份岗位描述，由一个管理者来路由任务。心智模型是一支由专家组成的小团队。
- **AutoGen** → 你画的是一段 Slack 私聊。两个 agent 互发消息；如果你需要一个主持人，就让第三个加入。心智模型是聊天。
- **Agno** → 你画的是一个单独的方框，上面挂着各种工具。把多个方框并排放在一起就是一个团队。心智模型是「自带电池的 agent」。

### state 问题

state 正是大多数框架选择在生产环境中崩溃的地方。

- **LangGraph。** 带类型的 state（`TypedDict` 或 Pydantic model）、逐字段 reducer、一等公民的 checkpointer（SQLite/Postgres/Redis）。恢复、中断与时间回溯都是免费附赠的。*（参见 Phase 11 · 16。）*
- **CrewAI。** state 以字符串形式通过 `context` 字段在任务之间流动，或通过 `output_pydantic` 以结构化方式传递。开箱即用时没有持久的逐 crew 存储；如果 crew 必须在重启后存活，你得自己拼装一套。
- **AutoGen。** state 就是对话历史以及任何用户自定义的 `context`。对话记录会持久化；除非你编写适配器，否则任意的工作流 state 不会持久化。
- **Agno。** 通过 `storage=` 挂接到 `Agent` 上的内置存储驱动（SQLite、Postgres、Mongo、Redis、DynamoDB）—— 对话会话与用户记忆会自动持久化。它不是一个完整的图 checkpointer；而是一个会话存储。

### 分支问题

每个非平凡的 agent 都会分支。由谁来决定分支，这一点很关键。

- **LangGraph** —— 由你决定，通过条件边来实现。路由是一个带命名分支的 Python 函数。分支在编译后的图中是一等公民；checkpointer 会记录走的是哪条分支。
- **CrewAI** —— 在层级模式下由管理者决定；在顺序模式下由你在构建时决定。路由隐含在任务列表里；除了管理者的 prompt 之外，没有一等公民的「if」。
- **AutoGen** —— 由 agent 通过对话决定。分支从「下一个由谁发言」中涌现出来。`GroupChatManager` 负责选出下一个发言者；你可以手写一个 `speaker_selection_method`，但默认是 LLM 驱动的。
- **Agno** —— 由 agent 通过决定下一步调用哪个工具来分支。团队拥有协调者/路由者/协作者模式；超出此范围的分支则由开发者负责。

### 可观测性问题

- **LangGraph** —— 通过 LangSmith 或任意 OTel exporter 实现 OpenTelemetry。每一次节点转移都是一个 trace span；检查点同时充当可重放的 trace。LangSmith 是第一方选项；Langfuse/Phoenix 也有适配器。
- **CrewAI** —— 自 2025 年底起拥有一等公民的 OpenTelemetry；与 Langfuse、Phoenix、Opik、AgentOps 集成。
- **AutoGen** —— 通过 `autogen-core` 实现 OpenTelemetry 集成；AgentOps 与 Opik 有连接器。追踪粒度是逐 agent 消息，而非逐节点。
- **Agno** —— 内置的 `monitoring=True` 标志外加 OpenTelemetry exporter；与 Langfuse 在会话追踪上紧密集成。

### 成本与延迟

四个框架都会增加每次调用的额外开销（框架逻辑、验证、序列化）。按开销递增大致排序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。其差异主要由框架做了多少额外的 LLM 路由所主导。CrewAI 的层级管理者会花费 token 来决定下一个由谁上场；AutoGen 的 `GroupChatManager` 同样如此。LangGraph 只在你写下 `llm.invoke` 的地方才花费 token。Agno 的单 agent 路径很轻薄。

当每次运行的成本很关键时，优先选择显式路由（LangGraph 的边、AutoGen 的 `speaker_selection_method`），而非 LLM 选择的路由。

### 互操作性

- **LangGraph** ↔ **LangChain** 的工具、retriever、LLM。一等公民的 MCP 适配器（工具以 MCP server 形式导入）。
- **CrewAI** ↔ 工具继承自 `BaseTool`；LangChain 工具、LlamaIndex 工具与 MCP 工具都能适配进来。通过 `allow_delegation=True` 实现 crew 到 crew 的委派。
- **AutoGen** → `FunctionTool` 可包装任意 Python callable；提供 MCP 适配器。在 agent 到 agent 的模式上与 AG2 生态紧密耦合。
- **Agno** → `@tool` 装饰器或 BaseTool 子类；提供 MCP 适配器；工具可在多个 agent 与团队之间共享。

## 核心技能

> 你能用一句话解释，为什么某个特定框架适合某个特定的 agent 问题。

构建前检查清单：

1. **画出形态。** 这是一张图吗（带类型的 state、命名的转移）？是一场角色扮演吗（专家们交接工作）？是一段对话吗（agent 们一直聊到完成）？还是一个带工具的单 agent？
2. **决定由谁分支。** 由开发者决定的分支 → LangGraph。由管理者 agent 决定 → CrewAI 层级式。从对话中涌现 → AutoGen。由工具调用决定 → Agno。
3. **核查 state 预算。** 你需要从检查点恢复吗？需要时间回溯吗？需要在运行中途由人介入吗？如果需要，LangGraph 是默认之选；Agno 的会话则覆盖对话范围内的 state。
4. **核查成本预算。** LLM 选择的路由在每一轮都会花费额外的 token。如果 agent 每天运行成千上万次，请优先选择显式路由。
5. **为框架开销做预算。** 每个框架都是又一个依赖。如果任务只是两次 LLM 调用加一个工具，那就写 30 行纯 Python；没有任何框架能比「不用框架」更省。

在你能画出那张图、那张组织架构图、那段对话或那个 agent 方框之前，拒绝去抓取一个框架。拒绝挑选一个会逼你为你真正需要的东西去和它的 state 模型搏斗的框架。

## 决策矩阵

| 问题形态 | 首选框架 | 原因 |
|---------------|---------------------|-----|
| 带类型 state、有人工审批、长时运行的工作流 DAG | LangGraph | 一等公民的 state、checkpointer、中断与时间回溯。 |
| 带有不同角色的研究/写作流水线 | CrewAI（顺序式）或 LangGraph subgraph | 在 CrewAI 中「每个任务一个角色」表达起来很廉价；当分支变复杂时再用 LangGraph 扩展。 |
| 提议者-批评者或师生对话 | AutoGen | 双 agent 对话是它的原生形态。 |
| 带工具、会话与记忆的单 agent | Agno | 最薄的搭建、内置的存储与记忆。 |
| 带 reducer 的成千上万次并行扇出 | LangGraph + `Send` | 唯一拥有一等公民并行分发 API 的框架。 |
| 快速原型，不绑定框架 | 纯 Python + provider SDK | 没有框架就是最快的框架。 |

## 练习

1. **简单。** 取同一个任务 —— 「研究 Anthropic 的总部所在地，写一篇 200 词的简报，并引用来源」—— 分别用 LangGraph（四个节点：plan、search、write、cite）和 CrewAI（三个角色：researcher、writer、editor）来实现。报告每次运行的 token 成本与代码行数。
2. **中等。** 用 AutoGen（researcher ↔ writer 对话，editor 通过 `GroupChat` 加入）和 Agno（一个带 `search_tools` 与 `write_tools` 的单 agent，外加一个会话存储）来构建同一个任务。从以下三方面给这四种实现排名：（a）每次运行的成本，（b）崩溃后恢复的能力，（c）在 write 步骤之前注入人工审批的能力。
3. **困难。** 构建一个决策树脚本 `pick_framework.py`，它接收一段简短的问题描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`）并返回一条带一句话理由的推荐。在你自己设计的六个用例上验证它。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|------|-----------------|-----------------------|
| Orchestration | 「agent 们如何协调」 | 决定下一个由哪个节点/角色/agent 运行的那一层。 |
| Durable state | 「重启后恢复」 | 在进程死亡后仍能存活的 state，挂接到一个检查点或会话存储上。 |
| LLM-selected routing | 「让模型决定」 | 一个规划器 LLM 在每一轮挑选下一步；灵活，但每次决策都要付出 token。 |
| Explicit routing | 「开发者决定」 | 由一个 Python 函数或静态边挑选下一步；廉价且可审计。 |
| Crew | 「一个 CrewAI 团队」 | 角色 + 任务 + 流程（顺序式或层级式）绑定成一个可运行单元。 |
| GroupChat | 「AutoGen 的多 agent 对话」 | 由一个发言者选择器管理的、N 个 agent 之间的对话。 |
| Team（Agno） | 「多 agent 的 Agno」 | 在一组 agent 之上的路由/协调/协作模式。 |
| StateGraph | 「LangGraph 的图」 | 带类型 state、节点、条件边、checkpointer 的抽象。 |

## 延伸阅读

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) —— StateGraph、checkpointer、中断、时间回溯。
- [CrewAI documentation](https://docs.crewai.com/) —— Crews、Flows、Agents、Tasks、Processes。
- [AutoGen documentation](https://microsoft.github.io/autogen/) —— ConversableAgent、GroupChat、团队、工具。
- [Agno documentation](https://docs.agno.com/) —— Agent、Team、Workflow、存储、记忆。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) —— 与框架无关的模式库（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) —— 每个框架都在精心包装的那个循环。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) —— AutoGen 的设计论文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) —— CrewAI 式人设栈所依托的角色扮演基础。
- Phase 11 · 16（LangGraph）—— 本课用来作为基准对比的那个框架。
- Phase 11 · 19（Reflexion）—— 一个能干净地映射到 LangGraph、却别扭地映射到 CrewAI 的模式。
- Phase 11 · 22（Production observability）—— 如何为你所挑选的那个框架做埋点。
