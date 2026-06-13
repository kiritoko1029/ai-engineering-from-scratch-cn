# AutoGen v0.4：Actor 模型与智能体框架

> AutoGen v0.4（Microsoft Research，2025 年 1 月）围绕 Actor 模型重新设计了智能体编排。异步消息交换、事件驱动智能体、故障隔离、天然并发。该框架目前处于维护模式，而 Microsoft Agent Framework（2025 年 10 月公开预览）成为继任者。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 12（工作流模式）
**所需时间：** 约75分钟

## 学习目标

- 描述 Actor 模型：智能体作为 Actor、消息作为唯一 IPC、每个 Actor 的故障隔离。
- 说出 AutoGen v0.4 的三个 API 层——Core、AgentChat、Extensions——及其各自用途。
- 解释为什么将消息传递与处理解耦能实现故障隔离和天然并发。
- 用 Python 实现一个标准库 Actor 运行时，并将一个双智能体代码审查流程移植到其上。

## 问题所在

大多数智能体框架是同步的：一个智能体产出，一个智能体消费，在一个调用栈中。故障会崩溃栈。并发是外加的。分发需要重写。

AutoGen v0.4 的答案：Actor 模型。每个智能体是一个带私有收件箱的 Actor。消息是唯一的交互。运行时将传递与处理解耦。故障隔离到一个 Actor。并发是原生的。分发只是不同的传输。

## 概念说明

### Actor

一个 Actor 有：

- 一个私有状态（外部永远不能直接访问）。
- 一个收件箱（消息队列）。
- 一个处理器：`receive(message) -> effects`，其中效果可以是"回复"、"发送给其他 Actor"、"生成新 Actor"、"更新状态"、"停止自身"。

两个 Actor 不能共享内存。它们只能发送消息。

### AutoGen v0.4 的三个 API 层

1. **Core。** 底层 Actor 框架。`AgentRuntime`、`Agent`、`Message`、`Topic`。异步消息交换，事件驱动。
2. **AgentChat。** 任务驱动的高层 API（v0.2 ConversableAgent 的替代品）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 集成——OpenAI、Anthropic、Azure、工具、记忆。

### 为什么解耦很重要

在 v0.2 模型中，调用 `agent_a.chat(agent_b)` 同步阻塞 agent_a 直到 agent_b 返回。在 v0.4 中，`send(agent_b, msg)` 将消息放入 agent_b 的收件箱并返回。运行时稍后传递。三个结果：

- **故障隔离。** Agent B 崩溃不会崩溃 Agent A——运行时在 B 的处理器中捕获故障并决定做什么（记录、重试、死信）。
- **天然并发。** 许多消息同时在飞；Actor 并发处理其收件箱。
- **分发就绪。** 收件箱 + 传输是相同的抽象，无论 Actor 是进程内还是在另一台主机上。

### 拓扑

- **RoundRobinGroupChat。** 智能体按固定轮转轮流。
- **SelectorGroupChat。** 选择器智能体根据对话上下文选择下一个。
- **Magentic-One。** 用于网页浏览、代码执行、文件处理的参考多智能体团队。基于 AgentChat 构建。

### 可观测性

内置 OpenTelemetry 支持。每个消息发出一个 span；工具调用按 2026 年 OTel GenAI 语义约定（第 23 课）携带 `gen_ai.*` 属性。

### 状态：维护模式

2026 年初：AutoGen v0.7.x 对研究和原型设计稳定。Microsoft 已将活跃开发转移到 Microsoft Agent Framework（2025 年 10 月 1 日公开预览；1.0 GA 目标 2026 年 Q1 末）。AutoGen 模式向前移植很干净——Actor 模型是持久的思想。

## 开始构建

`code/main.py` 实现了一个标准库 Actor 运行时：

- `Message`——带 `sender`、`recipient`、`topic`、`body` 的类型化载荷。
- `Actor`——带 `receive(message, runtime)` 的抽象。
- `Runtime`——带共享队列、传递、故障隔离的事件循环。
- 一个双 Actor 演示：`ReviewerAgent` 审查代码，`ChecklistAgent` 运行检查清单；它们交换消息直到达成共识。

运行它：

```
python3 code/main.py
```

轨迹展示了消息传递、一个 Actor 中的模拟故障不会崩溃另一个、以及对共享裁决的收敛。

## 使用它

- **AutoGen v0.4/v0.7**（维护模式）——对研究、原型设计、多智能体模式稳定。
- **Microsoft Agent Framework**（公开预览）——前进路径；在刷新的 API 中相同的 Actor 模型思想。
- **LangGraph 群体拓扑**（第 13 课）——通过共享工具移交的类似模式。
- **自定义 Actor 运行时**——当你需要特定传输（NATS、RabbitMQ、gRPC）时。

## 交付它

`outputs/skill-actor-runtime.md` 为给定多智能体任务生成一个最小 Actor 运行时加团队模板（RoundRobin 或 Selector）。

## 练习

1. 添加死信队列：当处理器抛异常时，停放失败消息供人工检查。在你的简易实现中 DLQ 被触发的频率如何？
2. 实现 `SelectorGroupChat`：选择器 Actor 根据对话状态选择谁处理下一条消息。
3. 添加分布式传输：将进程内队列替换为 JSON-over-HTTP 服务器，使 Actor 可以在不同进程中运行。
4. 为每条消息接入 OTel span（或无操作替代品）。按第 23 课发出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 阅读 AutoGen v0.4 的架构文章。将你的简易实现移植到真实的 `autogen_core` API。你跳过了什么在生产中重要的东西？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Actor | "智能体" | 私有状态 + 收件箱 + 处理器；无共享内存 |
| 消息（Message） | "事件" | 类型化载荷；Actor 交互的唯一方式 |
| 收件箱（Inbox） | "邮箱" | 每个 Actor 的待处理消息队列 |
| 运行时（Runtime） | "智能体宿主" | 路由消息和隔离故障的事件循环 |
| 主题（Topic） | "通道" | Actor 间的命名发布-订阅路由 |
| 故障隔离（Fault isolation） | "让它崩溃" | 一个 Actor 失败不会崩溃其他 Actor |
| RoundRobinGroupChat | "固定轮转团队" | 智能体按顺序轮流 |
| SelectorGroupChat | "上下文路由团队" | 选择器选择下一个 |
| Magentic-One | "参考团队" | 用于网页 + 代码 + 文件的多智能体小队 |

## 延伸阅读

- [AutoGen v0.4，Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)——重设计文章
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)——图形状的替代方案
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——AutoGen 默认发出的 span
