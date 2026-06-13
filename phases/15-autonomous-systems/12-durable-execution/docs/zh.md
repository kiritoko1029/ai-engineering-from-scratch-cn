# 长时运行后台智能体：持久化执行

> 生产中的长时域智能体不在 `while True` 中运行。每个 LLM 调用成为一个带有检查点、重试和重放的活动。Temporal 的 OpenAI Agents SDK 集成于 2026 年 3 月正式发布。Claude Code Routines（Anthropic）运行定时的 Claude Code 调用，无需持久化的本地进程。会话在人工输入时暂停，经受住部署，并从按 `thread_id` 键控的最新检查点恢复。新人体工学的背后是一个旧模式——工作流编排——加上一个新输入：LLM 调用作为非确定性活动，必须在恢复时确定性地重放。

**类型：** 学习
**语言：** Python（标准库，最小持久化执行状态机）
**前置要求：** 第 15 阶段 · 10（权限模式），第 15 阶段 · 01（长时域智能体）
**所需时间：** 约60分钟

## 问题所在

考虑一个运行四小时的智能体。它调用三个工具，提示用户两次，进行四十次 LLM 调用。运行到一半时，它所在的主机重启了。会发生什么？

- 在朴素的 `while True` 循环中：一切都丢失了。运行从头重启。三个工具调用（有真实副作用）再次执行。用户被再次提示他们已经批准的内容。四十次 LLM 调用被重新计费。
- 使用持久化执行：运行从最近的检查点恢复。已完成的活动不会重新执行；它们的结果从持久化日志中重放。用户不会重新批准已经批准的内容。已经进行的 LLM 调用不会被重新计费。

这是工作流引擎已经交付了十年的相同模式（Temporal、Cadence、Uber 的 Cherami）。新的是 LLM 调用现在是一种活动——非确定性、昂贵、有副作用——它们干净地契合这个模式。

本课的贯穿主题：长时域可靠性衰减（METR 观察到"35 分钟退化"——成功率随时间域大致二次方下降）。持久化执行使运行时间超过可靠性配置文件支持的时间，如果设计正确，这是一种安全失败的新方式；如果设计错误，则是不安全的。

## 概念说明

### 活动、工作流和重放

- **工作流**：确定性编排代码。定义活动的序列、分支、等待。必须是确定性的，以便从事件日志重放时不会出现意外分歧。
- **活动**：一个非确定性的、可能失败的工作单元。LLM 调用、工具调用、文件写入、HTTP 请求。每个活动记录其输入和（完成后）输出。
- **事件日志**：持久化后备存储。每个活动的开始、完成、失败、重试以及每个工作流决策都被记录。
- **重放**：恢复时，工作流代码从头重新运行；每个已完成的活动返回其记录的结果而不重新执行。只有尚未完成的活动才实际运行。

这与 React 对虚拟 DOM 的重新渲染或 Git 从提交重建工作树的形状相同。编排器中的确定性使得持久化廉价。

### 为什么 LLM 调用契合这个模式

LLM 调用是：
- 非确定性的（temperature > 0；即使是 temperature 0 在模型版本间也会漂移）。
- 昂贵的（金钱和延迟）。
- 可能失败的（速率限制、超时）。
- 有副作用的（如果它们调用工具）。

这正是活动的特征。将每个 LLM 调用包装为活动给你带指数退避的重试、跨重启的检查点以及可重放的调试跟踪。

### 按 `thread_id` 键控的检查点

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 和 Claude Code Routines 都收敛到相同的 API 形状：`thread_id`（或等价物）标识会话；每个状态转换持久化到后端（PostgreSQL 默认，SQLite 用于开发，Redis 用于缓存）；恢复读取最新检查点。

后端选择很重要：

- **PostgreSQL**：持久化、可查询、经受住部署。LangGraph 的默认。
- **SQLite**：仅本地开发；跨主机丢失数据。
- **Redis**：快速但临时，除非配置了 AOF/快照。
- **Cloudflare Durable Objects**：透明分布式；按唯一键限定范围；持续数小时到数周。

### 人工输入作为一等状态

先提议后提交（第 15 课）需要一个持久化的"等待人工"状态。工作流暂停，外部队列持有待处理请求，批准从恰好该点恢复。没有持久化这是尽力而为；有了它，隔夜的批准到达后工作流在早上继续。

### 35 分钟退化

METR 观察到每个被测量的智能体类别在连续运行超过约 35 分钟后都显示可靠性衰减。任务时长翻倍大致使失败率增加四倍。持久化执行不修复这个问题；它让你运行得比可靠性配置文件支持的更久。安全模式是将持久化与在重新进入时需要新的人机协同的检查点结合，以及与预算紧急停止开关（第 13 课）结合，以限制总计算量而不受挂钟时间影响。

### 持久化执行是错误答案的情况

- 几分钟内无人工输入的运行。开销 > 收益。
- 严格的只读信息检索。
- 正确性要求在一个上下文窗口内端到端完成的任务（某些推理任务；某些一次性生成）。

```figure
memory-consolidation
```

## 开始构建

`code/main.py` 用标准库 Python 实现了一个最小的持久化执行引擎。它支持：

- `@activity` 装饰器，将输入和输出记录到 JSON 事件日志。
- 一个对活动排序的工作流函数。
- 一个 `run_or_replay(workflow, event_log)` 函数，重放已完成的活动而不重新执行它们。

驱动器模拟一个三活动工作流，在运行到一半时崩溃，展示（a）朴素重试重新执行所有内容 vs（b）重放仅运行缺失的活动。

## 交付产出

`outputs/skill-durable-execution-review.md` 审查拟议的长时运行智能体部署的正确持久化执行形状：活动、确定性、检查点后端、人工输入状态以及恢复时的人机协同策略。

## 练习

1. 运行 `code/main.py`。观察朴素重试和重放之间活动执行计数的差异。更改崩溃点并展示重放计数相应变化。

2. 将玩具引擎转换为显式使用 `thread_id`。模拟两个并发会话共享引擎并确认它们的事件日志不冲突。

3. 取玩具引擎中的一个活动。引入非确定性（工作流决策中的挂钟时间戳）。演示重放时的分歧。解释真实引擎如何处理这个问题（副作用注册、`Workflow.now()` API）。

4. 阅读 LangChain 的"生产深度智能体背后的运行时"帖子。列出运行时持久化的每个状态并说出每个覆盖的故障模式。

5. 为一个 6 小时自主编码任务设计检查点策略。你在哪里设置检查点？崩溃后恢复是什么样子？什么需要新的人机协同？

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|---|---|---|
| Workflow（工作流） | "智能体的脚本" | 确定性编排代码；可从事件日志重放 |
| Activity（活动） | "一个步骤" | 非确定性单元（LLM 调用、工具调用）；前后记录 |
| Event log（事件日志） | "后备存储" | 每个状态转换的持久化记录 |
| Replay（重放） | "恢复" | 重新运行工作流；已完成的活动返回记录的结果而不重新执行 |
| Checkpoint（检查点） | "存档点" | 按 thread_id 键控的持久化状态；恢复时最新的获胜 |
| thread_id | "会话键" | 限定持久化范围的标识符 |
| 35-minute degradation（35 分钟退化） | "可靠性衰减" | METR：成功率随时间域大致二次方下降 |
| Non-determinism（非确定性） | "重放漂移" | 挂钟、随机、LLM 输出；必须注册为副作用 |

## 延伸阅读

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 预算、回合和恢复语义。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent 形状。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具体运行时要求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM 调用的活动形状。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35 分钟退化参考。
