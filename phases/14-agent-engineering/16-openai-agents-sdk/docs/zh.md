# OpenAI Agents SDK：Handoffs、Guardrails、Tracing

> OpenAI Agents SDK 是基于 Responses API 构建的轻量级多智能体框架。五个原语：Agent、Handoff、Guardrail、Session、Tracing。Handoffs 是名为 `transfer_to_<agent>` 的工具。Guardrails 在输入或输出上触发。Tracing 默认开启。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 06（工具使用）
**所需时间：** 约75分钟

## 学习目标

- 说出 OpenAI Agents SDK 的五个原语。
- 解释 Handoffs：为什么建模为工具、模型看到什么名称形状、上下文如何转移。
- 区分输入防护栏、输出防护栏和工具防护栏；解释 `run_in_parallel` vs 阻塞模式。
- 用标准库实现一个带 Handoffs + Guardrails + span 式 Tracing 的运行时。

## 问题所在

无法干净委派的智能体最终将一切塞进一个提示。没有防护栏的智能体会泄露 PII、违反策略的输出或无限循环。OpenAI 的 SDK 固化了使多智能体工作可控的三个原语。

## 概念说明

### 五个原语

1. **Agent。** LLM + 指令 + 工具 + Handoffs。
2. **Handoff。** 委派给另一个智能体。对模型表示为名为 `transfer_to_<agent_name>` 的工具。
3. **Guardrail。** 对输入（仅第一个智能体）、输出（仅最后一个智能体）或工具调用（每个函数工具）的验证。
4. **Session。** 跨轮次的自动对话历史。
5. **Tracing。** LLM 生成、工具调用、Handoffs、Guardrails 的内置 span。

### Handoffs 作为工具

模型在其工具列表中看到 `transfer_to_billing_agent`。调用它向运行时表示：

1. 复制对话上下文（或通过 `nest_handoff_history` beta 折叠）。
2. 用其指令初始化目标智能体。
3. 用目标智能体继续运行。

这就是主管模式（第 13 课 / 第 28 课）的产品化。

### Guardrails

三种风味：

- **输入防护栏。** 在第一个智能体的输入上运行。在任何 LLM 调用之前拒绝不安全或超出范围的请求。
- **输出防护栏。** 在最后一个智能体的输出上运行。捕获 PII 泄露、策略违规、格式错误的响应。
- **工具防护栏。** 每个函数工具运行。验证参数、检查权限、审计执行。

模式：

- **并行**（默认）。防护栏 LLM 与主 LLM 同时运行。更低的尾部延迟。如果触发，主 LLM 的工作被丢弃（token 浪费）。
- **阻塞**（`run_in_parallel=False`）。防护栏 LLM 先运行。如果触发，不浪费主调用的 token。

触发器抛出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

默认开启。每次 LLM 生成、工具调用、Handoff 和 Guardrail 发出一个 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 退出。`add_trace_processor(processor)` 将 span 扇出到你自己的后端，与 OpenAI 的并行。

### Sessions

`Session` 在后端（SQLite、Redis、自定义）存储对话历史。`Runner.run(agent, input, session=session)` 自动加载和追加。

### 此模式出错的地方

- **Handoff 漂移。** Agent A 移交给 Agent B，Agent B 又移交回 Agent A。添加跳数计数器。
- **Guardrail 绕过。** 工具防护栏仅对函数工具触发；内置工具（文件读取器、网页抓取）需要独立策略。
- **过度追踪。** span 中的敏感内容。配合 OTel GenAI 内容捕获规则（第 23 课）——外部存储，按 ID 引用。

## 开始构建

`code/main.py` 用标准库实现了 SDK 形状：

- `Agent`、`FunctionTool`、`Handoff`（作为带转移语义的函数工具）。
- `Runner`，带输入/输出/工具防护栏、Handoff 分派和跳数计数器。
- 一个简单的 span 发射器展示轨迹形状。
- 一个分诊智能体根据用户查询移交给 billing 或 support；一个输入防护栏触发。

运行它：

```
python3 code/main.py
```

轨迹展示了两次成功的 Handoff、一次输入防护栏触发和一个镜像真实 SDK 输出的 span 树。

## 使用它

- **OpenAI Agents SDK** 用于 OpenAI 优先的产品。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品。
- **LangGraph**（第 13 课）当你想要显式状态和持久恢复时。
- **自定义**当你需要精确控制（语音、多提供商、联邦部署）时。

## 交付它

`outputs/skill-agents-sdk-scaffold.md` 脚手架一个 Agents SDK 应用，带分诊智能体、Handoffs、输入/输出/工具防护栏、会话存储和轨迹处理器。

## 练习

1. 添加 Handoff 跳数计数器：N 次转移后拒绝。追踪行为。
2. 实现 `nest_handoff_history` 作为选项——在转移前将先前消息折叠为一个摘要。
3. 写一个阻塞输出防护栏。比较会触发的提示与通过的提示的延迟。
4. 将 `add_trace_processor` 接入 JSON 日志器。每个 span 它发出什么形状？
5. 阅读 SDK 文档。将你的标准库实现移植到 `openai-agents-python`。你建模错了什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Agent | "LLM + 指令" | SDK 中的 Agent 类型；拥有工具和 Handoffs |
| Handoff | "转移" | 模型调用以委派给另一个智能体的工具 |
| Guardrail | "策略检查" | 对输入/输出/工具调用的验证 |
| Tripwire | "防护栏触发" | 防护栏拒绝时抛出的异常 |
| Session | "历史存储" | 在运行之间持久化的对话记忆 |
| Tracing | "Spans" | LLM + 工具 + Handoff + Guardrail 的内置可观测性 |
| 阻塞防护栏（Blocking guardrail） | "顺序检查" | 防护栏先运行；触发时不浪费 token |
| 并行防护栏（Parallel guardrail） | "并发检查" | 防护栏同时运行；延迟更低，触发时浪费 token |

## 延伸阅读

- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/)——原语、Handoffs、Guardrails、Tracing
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview)——Claude 风格的对应物
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——何时需要 Handoffs
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)——Agents SDK span 映射到的标准
