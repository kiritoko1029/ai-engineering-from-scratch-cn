# OpenTelemetry GenAI —— 端到端追踪工具调用

> 一个智能体调用五个工具、三个 MCP 服务器和两个子智能体。你需要一个跨越所有这些的追踪。OpenTelemetry GenAI 语义约定（v1.37 及以上版本中的稳定属性）是 2026 年的标准，Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 原生支持。本课列出必需的属性，走通 span 层次结构（智能体 → LLM → 工具），并提供一个可插入任何 OTel 导出器的标准库 span 发射器。

**类型：** 构建
**语言：** Python（标准库，OTel span 发射器）
**前置要求：** Phase 13 · 07（MCP 服务器），Phase 13 · 08（MCP 客户端）
**所需时间：** 约75分钟

## 学习目标

- 列出 LLM span 和工具执行 span 的必需 OTel GenAI 属性。
- 构建覆盖智能体循环、LLM 调用、工具调用和 MCP 客户端调度的追踪层次结构。
- 决定哪些内容需要捕获（选择加入）vs 脱敏（默认）。
- 将 span 发射到本地收集器（Jaeger、Langfuse），无需重写工具代码。

## 问题所在

2026 年 2 月的一个调试：用户报告"我的智能体有时需要 30 秒响应；其他时候 3 秒"。没有追踪。日志显示 LLM 调用，但没有工具调度、没有 MCP 服务器往返、没有子智能体。你只能猜测。最终你发现：一个 MCP 服务器偶尔在冷启动时挂起。

没有端到端追踪，你无法发现这个问题。OTel GenAI 修复了它。

这些约定在 2025-2026 年由 OpenTelemetry 语义约定组确定。它们定义了稳定的属性名称，使 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 span。一次插桩；发送到任何后端。

## 概念说明

### Span 层次结构

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

所有内容嵌套在一个 trace id 下。Span id 链接父子关系。

### 必需属性

根据 2025-2026 语义约定：

- `gen_ai.operation.name` —— `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` —— `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` —— 请求的模型字符串（如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` —— 实际提供的模型。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` —— 提供商响应 ID，用于关联。

工具 span：

- `gen_ai.tool.name` —— 工具标识符。
- `gen_ai.tool.call.id` —— 特定调用 ID。
- `gen_ai.tool.description` —— 工具描述（可选）。

智能体 span：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 类型

- `SpanKind.CLIENT` 用于跨越进程边界的调用（LLM 提供商、MCP 服务器）。
- `SpanKind.INTERNAL` 用于智能体自身的循环步骤和工具执行。

### 选择加入内容捕获

默认情况下，span 携带指标和计时 —— 而非提示或补全。大型载荷和 PII 默认关闭。设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的内容捕获环境变量以包含内容。在生产环境中启用前请仔细审查。

### Span 上的事件

Token 级事件可以作为 span 事件添加：

- `gen_ai.content.prompt` —— 输入消息。
- `gen_ai.content.completion` —— 输出消息。
- `gen_ai.content.tool_call` —— 记录的工具调用。

事件在 span 内按时间排序，用于详细回放。

### 导出器

OTel span 导出到：

- **Jaeger / Tempo。** 开源，本地部署。
- **Langfuse。** LLM 可观测性专用；可视化 token 使用量。
- **Arize Phoenix。** 评估 + 追踪结合。
- **Datadog。** 商业；原生解析 `gen_ai.*` 属性。
- **Honeycomb。** 列式存储；查询友好。

所有都使用 OTLP，即线路格式。你的代码无需关心。

### 跨 MCP 传播

当 MCP 客户端调用服务器时，将 W3C traceparent 头注入请求中。Streamable HTTP 支持标准头。Stdio 不原生携带 HTTP 头；规范的 2026 路线图讨论在 JSON-RPC 调用上添加 `_meta.traceparent` 字段。

在那之前：手动将 traceparent 包含在每个请求的 `_meta` 中。服务器记录 trace id。

### 指标

除了 span，GenAI 语义约定还定义了指标：

- `gen_ai.client.token.usage` —— 直方图。
- `gen_ai.client.operation.duration` —— 直方图。
- `gen_ai.tool.execution.duration` —— 直方图。

用于不需要逐调用详情的仪表板。

### AgentOps 层

AgentOps（2024 年成立）专注于 GenAI 可观测性。它包装流行框架（LangGraph、Pydantic AI、CrewAI）自动发射 OTel span。如果你的栈使用受支持的框架很有用；否则使用手动插桩。

## 开始构建

`code/main.py` 以 OTLP-JSON 类似格式向 stdout 发射 OTel 形状的 span，用于一个调用 LLM、调度两个工具并进行一次 MCP 往返的智能体。没有真实的导出器 —— 课程聚焦于 span 形状和属性集。将输出粘贴到兼容 OTLP 的查看器中或直接阅读。

需要关注的要点：

- 所有 span 共享一个 trace id。
- 父子链接通过 `parentSpanId` 编码。
- 必需的 `gen_ai.*` 属性已填充。
- 内容捕获默认关闭；一个场景通过环境变量开启。

## 交付产出

本课生成 `outputs/skill-otel-genai-instrumentation.md`。给定一个智能体代码库，该技能生成插桩计划：在哪里添加 span、填充哪些属性、面向哪些导出器。

## 练习

1. 运行 `code/main.py`。计算 span 数量并识别哪些是 CLIENT vs INTERNAL。

2. 开启内容捕获（环境变量），确认 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出现。注意对 PII 的影响。

3. 添加工具执行指标 `gen_ai.tool.execution.duration`，作为每次调用的直方图样本发射。

4. 将 traceparent 从父智能体 span 传播到 MCP 请求的 `_meta.traceparent` 字段。验证 MCP 服务器会看到相同的 trace id。

5. 阅读 OTel GenAI 语义约定规范。找出语义约定中列出但本课代码未发射的一个属性。添加它。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| OTel | "OpenTelemetry" | 追踪、指标、日志的开放标准 |
| GenAI 语义约定 | "GenAI semantic conventions" | LLM / 工具 / 智能体 span 的稳定属性名 |
| `gen_ai.*` | "属性命名空间" | 所有 GenAI 属性共享此前缀 |
| Span | "计时操作" | 具有开始、结束和属性的工作单元 |
| Trace | "跨 span 祖先" | 共享 trace id 的 span 树 |
| SpanKind | "CLIENT / SERVER / INTERNAL" | span 方向提示 |
| OTLP | "OpenTelemetry Line Protocol" | 导出器的线路格式 |
| 选择加入内容 | "提示/补全捕获" | 默认关闭；环境变量启用 |
| traceparent | "W3C 头" | 跨服务传播追踪上下文 |
| Exporter | "特定后端的发送器" | 将 span 发送到 Jaeger / Datadog 等的组件 |

## 延伸阅读

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span、指标和事件的规范约定
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 和工具执行 span 属性列表
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — 智能体级 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 托管的真实来源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — 生产集成讲解
