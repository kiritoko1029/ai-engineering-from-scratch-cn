# OpenTelemetry GenAI 语义约定

> OpenTelemetry 的 GenAI SIG（2024 年 4 月启动）定义了智能体遥测的标准模式。Span 名称、属性和内容捕获规则在各厂商间趋同，使得智能体追踪在 Datadog、Grafana、Jaeger 和 Honeycomb 中具有相同含义。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 24（可观测性平台）
**所需时间：** 约60分钟

## 学习目标

- 说出 GenAI span 的类别：模型/客户端、智能体、工具。
- 区分 `invoke_agent` 的 CLIENT 与 INTERNAL span 及其适用场景。
- 列出顶层 GenAI 属性：提供商名称、请求模型、数据源 ID。
- 解释内容捕获契约：选择性启用、`OTEL_SEMCONV_STABILITY_OPT_IN`、外部引用推荐。

## 问题所在

每个厂商都发明自己的 span 名称。运维团队最终要为每个框架构建独立的仪表盘。OpenTelemetry 的 GenAI SIG 通过定义一个整个生态系统共同遵循的标准来解决这个问题。

## 概念说明

### Span 类别

1. **模型/客户端 span。** 覆盖原始 LLM 调用。由提供商 SDK（Anthropic、OpenAI、Bedrock）和框架模型适配器发出。
2. **智能体 span。** `create_agent`（智能体构建时）和 `invoke_agent`（运行时）。
3. **工具 span。** 每次工具调用一个；通过父子关系连接到智能体 span。

### 智能体 span 命名

- Span 名称：如果已命名则为 `invoke_agent {gen_ai.agent.name}`；否则回退到 `invoke_agent`。
- Span 类型：
  - **CLIENT** — 用于远程智能体服务（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL** — 用于进程内智能体框架（LangChain、CrewAI、本地 ReAct）。

### 关键属性

- `gen_ai.provider.name` — `anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model` — 模型 ID。
- `gen_ai.response.model` — 解析后的模型（可能因路由而与请求不同）。
- `gen_ai.agent.name` — 智能体标识符。
- `gen_ai.operation.name` — `chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id` — 用于 RAG：查询了哪个语料库或存储。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 各有其特定技术约定。

### 内容捕获

默认规则：检测工具默认不应捕获输入/输出。捕获通过以下方式选择性启用：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

推荐的生产模式：将内容存储在外部（S3、你的日志存储），在 span 上记录引用（指针 ID，而非文本内容）。这是第 27 课中防止内容投毒的防御措施在可观测性中的体现。

### 稳定性

截至 2026 年 3 月，大多数约定仍为实验性。通过以下方式启用稳定预览：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 将 GenAI 属性原生映射到其 LLM Observability 模式。其他后端（Grafana、Honeycomb、Jaeger）支持原始属性。

### 这个模式出问题的地方

- **在 span 中捕获完整提示词。** PII、密钥、客户数据出现在运维人员可读的追踪中。应外部存储。
- **缺少 `gen_ai.provider.name`。** 多提供商仪表盘在归因缺失时会失效。
- **Span 缺少父链接。** 孤立的工具 span。务必传播上下文。
- **未设置稳定性选择性启用。** 后端升级时你的属性可能被重命名。

## 开始构建

`code/main.py` 实现了一个符合 GenAI 约定的标准库 span 发射器：

- 带有 GenAI 属性模式的 `Span`。
- 带有 `start_span` 和嵌套上下文的 `Tracer`。
- 脚本化的智能体运行，发射：`create_agent`、`invoke_agent`（INTERNAL）、逐工具 span、用于 LLM 调用的 `chat` span。
- 内容捕获模式，将提示词外部存储并在 span 上记录 ID。

运行方式：

```
python3 code/main.py
```

输出：一棵包含所有必需 GenAI 属性的 span 树，以及一个展示选择性内容引用的"外部存储"。

## 使用建议

- **Datadog LLM Observability**（v1.37+）原生映射属性。
- **Langfuse / Phoenix / Opik**（第 24 课）— 自动检测生态系统。
- **Jaeger / Honeycomb / Grafana Tempo** — 原始 OTel 追踪；从 GenAI 属性构建仪表盘。
- **自托管** — 使用 GenAI 处理器运行 OTel Collector。

## 交付产物

`outputs/skill-otel-genai.md` 将 OTel GenAI span 接入现有智能体，配置内容捕获默认值和外部引用存储。

## 练习

1. 用 `invoke_agent`（INTERNAL）+ 逐工具 span 检测你的第 01 课 ReAct 循环。发送到 Jaeger 实例。
2. 以"仅引用"模式添加内容捕获：提示词存入 SQLite，span 属性仅携带行 ID。
3. 阅读 `gen_ai.data_source.id` 的规范。将其接入你的第 09 课 Mem0 搜索。
4. 设置 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 并验证你的属性不会被收集器重命名。
5. 构建一个仪表盘：仅从 GenAI 属性分析"哪些工具错误与哪些模型相关"。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| GenAI SIG | "OpenTelemetry GenAI 组" | 定义该模式的 OTel 工作组 |
| invoke_agent | "智能体 span" | 表示智能体运行的 span 名称 |
| CLIENT span | "远程调用" | 调用远程智能体服务的 span |
| INTERNAL span | "进程内" | 进程内智能体运行的 span |
| gen_ai.provider.name | "提供商" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG 源" | 检索命中了哪个语料库/存储 |
| 内容捕获 | "提示词日志" | 选择性捕获消息；生产环境外部存储 |
| 稳定性选择性启用 | "预览模式" | 固定实验性约定的环境变量 |

## 延伸阅读

- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — 默认启用 GenAI span
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — 内置 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C 追踪上下文传播
