# 函数调用深入解析 -- OpenAI、Anthropic、Gemini

> 三大前沿供应商在 2024 年收敛到了相同的工具调用循环，但在其他所有方面都各不相同。OpenAI 使用 `tools` 和 `tool_calls`，Anthropic 使用 `tool_use` 和 `tool_result` 块，Gemini 使用 `functionDeclarations` 和唯一 id 关联。本课将三者并排对比，让你在一个供应商上编写的代码移植到另一个供应商时不会出错。

**类型：** 构建
**语言：** Python（标准库，模式转换器）
**前置要求：** 第 13 阶段第 01 课（工具接口）
**所需时间：** 约75分钟

## 学习目标

- 说明 OpenAI、Anthropic 和 Gemini 函数调用载荷（声明、调用、结果）的三个形式差异。
- 将一个工具声明转换为三种供应商格式，并预测严格模式约束在何处会有差异。
- 在每个供应商中使用 `tool_choice` 来强制、禁止或自动选择工具调用。
- 了解每个供应商的硬性限制（工具数量、模式深度、参数长度）以及超出限制时发出的错误信号。

## 问题所在

函数调用请求的形式因供应商而异。以下是来自 2026 年生产技术栈的三个具体示例：

**OpenAI Chat Completions / Responses API。** 你传入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的响应包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是一个需要你解析的 JSON 字符串。严格模式（`strict: true`）通过约束解码来强制遵守模式。

**Anthropic Messages API。** 你传入 `tools: [{name, description, input_schema}]`。响应以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式返回。`input` 已经是解析后的对象（不是字符串）。你用一条包含 `{type: "tool_result", tool_use_id, content}` 块的新 `user` 消息来回复。

**Google Gemini API。** 你传入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（嵌套在 `functionDeclarations` 下）。响应以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式到达，其中 `id` 在 Gemini 3 及以上版本中是唯一的，用于并行调用关联。你用 `{functionResponse: {name, id, response}}` 来回复。

相同的循环。不同的字段名、不同的嵌套结构、不同的字符串与对象约定、不同的关联机制。一个在 OpenAI 上编写天气智能体的团队，移植到 Anthropic 需要两天，移植到 Gemini 再需要一天，这些时间都花在了管道代码上。

本课构建一个转换器，将三种格式统一为一种规范的工具声明，并在边界层进行路由。第 13 阶段第 17 课将同样的模式泛化为 LLM 网关。

## 概念说明

### 共同结构

每个供应商都需要五样东西：

1. **工具列表。** 每个工具的名称、描述和输入模式。
2. **工具选择（tool_choice）。** 强制使用特定工具、禁止工具、或让模型自行决定。
3. **调用输出。** 指明工具名称和参数的结构化输出。
4. **调用 id。** 将响应与正确的调用关联（在并行场景中很重要）。
5. **结果注入。** 将结果与调用绑定的消息或块。

### 形式差异，逐字段对比

| 方面 | OpenAI | Anthropic | Gemini |
|------|--------|-----------|--------|
| 声明封装 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| 模式字段 | `parameters` | `input_schema` | `parameters` |
| 响应容器 | 助手消息上的 `tool_calls[]` | 类型为 `tool_use` 的 `content[]` | 类型为 `functionCall` 的 `parts[]` |
| 参数类型 | JSON 字符串 | 已解析的对象 | 已解析的对象 |
| id 格式 | `call_...`（OpenAI 生成） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| 结果块 | 角色 `tool`，`tool_call_id` | `user` 消息中包含 `tool_result`，`tool_use_id` | `functionResponse` 匹配 `id` |
| 强制工具 | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止工具 | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| 严格模式 | `strict: true` | 模式即契约（始终强制） | 请求级别的 `responseSchema` |

### 你实际会遇到的限制

- **OpenAI。** 每个请求最多 128 个工具。模式深度 5 层。参数字符串 <= 8192 字节。严格模式要求不能有 `$ref`，不能有重叠的 `oneOf`/`anyOf`/`allOf`，每个属性都必须列在 `required` 中。
- **Anthropic。** 每个请求最多 64 个工具。模式深度实际上无限制，但实际限制为 10 层。没有严格模式标志；模式就是契约，模型倾向于遵守。
- **Gemini。** 每个请求最多 64 个函数。模式类型是 OpenAPI 3.0 的子集（与 JSON Schema 2020-12 略有差异）。Gemini 3 起并行调用使用唯一 id。

### `tool_choice` 行为

每个供应商都支持三种模式，只是名称不同。

- **Auto（自动）。** 模型自行选择工具或文本。默认模式。
- **Required / Any（必须）。** 模型必须至少调用一个工具。
- **None（禁止）。** 模型不得调用工具。

此外每个供应商还有一种独特模式：

- **OpenAI。** 按名称强制使用特定工具。
- **Anthropic。** 按名称强制使用特定工具；`disable_parallel_tool_use` 标志区分单个与多个。
- **Gemini。** `mode: "VALIDATED"` 无论模型意图如何，都将每个响应通过模式验证器路由。

### 并行调用

OpenAI 的 `parallel_tool_calls: true`（默认）在一个助手消息中输出多个调用。你运行所有调用，然后用一条包含每个 `tool_call_id` 对应条目的批量工具角色消息回复。Anthropic 历史上是单调用；`disable_parallel_tool_use: false`（Claude 3.5 起默认）启用了多调用。Gemini 2 允许并行调用但没有提供稳定 id；Gemini 3 添加了 UUID，使乱序响应可以正确关联。

### 流式传输

三家都支持流式工具调用。线路格式不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的增量块逐步到达。你累积数据直到 `finish_reason: "tool_calls"`。
- **Anthropic。** 块开始 / 块增量 / 块停止事件。`input_json_delta` 块携带部分参数。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）输出带有 `functionCallId` 的块，使多个并行调用可以交错进行。

第 13 阶段第 03 课深入讲解并行 + 流式重组。本课聚焦于声明和单调用的形式。

### 错误和修复

无效参数的错误表现也不同。

- **OpenAI（非严格）。** 模型返回 `arguments: "{bad json}"`，你的 JSON 解析失败，你注入错误消息并重新调用。
- **OpenAI（严格）。** 验证在解码过程中发生；无效 JSON 不可能出现，但 `refusal` 可能会出现。
- **Anthropic。** `input` 可能包含意外字段；模式仅供参考。需要在服务端验证。
- **Gemini。** OpenAPI 3.0 的怪癖：对象字段上的 `enum` 会被静默忽略；需要自行验证。

### 转换器模式

代码中的规范工具声明如下（你来选择形式）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三个小函数将其转换为三种供应商格式。`code/main.py` 中的脚手架正是这样做的，然后通过每种供应商的响应形式往返一个模拟工具调用。不需要网络 -- 本课教授的是形式，不是 HTTP。

生产团队将这个转换器封装在 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）中。第 13 阶段第 17 课发布的网关在任何一种供应商前面暴露 OpenAI 形式的 API。

## 开始构建

`code/main.py` 定义了一个规范的 `Tool` 数据类和三个转换器，分别输出 OpenAI、Anthropic 和 Gemini 的声明 JSON。然后它解析每种供应商形式的手工响应到同一个规范调用对象，证明表层之下语义是完全相同的。运行它，并排对比三种声明。

需要关注的要点：

- 三个声明块仅在封装和字段名上不同。
- 三个响应块在调用所在位置不同（顶层 `tool_calls`、`content[]` 块、`parts[]` 条目）。
- 一个 `canonical_call()` 函数从三种响应形式中提取 `{id, name, args}`。

## 发布成果

本课生成 `outputs/skill-provider-portability-audit.md`。给定一个针对某个供应商的函数调用集成，该技能会生成可移植性审计报告：依赖了哪些供应商限制、哪些字段需要重命名、以及移植到其他供应商时什么会出问题。

## 练习

1. 运行 `code/main.py`，验证三个供应商的声明 JSON 都序列化了同一个底层 `Tool` 对象。修改规范工具以添加一个 enum 参数，确认只有 Gemini 转换器需要处理 OpenAPI 的怪癖。

2. 为每个供应商添加一个 `ListToolsResponse` 解析器，提取模型在 `list_tools` 或发现调用后返回的工具列表。OpenAI 原生没有这个功能；注意这种不对称性。

3. 实现 `tool_choice` 转换：将规范的 `ToolChoice(mode="force", tool_name="x")` 映射到三种供应商格式。然后映射 `mode="any"` 和 `mode="none"`。对照本课的差异表格进行检查。

4. 选择三个供应商中的一个，从头到尾阅读其函数调用指南。找出其模式规范中其他两家不支持的一个字段。候选：OpenAI `strict`、Anthropic `disable_parallel_tool_use`、Gemini `function_calling_config.allowed_function_names`。

5. 编写一个测试向量：一个参数违反声明模式的工具调用。通过每个供应商的验证器运行它（第 01 课的标准库验证器可以作为代理），记录触发了哪些错误。记录你在生产环境中会选择哪个供应商来实现严格性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 函数调用（Function calling） | "工具使用" | 供应商级别的结构化工具调用输出 API |
| 工具声明（Tool declaration） | "工具规格" | 名称 + 描述 + JSON Schema 输入载荷 |
| `tool_choice` | "强制 / 禁止" | 自动 / 必须 / 禁止 / 指定名称模式 |
| 严格模式（Strict mode） | "模式强制" | OpenAI 标志，通过约束解码匹配模式 |
| `tool_use` 块 | "Anthropic 的调用形式" | 包含 id、name、input 的内联内容块 |
| `functionCall` part | "Gemini 的调用形式" | 包含 name、args 和 id 的 `parts[]` 条目 |
| 参数即字符串（Arguments-as-string） | "JSON 字符串化" | OpenAI 将参数作为 JSON 字符串返回，而非对象 |
| 并行工具调用（Parallel tool calls） | "一轮内扇出" | 一条助手消息中的多个工具调用 |
| 拒绝（Refusal） | "模型拒绝" | 严格模式专用的拒绝块，而非调用 |
| OpenAPI 3.0 子集 | "Gemini 模式怪癖" | Gemini 使用类似 JSON Schema 的方言，存在细微差异 |

## 延伸阅读

- [OpenAI -- Function calling guide](https://platform.openai.com/docs/guides/function-calling) -- 包含严格模式和并行调用的权威参考
- [Anthropic -- Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) -- `tool_use` 和 `tool_result` 块语义
- [Google -- Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) -- 并行调用、唯一 id 和 OpenAPI 子集
- [Vertex AI -- Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) -- Gemini 的企业接口
- [OpenAI -- Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) -- 严格模式模式强制详情
