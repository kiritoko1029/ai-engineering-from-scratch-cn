# 并行工具调用与工具流式传输

> 三次独立的天气查询串行执行就是三次往返。并行执行则将总时间压缩到最慢的那一次调用。如今每个前沿供应商都能在一轮中输出多个工具调用。收益是实实在在的；但管道代码却颇为微妙。本课将讲解两个方面：并行扇出和流式参数重组，重点关注 id 关联陷阱。

**类型：** 构建
**语言：** Python（标准库，线程池 + 流式脚手架）
**前置要求：** 第 13 阶段第 02 课（函数调用深入解析）
**所需时间：** 约75分钟

## 学习目标

- 解释为什么 `parallel_tool_calls: true` 存在以及何时应该禁用它。
- 在并行扇出期间将流式参数块关联到正确的工具调用 id。
- 在不提前解析的情况下将部分 `arguments` 字符串重组为完整的 JSON。
- 运行一个三城市天气基准测试，演示串行与并行的延迟差异。

## 问题所在

没有并行调用时，一个回答"班加罗尔、东京和苏黎世的天气怎么样"的智能体会这样做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，每次还要加上执行器延迟。大约是理想情况的 4 倍。

使用并行调用：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

一次 LLM 往返。执行器时间是三者中的最大值，而非总和。在 OpenAI、Anthropic 和 Gemini 上的生产基准测试表明，扇出工作负载的墙钟时间减少了 60% 到 70%。

代价是关联复杂度。当三个调用乱序完成时，你的结果必须携带匹配的 `tool_call_id`，这样模型才能正确排列。当结果流式返回时，你必须在执行之前将部分参数片段组装成完整的 JSON。Gemini 3 添加唯一 id，部分原因就是为了解决一个现实问题：两个并行调用同一个工具时无法区分。

## 概念说明

### 启用并行

- **OpenAI。** `parallel_tool_calls: true` 默认开启。设置 `false` 强制串行。
- **Anthropic。** 通过 `disable_parallel_tool_use: false` 实现并行（Claude 3.5 及以上默认）。设置 `true` 为串行。
- **Gemini。** 始终支持并行；`tool_config.function_calling_config.mode = "AUTO"` 让模型自行决定。

当工具存在顺序依赖时（先 `create_file` 再 `write_file`）、当一个调用的输出是另一个调用的输入时、或当速率限制器无法处理扇出时，禁用并行。

### id 关联

模型输出的每个调用都有一个 `id`。宿主返回的每个结果都必须包含相同的 id。没有这个，结果就是模糊的。

- **OpenAI。** 每条工具角色消息上的 `tool_call_id`。
- **Anthropic。** 每个 `tool_result` 块上的 `tool_use_id`。
- **Gemini。** 每个 `functionResponse` 上的 `id`（Gemini 3 及以上；Gemini 2 通过名称匹配，这在同名并行调用时会出问题）。

### 并发执行调用

宿主在各自的线程、协程或远程工作进程上运行每个调用的执行器。最简单的脚手架使用线程池；生产环境使用 `asyncio.gather` 或结构化并发。完成顺序不可预测 -- id 才是标识符。

一个常见 bug：按调用列表顺序而非完成顺序回复结果。这通常能工作，因为模型只关心 `tool_call_id`，但如果结果被丢弃或重复，乱序提交会使调试更困难。建议按完成顺序回复并显式标注 id。

### 流式工具调用

当模型流式输出时，`arguments` 分片到达。三个并行调用的三个独立流在网络上交错。你需要为每个 id 准备一个累加器。

各供应商的形式：

- **OpenAI。** 每个块是 `choices[0].delta.tool_calls[i].function.arguments`（部分字符串）。块携带 `index`（调用列表中的位置）。你按 index 累积，在首次出现时读取 `id`，在 `finish_reason = "tool_calls"` 时解析 JSON。
- **Anthropic。** 流事件为 `message_start`，然后每个块一个 `content_block_start`，类型为 `tool_use`（包含 id、name、空 input）。`content_block_delta` 事件携带 `input_json_delta` 块。`content_block_stop` 关闭每个块。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 及以上）输出带有 `functionCallId` 的块，使调用可以干净地交错。Gemini 3 之前，流式传输一次返回一个完整调用。

### 部分 JSON 和提前解析陷阱

在 `arguments` 完成之前你无法解析它。部分 JSON 如 `{"city": "Beng` 不是有效的，会抛出异常。正确的门控信号是供应商的调用结束信号：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`、或 Gemini 的流结束事件。只有到那时才尝试 `json.loads`。更稳健的方法使用增量 JSON 解析器，在结构完成时产出事件；OpenAI 的流式指南推荐这种做法用于显示实时"思考"指示器的 UX。大括号计数作为完整性测试是不可靠的（引号字符串或转义内容中的大括号会导致误报），应仅作为非正式的调试启发式方法使用。

### 乱序完成

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

宿主回复仍然必须引用这些 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 上，回复中的顺序对正确性没有影响。Gemini 接受任何顺序，只要 id 匹配。

### 基准测试：串行 vs 并行

`code/main.py` 中的脚手架模拟三个延迟分别为 400、600 和 800 毫秒的执行器。串行总共需要 1800 毫秒。并行需要 max(400, 600, 800) = 800 毫秒。差异是常数而非比例关系，因此工具数量越多，节省越大。

现实中的注意事项：并行调用会对下游 API 造成压力。10 路扇出到一个有速率限制的服务将会失败。第 13 阶段第 17 课涵盖网关级别的背压机制；重试语义计划在未来阶段中介绍。

### 流式扇出墙钟时间

如果模型本身是流式的，你可以在一个调用的参数完成后就开始执行，而不是等待所有调用最终完成。这是 OpenAI 文档中记录的一种优化，但并非所有 SDK 都暴露了它。本课的脚手架实现了这一点：一旦模拟流产出完整的参数对象，宿主就启动该调用。

## 开始构建

`code/main.py` 分为两部分。第一部分使用 `concurrent.futures.ThreadPoolExecutor` 串行和并行运行三个模拟天气调用，并打印墙钟时间。第二部分重放一个模拟的流式响应 -- 三个并行调用的 `arguments` 块在一个流上交错 -- 并使用 `StreamAccumulator` 按 id 重组它们。没有 LLM，没有网络，只有重组逻辑。

需要关注的要点：

- 串行计时器达到 1.8 秒。并行计时器在相同的模拟延迟下达到 0.8 秒。
- 累加器通过按 id 缓冲来处理乱序到达的块，只有在每个调用的 JSON 完成时才解析。
- 执行器在 id 的参数最终完成时就启动，而不是在所有流结束之后。

## 发布成果

本课生成 `outputs/skill-parallel-call-safety-check.md`。给定一个工具注册表，该技能审计哪些工具可以安全并行、哪些有顺序依赖、哪些会压垮下游速率限制 -- 返回一个带有每个工具 `parallel_safe` 标志的修订注册表。

## 练习

1. 运行 `code/main.py`，调整模拟延迟。确认并行与串行的比率大约是 `max/sum`（实际运行因线程调度、序列化和脚手架开销会略有偏差）。在什么延迟分布下并行不再有意义？

2. 扩展累加器以处理"调用在流中途被取消"的情况，丢弃其缓冲区并发出 `cancelled` 事件。哪个供应商明确文档化了这种情况？查看 Anthropic 的 `content_block_stop` 语义和 OpenAI 的 `finish_reason: "length"` 行为。

3. 用 `asyncio.gather` 替换线程池。对两者进行基准测试。你应该在异步上看到小幅提升，因为上下文切换成本更低，但前提是执行器做的是真正的 I/O。

4. 选择两个不应该并行的工具（例如先 `create_file` 再 `write_file`）。在注册表中添加一个 `ordering_dependency` 图，并基于该图门控并行扇出。这是依赖感知调度的最小机制，未来的智能体工程阶段会将其形式化。

5. 阅读 OpenAI 的并行函数调用章节和 Anthropic 的 `disable_parallel_tool_use` 文档。找出 Anthropic 建议禁用并行的一种真实工具类型。（提示：对同一资源的重大影响变更。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 并行工具调用（Parallel tool calls） | "一轮内扇出" | 模型在一条助手消息中输出多个工具调用 |
| `parallel_tool_calls` | "OpenAI 的标志" | 启用或禁用多调用输出 |
| `disable_parallel_tool_use` | "Anthropic 的反向标志" | 退出标志；默认为并行启用 |
| 工具调用 id（Tool call id） | "关联句柄" | 结果消息必须回显的每调用标识符 |
| 累加器（Accumulator） | "流缓冲区" | 用于部分 `arguments` 块的每 id 字符串缓冲区 |
| 乱序完成（Out-of-order completion） | "最快的先到" | 并行调用以不可预测的顺序完成；id 是粘合剂 |
| 依赖图（Dependency graph） | "顺序约束" | 输出作为其他工具输入的工具；不能并行 |
| 提前解析陷阱（Parse-early trap） | "JSON.parse 爆了" | 尝试解析不完整的 `arguments` 字符串 |
| `streamFunctionCallArguments` | "Gemini 3 特性" | 带唯一 id 的流式参数块 |
| 按完成顺序回复（Completion-order reply） | "不必等全部完成" | 结果到达就回复，以 id 为键 |

## 延伸阅读

- [OpenAI -- Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) -- 默认行为和退出标志
- [Anthropic -- Tool use: implementing tool use](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/implementing-tool-use) -- `disable_parallel_tool_use` 和结果批处理
- [Google -- Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) -- Gemini 3 起的 id 关联并行调用
- [OpenAI -- Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) -- OpenAI 流的块状参数重组
- [Anthropic -- Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) -- `content_block_delta` 与 `input_json_delta`
