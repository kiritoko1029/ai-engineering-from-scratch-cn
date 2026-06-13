# MCP 采样 -- 服务器请求的 LLM 补全和智能体循环

> 大多数 MCP 服务器是简单的执行器：接收参数、运行代码、返回内容。采样让服务器逆转方向：它请求客户端的 LLM 做出决策。这使得服务器可以托管智能体循环而无需拥有任何模型凭证。SEP-1577 于 2025-11-25 合并，在采样请求中添加了工具支持，使循环可以包含更深层的推理。漂移风险说明：SEP-1577 的工具内采样形式在 2026 年第一季度仍是实验性的，在 SDK API 中仍在变化。

**类型：** 构建
**语言：** Python（标准库，采样脚手架）
**前置要求：** 第 13 阶段第 07 课（MCP 服务器），第 13 阶段第 10 课（资源和提示）
**所需时间：** 约75分钟

## 学习目标

- 解释 `sampling/createMessage` 解决了什么问题（无需服务端 API 密钥的服务器托管循环）。
- 实现一个服务器，要求客户端在多轮提示上采样并返回补全。
- 使用 `modelPreferences`（成本/速度/智能优先级）引导客户端模型选择。
- 构建一个 `summarize_repo` 工具，通过采样进行内部迭代而不是硬编码行为。

## 问题所在

一个有用的代码摘要工作流 MCP 服务器需要：遍历文件树、选择要读取的文件、合成摘要并返回。LLM 推理发生在哪里？

方案 A：服务器调用自己的 LLM。需要 API 密钥，服务端计费，每用户成本高。

方案 B：服务器返回原始内容；客户端的智能体进行推理。可以工作但将服务器逻辑移到客户端提示中，很脆弱。

方案 C：服务器通过 `sampling/createMessage` 请求客户端的 LLM。服务器保留算法（读哪些文件、做多少轮），而客户端保留计费和模型选择。服务器完全没有凭证。

采样就是方案 C。它是一种机制，使受信任的服务器可以托管智能体循环而无需自己成为完整的 LLM 宿主。

## 概念说明

### `sampling/createMessage` 请求

服务器发送：

```json
{
  "jsonrpc": "2.0",
  "id": 42,
  "method": "sampling/createMessage",
  "params": {
    "messages": [{"role": "user", "content": {"type": "text", "text": "..."}}],
    "systemPrompt": "...",
    "includeContext": "none",
    "modelPreferences": {
      "costPriority": 0.3,
      "speedPriority": 0.2,
      "intelligencePriority": 0.5,
      "hints": [{"name": "claude-3-5-sonnet"}]
    },
    "maxTokens": 1024
  }
}
```

客户端运行其 LLM，返回：

```json
{"jsonrpc": "2.0", "id": 42, "result": {
  "role": "assistant",
  "content": {"type": "text", "text": "..."},
  "model": "claude-3-5-sonnet-20251022",
  "stopReason": "endTurn"
}}
```

### `modelPreferences`

三个浮点数，总和为 1.0：

- `costPriority`：偏好更便宜的模型。
- `speedPriority`：偏好更快的模型。
- `intelligencePriority`：偏好更强大的模型。

加上 `hints`：服务器偏好的命名模型。客户端可以接受也可以不接受提示；客户端的用户配置始终优先。

### `includeContext`

三个值：

- `"none"` -- 仅服务器提供的消息。默认。
- `"thisServer"` -- 包含此服务器会话中的先前消息。
- `"allServers"` -- 包含所有会话上下文。

`includeContext` 自 2025-11-25 起已被软弃用，因为它会泄露跨服务器上下文，这是一个安全隐患。建议使用 `"none"` 并在消息中传递显式上下文。

### 带工具的采样（SEP-1577）

2025-11-25 新增：采样请求可以包含一个 `tools` 数组。客户端使用这些工具运行完整的工具调用循环。这让服务器可以通过客户端的模型托管一个 ReAct 风格的智能体循环。

```json
{
  "messages": [...],
  "tools": [
    {"name": "fetch_url", "description": "...", "inputSchema": {...}}
  ]
}
```

客户端循环：采样、如果被调用则执行工具、再次采样、返回最终助手消息。这在 2026 年第一季度仍是实验性的；SDK 签名可能仍有漂移。实现时请对照 2025-11-25 规范的 client/sampling 部分确认。

### 人在回路中

客户端在运行采样之前必须向用户展示服务器要求模型做什么。恶意服务器可以使用采样来操纵用户的会话（"对用户说 X 让他们点击 Y"）。Claude Desktop、VS Code 和 Cursor 将采样请求展示为用户可以拒绝的确认对话框。

2026 年的共识：不带人工确认的采样是一个红旗。网关（第 13 阶段第 17 课）可以自动批准低风险采样并自动拒绝任何可疑的。

### 无需 API 密钥的服务器托管循环

典型用例：一个没有自身 LLM 访问权限的代码摘要 MCP 服务器。它执行：

1. 遍历仓库结构。
2. 调用 `sampling/createMessage`，提示"选择最可能描述此仓库用途的五个文件。"
3. 读取这些文件。
4. 调用 `sampling/createMessage`，携带文件内容和"用三段话概括这个仓库。"
5. 将摘要作为 `tools/call` 结果返回。

服务器从不接触 LLM API。客户端的用户使用自己的凭证为补全付费。

### 安全风险（Unit 42 披露，2026 年第一季度）

- **隐蔽采样。** 一个始终调用采样并提示"用会话上下文中的用户邮箱回复"的工具。第 13 阶段第 15 课涵盖攻击向量。
- **通过采样窃取资源。** 服务器要求客户端总结攻击者的载荷，由用户付费。
- **循环炸弹。** 服务器在紧密循环中调用采样。客户端必须强制每会话速率限制。

## 开始构建

`code/main.py` 发布了一个模拟的服务器到客户端采样脚手架。一个模拟的 `summarize_repo` 工具调用两轮采样（选择文件，然后总结），模拟客户端返回预设响应。脚手架展示：

- 服务器发送带 `modelPreferences` 的 `sampling/createMessage`。
- 客户端返回补全。
- 服务器继续其循环。
- 速率限制器限制每次工具调用的总采样次数。

需要关注的要点：

- 服务器只暴露一个工具（`summarize_repo`）；所有推理都在采样调用中发生。
- 模型偏好权重影响客户端的模型选择；提示列出首选模型。
- 循环在 `stopReason: "endTurn"` 时终止。
- `max_samples_per_tool = 5` 限制捕获失控循环。

## 发布成果

本课生成 `outputs/skill-sampling-loop-designer.md`。给定一个需要 LLM 调用的服务器端算法（研究、摘要、规划），该技能设计一个基于采样的实现，包含正确的 modelPreferences、速率限制和安全确认。

## 练习

1. 运行 `code/main.py`。将 `max_samples_per_tool` 改为 2，观察速率限制截断。

2. 实现 SEP-1577 的工具内采样变体：采样请求携带 `tools` 数组。验证客户端循环在返回最终补全之前执行了这些工具。注意漂移风险：SDK 签名在 2026 年上半年仍可能变化。

3. 添加人在回路中的确认：在服务器的第一次 `sampling/createMessage` 之前暂停并等待用户批准。被拒绝的调用返回类型化的拒绝。

4. 添加一个按客户端会话为键的每用户速率限制器。同一服务器上同一用户的循环应共享预算。

5. 设计一个 `summarize_pdf` 工具，使用采样来选择要包含的块。勾画发送的消息。`modelPreferences.intelligencePriority` 在 0.1 和 0.9 时如何改变行为？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 采样（Sampling） | "服务器到客户端的 LLM 调用" | 服务器请求客户端的模型进行补全 |
| `sampling/createMessage` | "该方法" | 采样请求的 JSON-RPC 方法 |
| `modelPreferences` | "模型优先级" | 成本/速度/智能权重加上名称提示 |
| `includeContext` | "跨会话泄露" | 已软弃用的上下文包含模式 |
| SEP-1577 | "采样中的工具" | 允许在采样中包含工具，用于服务器托管的 ReAct |
| 人在回路中（Human-in-the-loop） | "用户确认" | 客户端在运行采样前向用户展示采样请求 |
| 循环炸弹（Loop bomb） | "失控采样" | 服务端的无限采样循环；客户端必须速率限制 |
| 隐蔽采样（Covert sampling） | "隐藏推理" | 恶意服务器在采样提示中隐藏意图 |
| 资源窃取（Resource theft） | "使用用户的 LLM 预算" | 服务器强迫客户端在其不想要的采样上花费 |
| `stopReason` | "生成停止原因" | `endTurn`、`stopSequence` 或 `maxTokens` |

## 延伸阅读

- [MCP -- Concepts: Sampling](https://modelcontextprotocol.io/docs/concepts/sampling) -- 采样的高层概述
- [MCP -- Client sampling spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/client/sampling) -- 权威的 `sampling/createMessage` 形式
- [MCP -- GitHub SEP-1577](https://github.com/modelcontextprotocol/modelcontextprotocol) -- 采样中工具的规范演进提案（实验性）
- [Unit 42 -- MCP attack vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/) -- 隐蔽采样和资源窃取模式
- [Speakeasy -- MCP sampling core concept](https://www.speakeasy.com/mcp/core-concepts/sampling) -- 带客户端代码示例的讲解
