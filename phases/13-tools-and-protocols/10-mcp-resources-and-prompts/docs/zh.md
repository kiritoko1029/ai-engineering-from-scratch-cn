# MCP 资源和提示 -- 工具之外的上下文暴露

> 工具占据了 MCP 90% 的关注度。另外两个服务器端原语解决的是不同的问题。资源暴露数据供读取；提示将可复用模板暴露为斜杠命令。许多服务器应该使用资源而不是将读取包装在工具中，使用提示而不是在客户端提示中硬编码工作流。本课命名决策规则，并走完 `resources/*` 和 `prompts/*` 消息。

**类型：** 构建
**语言：** Python（标准库，资源 + 提示处理器）
**前置要求：** 第 13 阶段第 07 课（MCP 服务器）
**所需时间：** 约45分钟

## 学习目标

- 为给定领域决定将能力暴露为工具、资源还是提示。
- 实现 `resources/list`、`resources/read`、`resources/subscribe`，并处理 `notifications/resources/updated`。
- 实现 `prompts/list` 和 `prompts/get`，带参数模板。
- 识别宿主何时将提示作为斜杠命令展示 vs 自动注入上下文。

## 问题所在

一个朴素的笔记应用 MCP 服务器将所有东西都暴露为工具：`notes_read`、`notes_list`、`notes_search`。这将每次数据访问都包装在模型驱动的工具调用中。后果：

- 模型必须为每个可能受益于上下文的查询决定是否调用 `notes_read`。
- 只读内容无法被订阅或流式传输到宿主的侧边面板。
- 客户端 UI（Claude Desktop 的资源附件面板、Cursor 的"Include file"选择器）无法展示数据。

正确的划分：将数据暴露为资源，将变更或计算动作暴露为工具，将可复用的多步工作流暴露为提示。每个原语都有其 UX 特性和访问模式。

## 概念说明

### 工具 vs 资源 vs 提示 -- 决策规则

| 能力 | 原语 |
|------|------|
| 用户想搜索、过滤或转换数据 | 工具 |
| 用户想让宿主将此数据作为上下文包含 | 资源 |
| 用户想要一个可重复运行的模板化工作流 | 提示 |

指导原则：如果模型在每次相关查询中调用它都会受益，它就是工具。如果用户将它附加到对话中会受益，它就是资源。如果整个多步工作流是用户想要复用的单元，它就是提示。

### 资源

`resources/list` 返回 `{resources: [{uri, name, mimeType, description?}]}`。`resources/read` 接受 `{uri}` 并返回 `{contents: [{uri, mimeType, text | blob}]}`。

URI 可以是任何可寻址的：

- `file:///Users/alice/notes/mcp.md`
- `postgres://my-db/query/SELECT ...`
- `notes://note-14`（自定义方案）
- `memory://session-2026-04-22/recent`（服务器特定）

`contents[]` 支持文本和二进制。二进制使用 `blob` 作为 base64 编码字符串加上 `mimeType`。

### 资源订阅

在能力中声明 `{resources: {subscribe: true}}`。客户端调用 `resources/subscribe {uri}`。当资源变更时，服务器发送 `notifications/resources/updated {uri}`。客户端重新读取。

用例：一个笔记服务器的资源是磁盘上的文件；文件监视器触发更新通知；Claude Desktop 在宿主外部编辑时重新将文件拉入上下文。

### 资源模板（2025-11-25 新增）

`resourceTemplates` 让你暴露参数化的 URI 模式：`notes://{id}`，其中 `id` 作为补全目标。客户端可以在资源选择器中自动补全 id。

### 提示

`prompts/list` 返回 `{prompts: [{name, description, arguments?}]}`。`prompts/get` 接受 `{name, arguments}` 并返回 `{description, messages: [{role, content}]}`。

提示是一个模板，填充为宿主馈送给其模型的消息列表。例如，一个 `code_review` 提示接受一个 `file_path` 参数，返回一个三消息序列：系统消息、包含文件内容的用户消息和带推理模板的助手启动消息。

### 宿主和提示

Claude Desktop、VS Code 和 Cursor 将提示作为斜杠命令展示在聊天 UI 中。用户输入 `/code_review` 并从表单中选择参数。服务器的提示是"用户快捷方式"与"发送给模型的完整提示"之间的契约。

并非所有客户端都支持提示 -- 检查能力协商。声明了提示能力但客户端不支持提示的服务器，其斜杠命令将不会被看到。

### "列表变更"通知

资源和提示在集合变更时都会发出 `notifications/list_changed`。一个刚刚导入了 20 条新笔记的笔记服务器发出 `notifications/resources/list_changed`；客户端重新调用 `resources/list` 以获取新增内容。

### 内容类型约定

文本：`mimeType: "text/plain"`、`text/markdown`、`application/json`。
二进制：`image/png`、`application/pdf`，加上 `blob` 字段。
MCP Apps（第 14 课）：`ui://` URI 中的 `text/html;profile=mcp-app`。

### 动态资源

资源 URI 不必对应静态文件。`notes://recent` 可以在每次读取时返回最新的五条笔记。`db://query/users/active` 可以执行参数化查询。服务器可以自由地动态计算内容。

规则：如果客户端可以按 URI 缓存，URI 必须是稳定的。如果计算是一次性的，URI 应该包含时间戳或 nonce，以免客户端缓存过期。

### 订阅 vs 轮询

支持订阅的客户端通过 `notifications/resources/updated` 获得服务器推送。不支持订阅的客户端或宿主通过重新读取来轮询。两者都符合规范。服务器的能力声明告诉客户端它支持哪种。

订阅的成本：服务器上的每会话状态（谁订阅了什么）。保持订阅集有界；断开的客户端应超时。

### 提示 vs 系统提示

MCP 中的提示不是系统提示。宿主的系统提示（其自身的操作指令）和 MCP 提示（服务器提供的、由用户调用的模板）并存。一个行为良好的客户端永远不会让服务器提示覆盖其自身的系统提示；它们是分层的。

## 开始构建

`code/main.py` 在第 07 课笔记服务器的基础上扩展了：

- 每笔记资源（`notes://note-1` 等），支持 `resources/subscribe`。
- 一个 `review_note` 提示，渲染为三消息模板。
- 文件监视器模拟，在笔记被修改时发出 `notifications/resources/updated`。
- 一个 `notes://recent` 动态资源，始终返回最新的五条笔记。

运行演示查看完整流程。

## 发布成果

本课生成 `outputs/skill-primitive-splitter.md`。给定一个拟议的 MCP 服务器，该技能将每个能力分类为工具/资源/提示，并给出理由。

## 练习

1. 运行 `code/main.py`。观察初始资源列表，然后触发一次笔记编辑，验证 `notifications/resources/updated` 事件被触发。

2. 添加一个 `resources/list_changed` 发出器：当创建新笔记时，发送通知以便客户端重新发现。

3. 为一个 GitHub MCP 服务器设计三个提示：`summarize_pr`、`triage_issue`、`release_notes`。每个都带参数模式。提示体应该无需进一步编辑即可运行。

4. 取第 07 课服务器中的一个现有工具，分类它应该保持为工具还是拆分为资源加工具对。用一句话证明。

5. 阅读规范中 `server/resources` 和 `server/prompts` 部分。找出 `resources/read` 中一个很少填充但规范支持的字段。提示：查看资源内容上的 `_meta`。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 资源（Resource） | "暴露的数据" | 宿主可读取的 URI 可寻址内容 |
| 资源 URI（Resource URI） | "数据指针" | 方案前缀的标识符（`file://`、`notes://` 等） |
| `resources/subscribe` | "监视变更" | 客户端可选的服务器推送更新，针对特定 URI |
| `notifications/resources/updated` | "资源已变更" | 信号客户端订阅的资源有新内容 |
| 资源模板（Resource template） | "参数化 URI" | 带补全提示的 URI 模式，用于宿主选择器 |
| 提示（Prompt） | "斜杠命令模板" | 带参数槽的命名多消息模板 |
| 提示参数（Prompt arguments） | "模板输入" | 宿主在渲染前收集的类型化参数 |
| `prompts/get` | "渲染模板" | 服务器返回填充后的消息列表 |
| 内容块（Content block） | "类型化块" | `{type: text \| image \| resource \| ui_resource}` |
| 斜杠命令 UX（Slash-command UX） | "用户快捷方式" | 宿主将提示展示为以 `/` 开头的命令 |

## 延伸阅读

- [MCP -- Concepts: Resources](https://modelcontextprotocol.io/docs/concepts/resources) -- 资源 URI、订阅和模板
- [MCP -- Concepts: Prompts](https://modelcontextprotocol.io/docs/concepts/prompts) -- 提示模板和斜杠命令集成
- [MCP -- Server resources spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/resources) -- 完整的 `resources/*` 消息参考
- [MCP -- Server prompts spec 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25/server/prompts) -- 完整的 `prompts/*` 消息参考
- [MCP -- Protocol info site: resources](https://modelcontextprotocol.info/docs/concepts/resources/) -- 扩展官方文档的社区指南
