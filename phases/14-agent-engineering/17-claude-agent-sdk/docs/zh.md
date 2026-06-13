# Claude Agent SDK：子智能体与会话存储

> Claude Agent SDK 是 Claude Code 工具链的库形式。内置工具、用于上下文隔离的子智能体、钩子、W3C 跟踪传播、会话存储对等。Claude Managed Agents 是用于长时间异步工作的托管替代方案。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 10（技能库）
**所需时间：** 约75分钟

## 学习目标

- 解释 Anthropic Client SDK（原始 API）和 Claude Agent SDK（工具链形状）的区别。
- 描述子智能体——并行化和上下文隔离——以及何时使用它们。
- 说出 Python SDK 的会话存储接口（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）和 `--session-mirror` 的角色。
- 用标准库实现一个带内置工具、带隔离上下文的子智能体生成、生命周期钩子和会话存储的工具链。

## 问题所在

原始 LLM API 给你一次往返。生产智能体需要工具执行、MCP 服务器、生命周期钩子、子智能体生成、会话持久化、跟踪传播。Claude Agent SDK 将此形状作为库提供——Claude Code 使用的同一个工具链，暴露给自定义智能体。

## 概念说明

### Client SDK vs Agent SDK

- **Client SDK（`anthropic`）。** 原始 Messages API。你拥有循环、工具、状态。
- **Agent SDK（`claude-agent-sdk`）。** 内置工具执行、MCP 连接、钩子、子智能体生成、会话存储。Claude Code 循环作为库。

### 内置工具

SDK 开箱提供 10+ 工具：文件读写、shell、grep、glob、网页抓取等。自定义工具通过标准工具 schema 接口注册。

### 子智能体

Anthropic 记录的两个用途：

1. **并行化。** 并发运行独立工作。"为这 20 个模块中的每一个找到测试文件"是 20 个并行子智能体任务。
2. **上下文隔离。** 子智能体使用自己的上下文窗口；只有结果返回给编排器。编排器的预算得到保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()` 用于读取子智能体记录。

### 会话存储

与 TypeScript 协议对等：

- `append(session_id, message)`——添加一轮。
- `load(session_id)`——恢复对话。
- `list_sessions()`——枚举。
- `delete(session_id)`——级联到子智能体会话。
- `list_subkeys(session_id)`——列出子智能体键。

`--session-mirror`（CLI 标志）在流式传输时将记录镜像到外部文件，用于调试。

### 钩子

可注册的生命周期钩子：

- `PreToolUse`、`PostToolUse`——门控或审计工具调用。
- `SessionStart`、`SessionEnd`——设置和拆除。
- `UserPromptSubmit`——在模型看到之前对用户输入执行操作。
- `PreCompact`——在上下文压缩前运行。
- `Stop`——智能体退出时清理。
- `Notification`——侧通道告警。

钩子是 pro-workflow（第 14 阶段课程参考）和类似系统添加横切行为的方式。

### W3C 跟踪上下文

调用者上活跃的 OTel span 通过 W3C 跟踪上下文头传播到 CLI 子进程。整个多进程跟踪在你的后端中显示为一条跟踪。

### Claude Managed Agents

托管替代方案（beta 头 `managed-agents-2026-04-01`）。长时间异步工作、内置提示缓存、内置压缩。用控制换取托管基础设施。

### 此模式出错的地方

- **子智能体过度生成。** 为 100 个小任务生成 100 个子智能体。开销占主导。改为批量处理。
- **钩子蔓延。** 每个团队添加钩子；启动时间膨胀。季度审查钩子。
- **会话膨胀。** 会话不断积累；大小增长。使用 `list_sessions` + 过期策略。

## 开始构建

`code/main.py` 用标准库实现了 SDK 形状：

- `Tool`、`ToolRegistry`，带内置 `read_file`、`write_file`、`list_dir`。
- `Subagent`——私有上下文、隔离运行、结果返回。
- `SessionStore`——append、load、list、delete、list_subkeys。
- `Hooks`——`pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 一个演示：主智能体并行生成 3 个子智能体（每个隔离），聚合结果，持久化会话。

运行它：

```
python3 code/main.py
```

轨迹展示了子智能体上下文隔离（编排器上下文大小保持有界）、钩子执行和会话持久化。

## 使用它

- **Claude Agent SDK** 用于想要 Claude Code 工具链形状的 Claude 优先产品。
- **Claude Managed Agents** 用于托管的长时间异步工作。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先的对应物。
- **LangGraph + 自定义工具**如果你想要图形状的状态机。

## 交付它

`outputs/skill-claude-agent-scaffold.md` 脚手架一个 Claude Agent SDK 应用，带子智能体、钩子、会话存储、MCP 服务器附加和 W3C 跟踪传播。

## 练习

1. 添加一个子智能体生成器，将 20 个任务批量分为每组 5 个并行子智能体。衡量编排器上下文大小与每任务一个的对比。
2. 实现一个 `PreToolUse` 钩子，对 `write_file` 调用限速（每会话每分钟 5 次）。追踪行为。
3. 将 `list_subkeys` 接入以渲染子智能体树。深度嵌套是什么样的？
4. 将简易实现移植到真实的 `claude-agent-sdk` Python 包。工具注册有什么变化？
5. 阅读 Claude Managed Agents 文档。何时从自托管切换到托管？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Agent SDK | "Claude Code 作为库" | 工具链形状：工具、MCP、钩子、子智能体、会话存储 |
| 子智能体（Subagent） | "子智能体" | 独立上下文、自己的预算；结果向上冒泡 |
| 会话存储（Session store） | "对话数据库" | 持久化、加载、列出、删除轮次，带子智能体级联 |
| 钩子（Hook） | "生命周期回调" | 工具前/后、会话、提示提交、压缩、停止 |
| W3C 跟踪上下文（W3C trace context） | "跨进程跟踪" | 父 span 传播到 CLI 子进程 |
| Managed Agents | "托管工具链" | Anthropic 托管的长时间异步工作 |
| `--session-mirror` | "记录镜像" | 在流式传输时将会话轮次写入外部文件 |
| MCP 服务器（MCP server） | "工具接口" | 附加到智能体的外部工具/资源来源 |

## 延伸阅读

- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview)——Claude Code 的库形式
- [Anthropic，Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)——生产模式
- [Claude Managed Agents 概览](https://platform.claude.com/docs/en/managed-agents/overview)——托管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)——对应物
