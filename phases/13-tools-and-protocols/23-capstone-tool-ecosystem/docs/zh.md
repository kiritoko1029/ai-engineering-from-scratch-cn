# 综合项目 —— 构建完整的工具生态系统

> Phase 13 教授了每个部分。本综合项目将它们串联为一个生产级系统：一个带工具 + 资源 + 提示 + 任务 + UI 的 MCP 服务器，边缘的 OAuth 2.1，RBAC 网关，多服务器客户端，A2A 子智能体调用，到收集器的 OTel 追踪，CI 中的工具投毒检测，以及 AGENTS.md + SKILL.md 包。完成后你可以为每个架构选择辩护。

**类型：** 构建
**语言：** Python（标准库，端到端生态系统实验框架）
**前置要求：** Phase 13 · 01 到 21
**所需时间：** 约120分钟

## 学习目标

- 组合一个暴露工具、资源、提示和带 `ui://` 应用的任务的 MCP 服务器。
- 在服务器前部署一个强制 RBAC 和固定哈希的 OAuth 2.1 网关。
- 编写一个使用 OTel GenAI 属性端到端追踪的多服务器客户端。
- 将部分工作负载委托给 A2A 子智能体；验证不透明性得以保持。
- 使用 AGENTS.md + SKILL.md 打包整个栈，使其他智能体可以驱动它。

## 问题所在

交付"研究与报告"系统：

- 用户问："总结 2026 年关于智能体协议被引用最多的三篇 arXiv 论文。"
- 系统：通过 MCP 搜索 arXiv；通过 A2A 将论文摘要委托给专业的写作智能体；聚合结果；将交互式报告渲染为 MCP Apps `ui://` 资源；将每一步记录到 OTel。

Phase 13 的所有原语都出现了。这不是玩具 —— 2026 年 Anthropic（Claude Research 产品）、OpenAI（带 Apps SDK 的 GPTs）和第三方交付的生产研究助手系统正是这种形态。

## 概念说明

### 架构

```
[user] -> [client] -> [gateway (OAuth 2.1 + RBAC)] -> [research MCP server]
                                                      |
                                                      +- MCP tool: arxiv_search (pure)
                                                      +- MCP resource: notes://recent
                                                      +- MCP prompt: /research_topic
                                                      +- MCP task: generate_report (long)
                                                      +- MCP Apps UI: ui://report/current
                                                      +- A2A call: writer-agent (tasks/send)
                                                      |
                                                      +- OTel GenAI spans
```

### 追踪层次结构

```
agent.invoke_agent
 ├── llm.chat (kick off)
 ├── mcp.call -> tools/call arxiv_search
 ├── mcp.call -> resources/read notes://recent
 ├── mcp.call -> prompts/get research_topic
 ├── a2a.tasks/send -> writer-agent
 │    └── task transitions (opaque internals)
 ├── mcp.call -> tools/call generate_report (task-augmented)
 │    └── tasks/status polling
 │    └── tasks/result (completed, returns ui:// resource)
 └── llm.chat (final synthesis)
```

一个 trace id。每个 span 都有正确的 `gen_ai.*` 属性。

### 安全态势

- OAuth 2.1 + PKCE，资源指示器将受众固定到网关。
- 网关持有上游凭据；用户永远看不到。
- RBAC：`alice` 有 `research:read`、`research:write`，可以调用所有工具。`bob` 有 `research:read`，不能调用 `generate_report`。
- 固定描述清单：丢弃任何工具哈希变更的服务器。
- 二法则审计：没有工具组合不可信输入、敏感数据和重大操作。

### 渲染

最终的 `generate_report` 任务返回内容块加一个 `ui://report/current` 资源。客户端的宿主（Claude Desktop 等）在沙箱 iframe 中渲染交互式仪表板。仪表板包含排序的论文列表、引用次数，以及一个用户点击任意论文时调用 `host.callTool('summarize_paper', {arxiv_id})` 的按钮。

### 打包

整个系统以如下形式交付：

```
research-system/
  AGENTS.md                     # project conventions
  skills/
    run-research/
      SKILL.md                  # the top-level workflow
  servers/
    research-mcp/               # the MCP server
      pyproject.toml
      src/
  agents/
    writer/                     # the A2A agent
  gateway/
    config.yaml                 # RBAC + pinned manifest
```

用户使用 `docker compose up` 部署。Claude Code、Cursor、Codex 和 opencode 用户可以通过调用 `run-research` 技能来驱动系统。

### 每个 Phase 13 课程的贡献

| 课程 | 综合项目使用的部分 |
|------|-------------------|
| 01-05 | 工具接口、提供商可移植性、并行调用、模式、检查 |
| 06-10 | MCP 原语、服务器、客户端、传输层、资源 + 提示 |
| 11-14 | 采样、根 + 引出、异步任务、`ui://` 应用 |
| 15-17 | 工具投毒、OAuth 2.1、网关 + 注册表 |
| 18 | A2A 子智能体委托 |
| 19 | OTel GenAI 追踪 |
| 20 | LLM 层的路由网关 |
| 21 | SKILL.md + AGENTS.md 打包 |

## 开始构建

`code/main.py` 将前面课程的模式串联为一个可运行的演示。全部使用标准库，全部在进程中，以便你可以端到端阅读。它运行研究与报告场景的完整流程：与网关握手、模拟 OAuth 2.1、合并 tools/list、将 generate_report 作为任务、A2A 调用写作智能体、返回 ui:// 资源、发射 OTel span。

需要关注的要点：

- 所有跳转共享一个 trace id。
- 网关策略阻止第二个用户写入。
- Task 生命周期经历 working → completed，返回文本和 ui:// 内容。
- A2A 调用的内部状态对编排器不透明。
- AGENTS.md 和 SKILL.md 是另一个智能体复现工作流所需的唯一文件。

## 交付产出

本课生成 `outputs/skill-ecosystem-blueprint.md`。给定一个产品需求（研究、摘要、自动化），该技能生成完整架构：哪些 MCP 原语、哪些网关控制、哪些 A2A 调用、哪些遥测、哪些打包。

## 练习

1. 运行 `code/main.py`。注意单一 trace id 以及 span 如何嵌套。计算演示涉及了多少个 Phase 13 的原语。

2. 扩展演示：添加第二个后端 MCP 服务器（如 `bibliography`），确认网关将其工具合并到同一命名空间。

3. 用运行在子进程中的真实 A2A 写作智能体替换假的。使用第 19 课的实验框架。

4. 在编排器和 LLM 之间的路由网关中添加 PII 脱敏步骤。确认用户查询中的电子邮件被清除。

5. 为将维护此系统的队友编写 AGENTS.md。阅读时间应不超过五分钟，并给他们驱动综合项目在 Cursor 或 Codex 中所需的一切。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| 综合项目 | "Phase 13 集成演示" | 使用每个原语的端到端系统 |
| 研究与报告 | "场景" | 搜索、摘要、渲染模式 |
| 生态系统 | "所有部分在一起" | 服务器 + 客户端 + 网关 + 子智能体 + 遥测 + 打包 |
| 追踪层次结构 | "单一 trace id" | 每跳的 span 共享 trace；父子通过 span id |
| 网关颁发的令牌 | "传递性认证" | 客户端只看到网关的令牌；网关持有上游凭据 |
| 合并命名空间 | "所有工具在一个扁平列表" | 网关处的多服务器合并，冲突时加前缀 |
| 不透明边界 | "A2A 调用隐藏内部" | 子智能体的推理对编排器不可见 |
| 三层栈 | "AGENTS.md + SKILL.md + MCP" | 项目上下文 + 工作流 + 工具 |
| 纵深防御 | "多安全层" | 固定哈希、OAuth、RBAC、二法则、审计日志 |
| 规范合规矩阵 | "我们交付规范要求的内容" | 将交付物映射到 2025-11-25 要求的清单 |

## 延伸阅读

- [MCP — Specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — 合并参考
- [MCP blog — 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — 协议的发展方向
- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A v1.0 参考
- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 规范追踪约定
- [Anthropic — Claude Agent SDK overview](https://code.claude.com/docs/en/agent-sdk/overview) — 生产智能体运行时模式
