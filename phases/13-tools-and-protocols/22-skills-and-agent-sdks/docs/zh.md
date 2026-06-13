# 技能与智能体 SDK —— Anthropic Skills、AGENTS.md、OpenAI Apps SDK

> MCP 说明"有哪些工具"。技能说明"如何执行任务"。2026 年的栈将两者叠加。Anthropic 的 Agent Skills（开放标准，2025 年 12 月）以 SKILL.md 形式发布，支持渐进式披露。OpenAI 的 Apps SDK 是 MCP 加小组件元数据。AGENTS.md（现已在 60,000 多个仓库中）位于仓库根目录，作为项目级智能体上下文。本课说明各层覆盖内容，并构建一个可在智能体间迁移的最小 SKILL.md + AGENTS.md 包。

**类型：** 学习
**语言：** Python（标准库，SKILL.md 解析器和加载器）
**前置要求：** Phase 13 · 07（MCP 服务器）
**所需时间：** 约45分钟

## 学习目标

- 区分三层：AGENTS.md（项目上下文）、SKILL.md（可复用知识）、MCP（工具）。
- 编写带 YAML 前置元数据和渐进式披露的 SKILL.md。
- 以文件系统风格将技能加载到智能体运行时。
- 将技能与 MCP 服务器和 AGENTS.md 组合，使一个包能在 Claude Code、Cursor 和 Codex 中工作。

## 问题所在

一位工程师将发布说明编写工作流提炼为多步提示："读取最新的已合并 PR。按领域分组。各自摘要。按团队风格编写变更日志条目。发布到 Slack 草稿。"他们将其放在 Notion 文档中给团队使用。

现在他们想从 Claude Code、Cursor 和 Codex CLI 使用这个工作流。每个智能体有不同的加载指令方式：Claude Code 斜杠命令、Cursor 规则、Codex `.codex.md`。工程师复制了三次工作流并维护三个副本。

AGENTS.md 和 SKILL.md 一起解决了这个问题：

- **AGENTS.md** 位于仓库根目录。每个兼容的智能体在会话开始时读取它。"这个项目如何工作？约定是什么？哪些命令运行测试？"
- **SKILL.md** 是可移植的包：YAML 前置元数据（名称、描述）+ markdown 正文 + 可选资源。支持技能的智能体按名称按需加载。
- **MCP**（Phase 13 · 06-14）处理技能需要调用的工具。

三层，一个可移植制品。

## 概念说明

### AGENTS.md（agents.md）

2025 年末推出，截至 2026 年 4 月已被 60,000 多个仓库采用。仓库根目录一个文件。格式：

```markdown
# Project: my-service

## Conventions
- TypeScript with strict mode.
- Use Pydantic for models on the Python side.
- Tests run with `pnpm test`.

## Build and run
- `pnpm dev` for local dev server.
- `pnpm build` for production bundle.
```

智能体在会话开始时读取此文件，并用它来校准该项目的行为。2026 年每个编码智能体都支持 AGENTS.md：Claude Code、Cursor、Codex、Copilot Workspace、opencode、Windsurf、Zed。

### SKILL.md 格式

Anthropic 的 Agent Skills（2025 年 12 月作为开放标准发布）：

```markdown
---
name: release-notes-writer
description: Write a changelog entry for the latest merged PRs following this project's style.
---

# Release notes writer

When invoked, run these steps:

1. List PRs merged since the last tag. Use `gh pr list --base main --state merged`.
2. Group by label: feature, fix, chore, docs.
3. For each PR in each group, write one line: `- <title> (#<num>)`.
4. Draft the release notes and stage them in CHANGELOG.md.

If the user says "ship", run `git tag vX.Y.Z` and `gh release create`.

## Notes

- Never include commits without a PR.
- Skip "chore" entries from the public changelog.
```

前置元数据声明技能的身份。正文是技能加载时呈现给模型的提示。

### 渐进式披露

技能可以引用智能体仅在需要时获取的子资源。示例：

```
skills/
  release-notes-writer/
    SKILL.md
    style-guide.md
    template.md
    scripts/
      generate.sh
```

SKILL.md 说"参见 style-guide.md 了解风格规则。"智能体仅在技能活跃运行时拉取 style-guide.md。这避免了用模型可能不需要的细节膨胀提示。

### 文件系统发现

智能体运行时扫描已知目录查找 SKILL.md 文件：

- `~/.anthropic/skills/*/SKILL.md`
- 项目 `./skills/*/SKILL.md`
- `~/.claude/skills/*/SKILL.md`

按文件夹名和前置元数据 `name` 加载。Claude Code、Anthropic Claude Agent SDK 和 SkillKit（跨智能体）都遵循此模式。

### Anthropic Claude Agent SDK

`@anthropic-ai/claude-agent-sdk`（TypeScript）和 `claude-agent-sdk`（Python）在会话开始时加载技能，将它们作为可调用的"智能体"暴露在运行时中。智能体循环在用户调用时调度到技能。

### OpenAI Apps SDK

2025 年 10 月推出；直接基于 MCP 构建。将 OpenAI 之前的 Connectors 和 Custom GPT Actions 统一到单一开发者表面。一个 Apps SDK 应用是：

- 一个 MCP 服务器（工具、资源、提示）。
- 加上 ChatGPT UI 的小组件元数据。
- 加上可选的 MCP Apps `ui://` 资源用于交互式表面。

相同协议，更丰富的用户体验。

### 通过 SkillKit 实现跨智能体可移植性

SkillKit 和类似的跨智能体分发层将单个 SKILL.md 翻译为 32 多个 AI 智能体（Claude Code、Cursor、Codex、Gemini CLI、OpenCode 等）的原生格式。一个真实来源；多个消费者。

### 三层栈

| 层 | 文件 | 加载时机 | 用途 |
|----|------|---------|------|
| AGENTS.md | 仓库根目录 | 会话开始 | 项目级约定 |
| SKILL.md | 技能目录 | 技能被调用 | 可复用工作流 |
| MCP 服务器 | 外部进程 | 需要工具时 | 可调用操作 |

三层组合：智能体在会话开始时读取 AGENTS.md，用户调用技能，技能的指令包含 MCP 工具调用，智能体通过 MCP 客户端调度。

## 开始构建

`code/main.py` 提供了一个标准库 SKILL.md 解析器和加载器。它发现 `./skills/` 下的技能，解析 YAML 前置元数据和 markdown 正文，生成以技能名为键的字典。然后模拟一个按名称调用 `release-notes-writer` 的智能体循环。

需要关注的要点：

- YAML 前置元数据使用最小标准库解析器解析（无 `pyyaml` 依赖）。
- 技能正文按原样存储；调用时智能体将其前置到系统提示。
- 通过 `read_subresource` 函数演示渐进式披露，按需拉取引用的文件。

## 交付产出

本课生成 `outputs/skill-agent-bundle.md`。给定一个工作流，该技能生成组合的 SKILL.md + AGENTS.md + MCP 服务器蓝图包，可在智能体间移植。

## 练习

1. 运行 `code/main.py`。在 `skills/` 下添加第二个技能并确认加载器能识别。

2. 为本课程仓库编写 AGENTS.md。包含测试命令、风格约定和 Phase 13 的心智模型。

3. 将团队内部文档中的多步工作流移植到 SKILL.md。验证它能在 Claude Code 中加载。

4. 手动将技能翻译为 Cursor 和 Codex 的原生规则格式。计算格式之间的差异 —— 这是 SkillKit 自动化的翻译面。

5. 阅读 Anthropic Agent Skills 博客文章。找出 Claude Agent SDK 中本课加载器未涵盖的一个功能。（提示：智能体子调用。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| SKILL.md | "技能文件" | YAML 前置元数据加 markdown 正文，由智能体运行时加载 |
| AGENTS.md | "仓库根智能体上下文" | 会话开始时读取的项目级约定文件 |
| 渐进式披露 | "延迟加载子资源" | 技能正文引用仅在需要时拉取的文件 |
| 前置元数据 | "顶部 YAML 块" | `---` 分隔符中的元数据（名称、描述） |
| Claude Agent SDK | "Anthropic 的技能运行时" | `@anthropic-ai/claude-agent-sdk`，加载技能和路由 |
| OpenAI Apps SDK | "MCP 加小组件元数据" | OpenAI 基于 MCP 加 ChatGPT UI 钩子构建的开发者表面 |
| 技能发现 | "文件系统扫描" | 遍历已知目录查找 SKILL.md，按名称索引 |
| 跨智能体可移植性 | "一个技能多个智能体" | 通过 SkillKit 类工具将一个 SKILL.md 翻译到 32 多个智能体 |
| Agent Skill | "可移植知识" | MCP 工具概念之外的可复用任务模板 |
| Apps SDK | "MCP 加 ChatGPT UI" | 在 MCP 上统一的 Connectors 和 Custom GPTs |

## 延伸阅读

- [Anthropic — Agent Skills announcement](https://www.anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills) — 2025 年 12 月发布
- [Anthropic — Agent Skills docs](https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview) — SKILL.md 格式参考
- [OpenAI — Apps SDK](https://developers.openai.com/apps-sdk) — 基于 MCP 的 ChatGPT 开发者平台
- [agents.md](https://agents.md/) — AGENTS.md 格式和采用列表
- [Anthropic — anthropics/skills GitHub](https://github.com/anthropics/skills) — 官方技能示例
