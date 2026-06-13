# 毕业项目 01 — 终端原生编码智能体

> 到 2026 年，编码智能体的形态已经基本定型。一个 TUI 框架、一份有状态的计划、一个沙箱化的工具接口、以及一个循环执行"规划—执行—观察—恢复"的流程。Claude Code、Cursor 3 和 OpenCode 在 50 英尺外看起来几乎一模一样。本毕业项目要求你从头构建一个端到端的编码智能体——从命令行输入到输出 Pull Request——并在 SWE-bench Pro 上与 mini-swe-agent 和 Live-SWE-agent 进行对比评估。你将理解为什么难点不在于模型调用，而在于工具循环、沙箱以及 50 轮运行的成本控制。

**类型：** 毕业项目
**语言：** TypeScript / Bun（框架）、Python（评估脚本）
**前置要求：** 阶段 11（LLM 工程）、阶段 13（工具与协议）、阶段 14（智能体）、阶段 15（自主系统）、阶段 17（基础设施）
**涉及阶段：** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**所需时间：** 35 小时

## 问题所在

编码智能体在 2026 年成为主流 AI 应用类别。Claude Code（Anthropic）、集成了 Composer 2 和 Agent Tabs 的 Cursor 3（Cursor）、Amp（Sourcegraph）、OpenCode（112k star）、Factory Droids 和 Google Jules 都采用了相同的架构变体：一个终端框架、一个带权限控制的工具接口、一个沙箱，以及围绕前沿模型构建的规划—执行—观察循环。前沿的差距很窄——Live-SWE-agent 使用 Opus 4.5 在 SWE-bench Verified 上达到了 79.2%——但工程技艺的宽度很大。大多数失败模式并非模型犯错，而是工具循环不稳定、上下文污染、Token 消耗失控以及破坏性的文件系统操作。

你无法仅从外部来理解这些智能体。你必须亲手构建一个，看着它在第 47 轮因为 ripgrep 返回了 8MB 的匹配结果而崩溃，然后重新构建截断层。这就是本毕业项目的意义所在。

## 概念说明

框架有四个功能层。**规划（Plan）** 维护一个 TodoWrite 风格的状态对象，模型每轮重写它。**执行（Act）** 分发工具调用（读取、编辑、运行、搜索、Git）。**观察（Observe）** 捕获标准输出/标准错误/退出码，进行截断处理后将摘要反馈给模型。**恢复（Recover）** 处理工具错误，避免上下文窗口溢出或无限循环。2026 年的架构新增了一个要素：**钩子（Hooks）**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact`——这些可配置的扩展点让运维人员可以注入策略、遥测和防护措施。

沙箱使用 E2B 或 Daytona。每个任务在全新的 devcontainer 中运行，挂载一个可读写的 git worktree。框架永远不会触及宿主机的文件系统。无论成功或失败，worktree 都会被销毁。成本控制在三个层面强制执行：单轮 Token 上限、单次会话的美元预算以及硬性轮次限制（通常为 50 轮）。可观测性层使用 OpenTelemetry Span 并遵循 GenAI 语义规范，数据发送到自托管的 Langfuse。

## 架构

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技术栈

- 框架运行时：Bun 1.2 + Ink 5（React-in-terminal）
- 模型访问：OpenRouter 统一 API，支持 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5（用于最困难的任务）
- 工具传输：Model Context Protocol StreamableHTTP（MCP 2026 修订版）
- 沙箱：E2B sandboxes（JS SDK）或 Daytona devcontainers
- 代码搜索：ripgrep 子进程、tree-sitter 解析器（支持 17 种语言，预编译）
- 隔离：每个任务使用 `git worktree add`，成功/失败后清理
- 评估框架：SWE-bench Pro（verified 子集）+ Terminal-Bench 2.0 + 自建 30 任务保留集
- 可观测性：OpenTelemetry SDK + `gen_ai.*` 语义规范 → 自托管 Langfuse
- PR 发布：GitHub App + 细粒度 Token，范围限定在目标仓库

## 开始构建

1. **TUI 和命令循环。** 使用 Ink 搭建 Bun 项目脚手架。接受 `agent run <repo> "<task>"` 命令。打印分屏视图：计划面板（顶部）、工具调用流（中部）、Token 预算（底部）。添加 Ctrl-C 取消功能，在退出前触发 `SessionEnd` 钩子。

2. **计划状态。** 定义一个类型化的 TodoWrite schema（待办/进行中/已完成条目，附带备注）。模型每轮以工具调用形式重写完整状态——不要让它增量修改。将计划持久化到 `.agent/state.json`，以便崩溃后恢复。

3. **工具接口。** 定义六个工具：`read_file`、`edit_file`（带 diff 预览）、`ripgrep`、`tree_sitter_symbols`、`run_shell`（带超时）、`git`（status / diff / commit / push）。通过 MCP StreamableHTTP 暴露，使框架与传输层无关。每个工具返回截断后的输出（每次调用上限 4k Token）。

4. **沙箱封装。** 每个任务启动一个 E2B 沙箱。使用 `git worktree add -b agent/$TASK_ID` 创建新分支。所有工具调用在沙箱内执行。宿主机文件系统不可访问。

5. **钩子。** 实现全部八种 2026 年钩子类型。至少接入四个用户编写的钩子：(a) `PreToolUse` 破坏性命令守卫，阻止在 worktree 外执行 `rm -rf`；(b) `PostToolUse` Token 计量；(c) `SessionStart` 预算初始化；(d) `Stop` 写入最终追踪包。

6. **评估循环。** 克隆 SWE-bench Pro Python 的 30 个 issue 子集。针对每个运行你的框架。与 mini-swe-agent（最小基线）在 pass@1、每任务轮次和每任务成本上进行对比。将结果写入 `eval/results.jsonl`。

7. **成本控制。** 硬性截止：50 轮、200k 上下文、每任务 $5。`PreCompact` 钩子在 150k 处将较早的轮次总结为先前状态块，为新的观察腾出空间而不丢失计划。

8. **PR 发布。** 成功后，最后一步是 `git push` 加一个 GitHub API 调用，创建 PR 并在正文包含计划和 diff 摘要。

## 使用示例

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付成果

交付的技能文件位于 `outputs/skill-terminal-coding-agent.md`。给定一个仓库路径和任务描述，它在沙箱中运行完整的规划—执行—观察循环，并返回 PR URL 和追踪包。本毕业项目的评分标准：

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 与基线对比 | 你的框架与 mini-swe-agent 在 30 个匹配 Python 任务上的对比 |
| 20 | 架构清晰度 | 规划/执行/观察分离、钩子接口、工具 schema——参照 Live-SWE-agent 架构评审 |
| 20 | 安全性 | 沙箱逃逸测试、权限提示、破坏性命令守卫通过红队测试 |
| 20 | 可观测性 | 追踪完整性（100% 工具调用都有 Span）、每轮 Token 计量 |
| 15 | 开发者体验 | 冷启动 < 2s，崩溃恢复可继续计划，Ctrl-C 在工具执行中干净取消 |
| **100** | | |

## 练习

1. 将底层模型从 Claude Sonnet 4.7 切换到使用 vLLM 服务的 Qwen3-Coder-30B。对比 pass@1 和每任务成本。报告开源模型在哪些方面表现不足。

2. 添加一个 `reviewer` 子智能体，在 PR 发布前审查 diff，可以请求修改循环。衡量虚假的正面审查是否会将 SWE-bench 通过率降低到单智能体基线以下（提示：通常是的）。

3. 压力测试沙箱：编写一个尝试 `curl` 外部 URL 的任务和一个尝试在 worktree 外写入的任务。确认两者都被 PreToolUse 钩子拦截。记录尝试日志。

4. 使用更小的模型（Haiku 4.5）实现 `PreCompact` 摘要。衡量 3 倍压缩时计划保真度的损失。

5. 将 MCP StreamableHTTP 传输替换为 stdio。对冷启动和单次调用延迟进行基准测试。为纯本地使用场景选择最优方案。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 框架（Harness） | "智能体循环" | 围绕模型的代码，负责分发工具、维护计划状态和执行预算控制 |
| 钩子（Hook） | "智能体事件监听器" | 用户编写的脚本，在八个生命周期事件之一被框架触发执行 |
| Worktree | "Git 沙箱" | 在独立路径创建的链接 Git 检出；可丢弃而不影响主克隆 |
| TodoWrite | "计划状态" | 模型每轮重写的类型化待办/进行中/已完成条目列表 |
| StreamableHTTP | "MCP 传输" | 2026 年 MCP 修订版：带双向流的长连接 HTTP；替代 SSE |
| Token 上限 | "上下文预算" | 单轮或单次会话的输入+输出 Token 上限；触发压缩或终止 |
| pass@1 | "单次通过率" | SWE-bench 任务在首次运行时无需重试或偷看测试集即通过的比例 |

## 延伸阅读

- [Claude Code 文档](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的参考框架
- [Cursor 3 更新日志](https://cursor.com/changelog) — Agent Tabs 和 Composer 2 产品说明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench 框架对比的最小基线
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 使用 Opus 4.5 在 SWE-bench Verified 上达到 79.2%
- [OpenCode](https://opencode.ai) — 开源框架，112k star
- [SWE-bench Pro 排行榜](https://www.swebench.com) — 本毕业项目的目标评估平台
- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具调用和 Token 使用的 Span schema
