# 毕业项目 16 — GitHub Issue 到 PR 自主智能体

> AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都在 2026 年推出了相同的产品形态：标记一个 issue，获得一个 PR。在云沙箱中运行智能体，验证测试通过，发布带有推理说明的可评审 PR。难点在于自动复现仓库的构建环境、防止凭证泄露、强制执行每个仓库的预算，以及确保智能体无法强制推送。本毕业项目构建自托管版本，并在成本和通过率上与托管方案进行对比。

**类型：** 毕业项目
**语言：** Python（智能体）、TypeScript（GitHub App）、YAML（Actions）
**前置要求：** Phase 11（LLM 工程）、Phase 13（工具）、Phase 14（智能体）、Phase 15（自主）、Phase 17（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P17
**所需时间：** 30 小时

## 问题所在

异步云编码智能体是与交互式编码智能体（毕业项目 01）不同的产品类别。交互方式是一个 GitHub 标签。你给 issue 打上 `@agent fix this` 标签，一个 worker 在云沙箱中启动，克隆仓库、运行测试、编辑文件、验证，并以智能体的推理说明作为正文打开 PR。没有交互循环，没有终端。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 都趋向这一形态。

工程挑战很具体：环境复现（智能体必须从零构建仓库，没有缓存的开发镜像）、不稳定测试（必须重跑或隔离）、凭证作用域（具有最小精细权限的 GitHub App）、每个仓库每天的预算强制，以及禁止强制推送策略。毕业项目度量通过率、成本和安全性，与托管方案对比。

## 概念说明

触发器是 GitHub webhook（issue 标签或 PR 评论）。调度器将任务入队到 ECS Fargate 或 Lambda。worker 将仓库拉入 Daytona 或 E2B 沙箱，使用从仓库推断的通用 Dockerfile（语言、框架）。智能体对 Claude Opus 4.7 或 GPT-5.4-Codex 运行 mini-swe-agent 或 SWE-agent v2 循环。它迭代执行：读取代码、提出修复、应用补丁、运行测试。

验证是关键步骤。PR 打开前必须在沙箱中通过完整 CI。计算覆盖率增量；如果超过阈值为负，PR 仍然打开但会被标记为 `needs-review`。智能体将推理说明作为 PR 描述加上 `@agent` 线程发布，评审者可以在其中 ping 进行后续跟进。

安全性通过两个不同的 GitHub 面来限定作用域：App 提供短期安装令牌，具有 `workflows: read` 和窄范围的仓库内容/PR 权限；分支保护（而非 App 权限）强制执行"不允许直接写入 `main`"和"不允许强制推送"——App 永远不会被加入绕过列表。对 `.github/workflows` 的路径限定只读访问不是真正的 GitHub App 原语，因此智能体对文件编辑的允许列表必须在 worker 端强制执行。每个仓库每天的预算上限在调度器端强制执行（例如，每个仓库每天最多 5 个 PR，每个 PR 20 美元）。

## 架构

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技术栈

- 触发器：具有精细令牌的 GitHub App；通过 Lambda 或 Fly.io 接收 webhook
- Worker：ECS Fargate 任务（或 GitHub Actions 自托管运行器）
- 沙箱：每个任务一个 Daytona devcontainer 或 E2B 沙箱
- 智能体循环：mini-swe-agent 基线或 SWE-agent v2，使用 Claude Opus 4.7 / GPT-5.4-Codex
- 检索：tree-sitter 仓库地图 + ripgrep
- 验证：沙箱内完整 CI + 覆盖率增量门控
- 可观测性：Langfuse，PR 正文中链接的每个 PR 追踪归档
- 预算：每个仓库每日美元上限；每个仓库每天最大 PR 数

## 开始构建

1. **GitHub App。** 精细安装令牌：issues 读写、pull_requests 写、contents 读写、workflows 读。分支保护（唯一能实现此功能的层面）强制执行"不允许直接推送到 `main`"和"不允许强制推送"；App 不在绕过列表中。worker 在提议的 diff 上作为允许列表检查强制执行"不允许写入 `.github/workflows` 下的文件"，因为 GitHub App 权限不是路径限定的。

2. **Webhook 接收器。** Lambda 函数接收 issue 标签 / PR 评论 webhook。按标签 `@agent fix this` 过滤。入队到 SQS。

3. **调度器。** 从 SQS 弹出任务。强制执行每个仓库每天的预算。启动 ECS Fargate 任务，携带仓库 URL、issue 正文和全新的 Daytona 沙箱。

4. **环境推断。** 检测语言（Python、Node、Go、Rust）和包管理器（uv、pnpm、go mod、cargo）。如果不存在则动态生成 Dockerfile。

5. **智能体循环。** mini-swe-agent 或 SWE-agent v2，使用 Claude Opus 4.7。工具：ripgrep、tree-sitter 仓库地图、read_file、edit_file、run_tests、git。硬限制：20 美元成本、30 分钟挂钟时间、30 轮智能体交互。

6. **验证。** 循环结束后，在沙箱中运行完整测试套件。通过 jacoco / coverage.py 计算覆盖率增量。如果 CI 失败：停止，不打开 PR。如果覆盖率下降超过 2%：打开 PR 并标记 `needs-review`。

7. **PR 发布。** 推送智能体分支。通过 GitHub API 打开 PR，包含：标题、推理说明、diff 摘要、追踪 URL、成本、轮次。

8. **凭证卫生。** worker 使用短期 GitHub App 安装令牌运行。日志在归档前清洗敏感信息。

9. **评估。** 30 个预设的不同难度内部 issue。度量通过率、PR 质量（diff 大小、风格、覆盖率）、成本、延迟。在相同 issue 上与 Cursor Background Agents 和 AWS Remote SWE Agents 对比。

## 使用示例

```
# 在 github.com 上
  - 用户给 issue #842 打上 `@agent fix this` 标签
  - 14 分钟后出现 PR #1903
  - 正文：
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付成果

`outputs/skill-issue-to-pr.md` 是交付物。一个 GitHub App + 异步云 worker，将标记的 issue 转化为可评审的 PR，具有有界成本和限定凭证。

| 权重 | 标准 | 度量方式 |
|:-:|---|---|
| 25 | 30 个 issue 的通过率 | 端到端成功（CI 绿灯 + 覆盖率正常） |
| 20 | PR 质量 | diff 大小、覆盖率增量、风格一致性 |
| 20 | 每个已解决 issue 的成本和延迟 | 每个 PR 的美元成本和挂钟时间 |
| 20 | 安全性 | 限定令牌、每个仓库预算、禁止强制推送、凭证卫生 |
| 15 | 操作员体验 | 推理说明评论、重试能力、@提及后续跟进 |
| **100** | | |

## 练习

1. 增加"修复不稳定测试"模式：标签 `@agent stabilize-flake TestX` 在沙箱中运行测试 50 次，提出稳定它的最小变更。

2. 在三个共享 issue 上与 Cursor Background Agents 对比成本。报告各工具在何处胜出。

3. 构建预算仪表盘：每个仓库每天的成本、每个用户的成本。异常告警。

4. 构建"试运行"模式：不运行 CI 就打开草稿 PR，让评审者低成本审查计划。

5. 增加保留策略：超过 7 天未合并的 PR 分支自动删除。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|-----------------|------------------------|
| GitHub App | "限定作用域的机器人身份" | 具有精细权限 + 短期安装令牌的 App |
| 异步云智能体 | "后台智能体" | 在云沙箱中运行的非交互式 worker，而非终端 |
| 环境推断 | "Dockerfile 合成" | 检测语言 + 包管理器，缺失时生成 Dockerfile |
| 验证 | "沙箱内 CI" | 打开 PR 前在 worker 内运行完整测试套件 |
| 覆盖率增量 | "覆盖率保持" | 从基线到智能体分支的测试覆盖率百分比变化 |
| 每个仓库预算 | "每日上限" | 在调度器端强制执行的美元和 PR 数量上限 |
| 推理说明 | "PR 正文解释" | 智能体对变更内容和原因的摘要；必须包含在 PR 正文中 |

## 延伸阅读

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 标准异步云智能体参考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 参考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商业替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex) — 托管竞品
- [Google Jules](https://jules.google) — Google 的托管版本
- [Factory Droids](https://www.factory.ai) — 备选商业参考
- [GitHub App documentation](https://docs.github.com/en/apps) — 限定作用域的机器人身份
- [Daytona cloud sandboxes](https://daytona.io) — 参考沙箱
