# 毕业项目 10 — 多智能体软件工程团队

> SWE-AF 的工厂架构、MetaGPT 的角色提示、AutoGen 0.4 的类型化 actor 图、Cognition 的 Devin 和 Factory 的 Droids 都收敛到了 2026 年的同一形态：一个架构师规划，N 个编码者在并行 worktree 中工作，一个审查者门控，一个测试者验证。并行 worktree 将墙钟时间转化为吞吐量。共享状态和交接协议成为失败面。毕业项目是构建团队，在 SWE-bench Pro 上评估，并报告哪些交接会断裂以及频率。

**类型：** 毕业项目
**语言：** Python / TypeScript（智能体）、Shell（worktree 脚本）
**前置要求：** 阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（智能体）、阶段 15（自主系统）、阶段 16（多智能体）、阶段 17（基础设施）
**涉及阶段：** P11 · P13 · P14 · P15 · P16 · P17
**所需时间：** 40 小时

## 问题所在

单智能体编码框架在大型任务上遇到了天花板。不是因为任何单个智能体弱，而是因为 20 万 Token 的上下文无法容纳架构计划加四个并行代码库切片加审查评论加测试输出。多智能体工厂拆分问题：架构师拥有计划，编码者在并行 worktree 中并行拥有实现，审查者门控，测试者验证。SWE-AF 的"工厂"架构、MetaGPT 的角色、AutoGen 的类型化 actor 图——三种框架描述同一形态。

失败面在于交接。架构师规划了编码者无法实现的内容。编码者产生了冲突的 diff。审查者批准了幻觉修复。测试者与仍在写代码的编码者竞争。你将构建这样一个团队，在 50 个 SWE-bench Pro issue 上运行，追踪每次交接，并发布事后分析。

## 概念说明

角色是类型化的智能体。**架构师**（Claude Opus 4.7）读取 issue，编写计划，将其拆分为带明确接口的子任务。**编码者**（Claude Sonnet 4.7，N 个并行实例，每个在 `git worktree` + Daytona 沙箱中）独立实现子任务。**审查者**（GPT-5.4）读取合并后的 diff，要么批准要么请求具体更改。**测试者**（Gemini 2.5 Pro）在隔离环境中运行测试套件，报告通过/失败及制品。

通过共享任务板（文件支持或 Redis）通信。每个角色消费其被允许处理的任务。交接是 A2A 协议类型的消息。协调关注点：合并冲突解决（协调者角色或自动三路合并）、共享状态同步（计划在编码者开始后冻结；重新计划是独立事件）、审查者门控（审查者不能批准自己的更改或其提议的更改）。

Token 放大是隐藏成本。每个角色边界都增加摘要提示和交接上下文。40 轮单智能体运行变成四个角色共 160 轮。评分标准特别衡量 Token 效率与单智能体基线的对比，因为问题不是"多智能体是否有效"而是"它是否每美元获胜"。

## 架构

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技术栈

- 编排：LangGraph，带共享状态 + 每智能体子图
- 消息：A2A 协议（Google 2025）用于类型化智能体间消息
- 模型：Opus 4.7（架构师）、Sonnet 4.7（编码者）、GPT-5.4（审查者）、Gemini 2.5 Pro（测试者）
- Worktree 隔离：每个编码者 `git worktree add` + Daytona 沙箱
- 合并协调器：自定义三路合并 + LLM 调解的冲突解决
- 评估：SWE-bench Pro（50 个 issue）、SWE-AF 场景、HumanEval++ 用于单元测试
- 可观测性：Langfuse，带角色标签 Span，每智能体 Token 计量
- 部署：K8s，每个角色作为独立 Deployment + 基于积压的 HPA

## 开始构建

1. **任务板。** 文件支持的 JSONL，含类型化消息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。智能体订阅标签。

2. **架构师。** 读取 GitHub issue，使用要求明确子任务接口（涉及的文件、公共函数、测试影响）的计划模板运行 Opus 4.7。发出一个 `plan_request`，包含子任务的 DAG。

3. **编码者。** N 个并行工作者，每个从任务板认领一个子任务。每个启动全新的 `git worktree add` 分支和 Daytona 沙箱。实现子任务。发出 `diff_ready`，包含 patch + 测试差异。

4. **合并协调器。** 所有编码者完成后，三路合并 N 个分支到暂存分支。仅在文件级重叠存在时使用 LLM 调解冲突解决。

5. **审查者。** GPT-5.4 读取合并后的 diff。不能批准自己编写的 diff。发出 `approved`（无操作）或 `review_feedback`，包含路由回相关编码者的具体更改请求。

6. **测试者。** Gemini 2.5 Pro 在干净沙箱中运行测试套件。捕获制品。发出 `test_passed` 或 `test_failed`，附带堆栈跟踪。失败测试循环回拥有失败子任务的编码者。

7. **交接计量。** 每个跨越角色边界的消息在 Langfuse 中获得一个 Span，包含负载大小和使用的模型。计算每子任务 Token 放大（coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens）。

8. **评估。** 在 50 个 SWE-bench Pro issue 上运行。与单智能体基线（单个 worktree 中的单个 Sonnet 4.7）对比 pass@1 和 $/已解决 issue。

9. **事后分析。** 对每个失败 issue，识别断裂的交接（计划太模糊、合并冲突、审查者误批准、测试者闪烁）。生成交接失败直方图。

## 使用示例

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付成果

`outputs/skill-multi-agent-team.md` 是交付成果。给定一个 issue URL 和并行级别，团队生成可合并的 PR，附带每角色 Token 计量。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 匹配的 50 个 issue 子集，pass@1 |
| 20 | 并行加速 | 与单智能体基线的墙钟时间对比 |
| 20 | 审查质量 | 注入 bug 探针上的误批准率 |
| 20 | Token 效率 | 每已解决 issue 的总 Token vs 单智能体 |
| 15 | 协调工程 | 合并冲突解决、交接失败直方图 |
| **100** | | |

## 练习

1. 在运行中途向 diff 注入一个明显 bug（主体前多一个 `return None`）。衡量审查者的误批准率。调整审查者提示直到误批准低于 5%。

2. 减少到两个编码者（架构师 + 编码者 + 审查者 + 测试者，编码者顺序运行两个子任务）。对比墙钟时间和通过率。

3. 用单写者约束（子任务触及不相交的文件集）替换合并协调器。衡量架构师的规划负担。

4. 将审查者从 GPT-5.4 替换为 Claude Opus 4.7。衡量误批准率和 Token 成本差异。

5. 添加第五个角色：文档编写者（Haiku 4.5）。审查后生成 changelog 条目。衡量文档质量是否值得额外 Token 支出。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 并行 worktree | "隔离分支" | `git worktree add` 为每个编码者生成全新工作树 |
| 任务板 | "共享消息总线" | 智能体订阅的类型化消息的文件或 Redis 存储 |
| 交接 | "角色边界" | 从一个角色上下文到另一个的任何消息 |
| Token 放大 | "多智能体开销" | 跨角色的总 Token / 同一任务的单智能体 Token |
| A2A 协议 | "智能体对智能体" | Google 2025 年的类型化智能体间消息规范 |
| 合并协调器 | "集成器" | 运行三路合并和调解冲突的组件 |
| 误批准 | "审查者幻觉" | 审查者批准了含已知 bug 的 diff |

## 延伸阅读

- [SWE-AF 工厂架构](https://github.com/Agent-Field/SWE-AF) — 2026 年多智能体工厂参考
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 基于角色的多智能体框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — 微软的类型化 actor 框架
- [Cognition AI（Devin）](https://cognition.ai) — 参考产品
- [Factory Droids](https://www.factory.ai) — 备选参考产品
- [Google A2A 协议](https://developers.google.com/agent-to-agent) — 智能体间消息规范
- [git worktree 文档](https://git-scm.com/docs/git-worktree) — 隔离底层
- [SWE-bench Pro](https://www.swebench.com) — 评估目标
