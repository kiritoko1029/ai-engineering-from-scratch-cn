# 智能体指令作为可执行约束

> 以散文形式写的指令是愿望。以约束形式写的指令是测试。工作台将每条规则转化为智能体在运行时可以检查、审查者在事后可以验证的东西。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 32（最小工作台）
**所需时间：** 约50分钟

## 学习目标

- 区分路由散文与操作规则。
- 将启动规则、禁止操作、完成定义、不确定处理和审批边界表达为机器可检查的约束。
- 实现一个规则检查器，对运行按规则集打分。
- 使规则集对差异友好，以便审查可以看到变更。

## 问题所在

典型的 `AGENTS.md` 读起来像入职文档。它告诉智能体"要仔细"、"彻底测试"、"不确定就问"。三天后，智能体发布了一个没有测试的变更，写入了禁止的目录，而且从未询问，因为它根本不知道界限在哪里。

当指令是操作性的时强大，当指令是愿望性的时无力。修复方法是编写工作台可以解释、审查者可以打分的规则。

## 概念说明

规则放在 `docs/agent-rules.md` 中，远离简短的根路由器。每条规则有名称、类别和检查。

```mermaid
flowchart LR
  Router[AGENTS.md] --> Rules[docs/agent-rules.md]
  Rules --> Checker[rule_checker.py]
  Checker --> Report[rule_report.json]
  Report --> Reviewer[Reviewer]
```

### 覆盖大多数规则的五个类别

| 类别 | 规则回答的问题 | 示例 |
|------|-------------|------|
| 启动 | 工作开始前什么必须为真？ | "状态文件存在且新鲜" |
| 禁止 | 什么绝不能发生？ | "不要编辑 `scripts/release.sh`" |
| 完成定义 | 什么证明任务完成？ | "pytest 退出码为 0 且验收行通过" |
| 不确定 | 智能体不确定时做什么？ | "打开问题笔记而不是猜测" |
| 审批 | 什么需要人工审批？ | "任何新依赖、任何生产写入" |

不适合这五个中任何一个的规则通常应该是两条规则。强制拆分。

### 规则是机器可读的

每条规则有 slug、类别、一行描述和一个 `check` 字段，指向 `rule_checker.py` 中的函数。添加规则意味着添加检查；检查器随工作台增长。

### 规则是差异友好的

规则在单个 markdown 文件中每个标题一条。重命名在差异中可见。新规则放在其类别顶部。过时的规则被删除而非注释掉，因为工作台是真相来源，而非团队上季度感受的聊天记录。

### 规则 vs 框架护栏

框架护栏（OpenAI Agents SDK 护栏、LangGraph 中断）在运行时级别执行规则。本课的规则集是那些护栏实现的人类可读、可审查的契约。两者都需要：运行时在轮次中捕获违规，规则集证明运行时在做正确的事。

### 渐进式披露：地图，不是百科全书

`AGENTS.md` 不断增长的原因是每个事故添加一条规则而没有事故删除规则。一年后，文件两千行，智能体读了第一屏就用完了注意力预算，只能按照被告知的一小部分行事。巨型指令文件失败的原因与四十页入职文档失败的原因相同：读者浏览一次就再也不会回到重要的部分。

修复不是更短的文件。而是分层的文件。根路由器保持足够小以供每次会话阅读，只包含指针。深度存在于智能体仅在任务触及时才加载的主题文件中。给智能体一张地图，而不是整本百科全书，让它走到需要的页面。

```
AGENTS.md                  # 路由器，< 50 行：这个仓库是什么、看哪里、5 条硬规则
docs/
  agent-rules.md           # 完整规则集（本课）
  architecture.md          # 任务触及时加载
  testing.md               # 任务写入或运行测试时加载
  deploy.md                # 仅在发布工作时加载，门控在审批规则后
feature_list.json          # 待办列表（第 14 阶段 · 36）
```

| 层级 | 存放位置 | 读取时机 | 大小预算 |
|------|---------|---------|---------|
| 路由器 | `AGENTS.md` | 每次会话，始终 | 约 50 行以内 |
| 规则 | `docs/agent-rules.md` | 每次会话，启动时 | 每个类别一屏 |
| 主题文档 | `docs/<topic>.md` | 仅当任务触及时 | 按需深入 |

两个测试保持分层诚实。可达性测试：智能体应从路由器最多两跳到达任何规则，因此路由器必须按路径链接每个主题文档，而非用散文描述。新鲜度测试：路由器足够短，审查者在每个 PR 上重读它，这是唯一阻止它悄悄长回它所替代的百科全书的方法。一个不再解析的指针比缺失的规则更糟糕，因此路由器中的断链本身就是一个启动检查违规。

## 开始构建

`code/main.py` 提供：

- `agent-rules.md` 解析器，将规则加载到数据类中。
- `rule_checker.py` 风格的检查函数，每个 `check` 引用一个。
- 演示智能体运行，违反两条规则以及一次检查通过来捕获它们。

运行方式：

```
python3 code/main.py
```

输出：解析后的规则集、运行追踪、每条规则的通过/失败，以及保存在脚本旁边的 `rule_report.json`。

## 实际生产模式

三种模式将能持续一个季度的规则集与一周内衰退的区分开来。

**写入时的严重性标记。** 每条规则携带 `severity`：`block`、`warn` 或 `info`。检查器报告全部三种；运行时仅在 `block` 时拒绝。大多数团队早期过度标记严重性，然后在截止日期压力下悄悄降低；写入时标记迫使提前校准。与验证门（第 14 阶段 · 38）配对，后者将任何 `block` 规则的覆盖签名到 `overrides.jsonl` 审计日志中。

**规则过期作为强制函数。** 每条规则携带 `expires_at` 日期（默认从编写起 90 天）。当未过期规则连续 60 天零违规时，检查器发出警告；下次季度审查要么保留它、降级为 `info`、要么删除它。Cloudflare 的生产 AI Code Review 数据（2026 年 4 月，30 天内 5,169 个仓库 131,246 次审查运行）显示，带显式过期的规则集每个仓库保持在 30 条以内；没有过期的增长到 80+ 条，大多数从未触发。

**Markdown 为源，JSON 为缓存。** `agent-rules.md` 是编辑文件；`agent-rules.lock.json` 是检查器在热路径中读取的缓存。锁由预提交钩子重新生成。Markdown 差异可审查；JSON 解析远离每轮调用。与 `package.json` / `package-lock.json` 和 `Cargo.toml` / `Cargo.lock` 相同的形态。

## 使用建议

生产中：

- Claude Code、Codex、Cursor 在会话开始时读取规则并在拒绝操作时引用它们。检查器在 CI 中重新运行以捕获静默漂移。
- OpenAI Agents SDK 护栏将相同的检查注册为输入和输出护栏。Markdown 是文档界面；SDK 是运行时界面。
- LangGraph 中断在进行中的节点违反规则时触发。中断处理器读取规则，询问人类，然后恢复。

规则集在三者之间可移植，因为它只是 markdown 加函数名。

## 交付产物

`outputs/skill-rule-set-builder.md` 访谈项目负责人，将其现有散文指令分类到五个类别，并输出带版本的 `agent-rules.md` 加检查器存根。

## 练习

1. 如果你的产品确实需要，添加第六个类别。论证为什么它不能归入五个之一。
2. 扩展检查器使规则可以携带严重性（`block`、`warn`、`info`），报告相应聚合。
3. 将检查器接入 CI：如果 block 严重性规则在最新智能体运行上失败则构建失败。
4. 为每条规则添加"过期"字段。90 天无检查失败后，规则进入审查。
5. 找一个真实的 `AGENTS.md` 并重写为五类规则。它的多少行是操作性的？多少行是愿望性的？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 操作性规则 | "真正的指令" | 工作台可以在运行时检查的规则 |
| 愿望性规则 | "要小心" | 没有检查的规则；要么删除要么升级 |
| 完成定义 | "验收" | 客观的、基于文件的任务完成证明 |
| Block 严重性 | "硬规则" | 违规停止运行；不能被静默覆盖 |
| 规则过期 | "过时规则清理" | N 天无失败的规则进入退役 |

## 延伸阅读

- [OpenAI Agents SDK 护栏](https://platform.openai.com/docs/guides/agents-sdk/guardrails)
- [LangGraph 中断](https://langchain-ai.github.io/langgraph/how-tos/human_in_the_loop/breakpoints/)
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [Rick Hightower，Agent RuleZ: A Deterministic Policy Engine](https://medium.com/@richardhightower/agent-rulez-a-deterministic-policy-engine-for-ai-coding-agents-9489e0561edf) — 生产中的 block/warn/info 严重性
- [Cloudflare，Orchestrating AI Code Review at Scale](https://blog.cloudflare.com/ai-code-review/) — 131k 次审查运行、规则组合经验
- [microservices.io，GenAI development platform — part 1: guardrails](https://microservices.io/post/architecture/2026/03/09/genai-development-platform-part-1-development-guardrails.html) — 规则与 CI 之间的纵深防御
- [Type-Checked Compliance: Deterministic Guardrails（arXiv 2604.01483）](https://arxiv.org/pdf/2604.01483) — Lean 4 作为规则即检查的上限
- [logi-cmd/agent-guardrails](https://github.com/logi-cmd/agent-guardrails) — 合并门实现：范围、变异测试、违规预算
- 第 14 阶段 · 32 — 本规则集嵌入的最小工作台
- 第 14 阶段 · 38 — 消费规则报告的验证门
- 第 14 阶段 · 39 — 为规则合规打分的审查者智能体
