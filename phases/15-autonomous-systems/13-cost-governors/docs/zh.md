# 动作预算、迭代上限与成本治理器

> 一个中型电商智能体的月度 LLM 成本在其团队启用"订单跟踪"技能后从 1,200 美元跳到 4,800 美元。这不是定价错误。这是一个找到了新循环并在其中持续消费的智能体。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）编纂了针对此类问题的防御：每请求 `max_tokens`、每任务 token 和美元预算、每日/月上限、迭代上限、分层模型路由、提示缓存、上下文窗口、昂贵动作的人机协同检查点、预算超支的紧急停止开关。Anthropic 的 Claude Code Agent SDK 以不同的名称发布了相同的原语。财务速率限制——例如 10 分钟内超过 50 美元即切断访问——比月度上限更快地捕获循环。

**类型：** 学习
**语言：** Python（标准库，分层成本治理器模拟器）
**前置要求：** 第 15 阶段 · 10（权限模式），第 15 阶段 · 12（持久化执行）
**所需时间：** 约60分钟

## 问题所在

自主智能体在每个回合都花费真实资金。聊天机器人的坏输出是一个坏回复；智能体的坏循环是一张账单。行业记录的此故障模式术语是"拒绝钱包攻击"（Denial of Wallet）——智能体持续推理、持续工具调用、持续计费，没有任何东西阻止它，因为没有设计来阻止它。

修复不是一个数字。它是不同时间尺度和粒度的限制栈：每请求、每任务、每小时、每天、每月。一个良好设计的栈在分钟内捕获失控循环，在小时内捕获缓慢泄漏，在一天内捕获坏发布。当智能体是长时域和自主的时，相同的栈在所有情况下保持预算。

这是一个工程课：数学是简单的，纪律才是团队失败的地方。下面的限制列表都在 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 文档中命名。

## 概念说明

### 成本治理器栈

1. **每请求 `max_tokens`。** 简单。防止任何一个调用发出无界的完成。
2. **每任务 token 预算。** 整个运行中不超过 N 个 token。到达上限时硬停止。
3. **每任务美元预算。** 与 token 相同但以货币计。Claude Code 中的 `max_budget_usd`。
4. **每工具调用上限。** 不超过 N 次 `WebFetch` 调用、N 次 `shell_exec` 调用等。
5. **迭代上限（`max_turns`）。** 智能体循环总迭代次数；防止无限推理循环。
6. **每分钟/每小时/每天/每月上限。** 滚动窗口。在不同时间尺度捕获泄漏。
7. **财务速率限制。** 例如"如果 10 分钟内消费超过 50 美元，切断访问。"在月度上限触发之前捕获基于循环的烧钱。
8. **分层模型路由。** 默认使用较小的模型；仅当分类器判断任务需要时才升级到较大的模型。
9. **提示缓存。** 系统提示和稳定上下文存储在提供商缓存中；重新发送的 token 成本接近零。
10. **上下文窗口。** 压缩/摘要以保持活跃上下文低于阈值；直接减少 token 成本。
11. **昂贵动作的人机协同检查点。** 在已知昂贵的动作（长时间工具调用、大型下载、昂贵的模型升级）之前，需要人工点击。
12. **预算超支的紧急停止开关。** 任何上限触发时会话中止。上限被记录；需要单独的重新启用路径。

### 为什么是栈，而非一个上限

单一的月度上限只在钱包耗尽后才捕获失控智能体。单一的每请求上限在会话级别什么都捕获不到。不同的故障模式需要不同的时间尺度：

- **失控循环**（智能体卡在 5 秒重试）：被速率限制捕获。
- **缓慢泄漏**（智能体每任务做约 2 倍预期工作）：被每日上限捕获。
- **坏发布**（新版本使用 5 倍 token）：被每周/月度上限捕获。
- **合法激增**（真实需求，非错误）：被带清晰日志的小时/天上限捕获。

### Claude Code 的预算面

Claude Code Agent SDK 暴露（公开文档）：

- `max_turns` — 迭代上限。
- `max_budget_usd` — 美元上限；超支时会话中止。
- `allowed_tools` / `disallowed_tools` — 工具允许和拒绝列表。
- 工具使用前的钩子点用于自定义成本核算。

与权限模式阶梯（第 10 课）结合。没有 `max_budget_usd` 的 `autoMode` 会话是无约束的自主性。Anthropic 明确将 Auto Mode 框架为需要预算控制；分类器与成本正交。

### EU AI Act、OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 涵盖 OWASP Agentic Top 10 和 EU AI Act 第 14 条（人工监督）要求。在欧盟的生产中，日志记录和上限强制不是可选的。

### 观察到的 $1,200 → $4,800 案例

Microsoft 文档中的真实案例：一个电商智能体在添加新工具后月度成本增加了三倍。该工具允许智能体在每次会话期间轮询订单状态。没有循环检测。没有每工具上限。没有周增长警报。修复是每工具上限加上每日增长警报。这是一个模板：每个新工具面都是一个新的潜在循环；每个新工具需要自己的上限和自己的警报。

## 开始构建

`code/main.py` 模拟有和没有分层成本治理器栈的智能体运行。模拟的智能体在若干回合后漂移到轮询循环；分层栈在速率窗口内捕获它，而单一的月度上限要到几天后才会触发。

## 交付产出

`outputs/skill-agent-budget-audit.md` 审计拟议智能体部署的成本治理器栈并标记缺失层。

## 练习

1. 运行 `code/main.py`。确认速率限制在轮询循环轨迹上先于迭代上限触发。现在禁用速率限制并测量在迭代上限捕获之前智能体"花费"了多少。

2. 为浏览器智能体（第 11 课）设计一套每工具上限。哪个工具需要最严格的上限？哪个工具可以无风险地无界运行？

3. 阅读 Microsoft Agent Governance Toolkit 文档。列出工具包命名的每种上限类型。将每种映射到一种故障模式（失控循环、缓慢泄漏、坏发布、激增）。

4. 为一个真实任务（例如"分类仓库中的 50 个 issue"）估算隔夜无人值守运行的价格。将 `max_budget_usd` 设为你的点估计的 2 倍。为 2 倍给出理由。

5. Claude Code 的 `max_budget_usd` 在会话总成本上触发。设计一个你会在外部强制执行的补充速率限制。什么触发切断，重新启用是什么样子？

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|---|---|---|
| Denial of Wallet（拒绝钱包攻击） | "失控账单" | 智能体循环在没有上限阻止的情况下产生消费 |
| max_tokens | "每请求上限" | 单次完成大小的天花板 |
| max_turns | "迭代上限" | 会话中智能体循环迭代的天花板 |
| max_budget_usd | "美元紧急停止开关" | 会话成本上限；超支时中止 |
| Velocity limit（速率限制） | "速率上限" | 短窗口内消费的限制（例如 $50 / 10 分钟） |
| Tiered routing（分层路由） | "小模型优先" | 廉价模型默认；仅当分类器保证时升级 |
| Prompt caching（提示缓存） | "缓存的系统提示" | 提供商端缓存将重新发送的 token 成本降至接近零 |
| HITL checkpoint（人机协同检查点） | "人工批准门" | 昂贵动作前需要人工点击 |

## 延伸阅读

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、工具允许列表。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — 成本治理器检查点。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 提供商端成本控制。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/prompt-caching) — 缓存机制。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时域智能体的成本特征。
