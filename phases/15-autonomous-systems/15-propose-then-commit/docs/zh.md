# 人机协同：先提议后提交

> 2026 年关于人机协同的共识是具体的。它不是"智能体询问，用户点击批准"。它是先提议后提交：提议的动作以幂等键持久化到持久存储；呈现给审查者，附带意图、数据沿袭、触及的权限、爆炸半径和回滚计划；仅在正面确认后提交；执行后验证以确认副作用确实发生。LangGraph 的 `interrupt()` 加 PostgreSQL 检查点、Microsoft Agent Framework 的 `RequestInfoEvent` 和 Cloudflare 的 `waitForApproval()` 都实现了相同的形状。经典的故障模式是橡皮图章式批准："批准？"被点击而没有审查。已记录的缓解措施是带有显式清单的挑战与响应。

**类型：** 学习
**语言：** Python（标准库，带幂等性的先提议后提交状态机）
**前置要求：** 第 15 阶段 · 12（持久化执行），第 15 阶段 · 14（触发器）
**所需时间：** 约60分钟

## 问题所在

智能体执行一个动作。用户必须决定：批准还是不批准。如果决定是即时的，它可能不是审查。如果决定是结构化的，它慢但可信。工程问题是如何使结构化审查成为阻力最小的路径。

2023 年代的人机协同模式是同步提示："智能体想发送邮件给 X，正文是 Y——批准？"用户点击批准。每个人都觉得系统安全。在实践中，这个界面被大量橡皮图章式使用：用户快速批准，批准预测性低，当智能体出错时，审计跟踪显示用户无法回忆的长批准历史。

2026 年的模式——先提议后提交——将人机协同移到持久化基底上，附加结构化元数据，并要求正面提交。每个托管智能体 SDK 都发布了一个版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名称不同；形状相同。

## 概念说明

### 先提议后提交状态机

1. **提议。** 智能体产生一个提议的动作。持久化到持久存储（PostgreSQL、Redis、Durable Object）。包括：
   - 意图（为什么智能体做这个）
   - 数据沿袭（什么来源导致了这个提议）
   - 触及的权限（哪些范围/文件/端点）
   - 爆炸半径（最坏情况是什么）
   - 回滚计划（如果提交，我们如何撤消）
   - 幂等键（每个提议唯一；重新提交返回相同记录）
2. **呈现。** 审查者看到带有所有元数据的提议。审查者是人（不是智能体审查自己）。
3. **提交。** 正面确认。动作执行。
4. **验证。** 执行后，读回副作用并确认。如果验证步骤失败，系统处于已知坏状态，警报启动。

### 幂等键

没有幂等键，瞬态故障后的重试可能双重执行已批准的动作。具体例子：用户批准"从 A 转账 100 美元到 B"。网络抖动。工作流重试。用户批准了一次但转账执行了两次。幂等键将批准绑定到单一、唯一的副作用；第二次执行是空操作。

这与 Stripe 和 AWS API 使用的幂等模式相同。Microsoft Agent Framework 文档中明确将其用于智能体批准。

### 持久化：为什么批准比进程更持久

批准等候室是智能体不拥有的一段状态。工作流被暂停（第 12 课）。当批准到达时，工作流从恰好该点恢复。这就是为什么 LangGraph 将 `interrupt()` 与 PostgreSQL 检查点配对而不仅仅是内存状态——两天后的批准仍然能找到完整的工作流。

### 橡皮图章式批准与挑战与响应缓解

人机协同的默认 UI（"批准"/"拒绝"按钮）产生没有真正审查的快速批准。已记录的缓解措施：一个挑战与响应清单，要求在批准按钮启用之前对特定问题给出正面回答。具体形状：

- "你理解这触及什么资源吗？ [ ]"
- "你验证了爆炸半径是可接受的吗？ [ ]"
- "如果失败你有回滚计划吗？ [ ]"

不是为了官僚而官僚——是一个强制功能。无法勾选框的审查者要么要求澄清（提升）要么拒绝（安全默认）。Anthropic 智能体安全研究明确引用清单驱动的人机协同作为橡皮图章式批准模式的缓解措施。

### 什么算作重要

不是每个动作都需要先提议后提交。2026 年指导：

- **重要动作**（始终人机协同）：不可逆写入、金融交易、外发通信、生产数据库更改、破坏性文件系统操作。
- **可逆动作**（有时人机协同）：本地文件编辑、暂存环境更改、有明确回滚的可逆写入。
- **读取和检查**（从不人机协同）：读取文件、列出资源、调用只读 API。

### 动作后验证

"提交已运行"与"副作用已发生"不同。网络分区和竞争条件可能产生工作流认为成功而后端未持久化的情况。验证步骤在提交后重新读取目标资源以确认。这与带 `RETURNING` 子句的数据库事务或 `PutObject` 后的 AWS `GetObject` 模式相同。

### EU AI Act 第 14 条

第 14 条要求欧盟高风险 AI 系统的"有效人工监督"。"有效"不是装饰性的。监管语言明确排除橡皮图章模式。在 Microsoft Agent Governance Toolkit 合规文档中，带有挑战与响应的先提议后提交是能通过第 14 条审查的形状。

## 开始构建

`code/main.py` 用标准库 Python 实现了一个先提议后提交状态机。持久存储是 JSON 文件。幂等键是 (thread_id, action_signature) 的哈希。驱动器模拟三种情况：干净的批准流程、瞬态故障后的重试（不得双重执行）以及橡皮图章默认与挑战与响应流程的对比。

## 交付产出

`outputs/skill-hitl-design.md` 审查拟议的人机协同工作流的先提议后提交形状，并标记缺失的元数据、幂等性、验证或挑战与响应层。

## 练习

1. 运行 `code/main.py`。确认已批准提议的重试使用持久记录而不重新执行。现在将幂等键更改为包含时间戳，展示重试双重执行。

2. 用 `rollback` 字段扩展提议记录。模拟一个验证步骤失败的执行。展示回滚自动触发。

3. 阅读 Microsoft Agent Framework 的 `RequestInfoEvent` 文档。识别 API 包含但玩具引擎缺少的一个元数据字段。添加它并解释它保护什么。

4. 为特定动作（例如"发布到公开 Twitter 帐户"）设计一个挑战与响应清单。审查者必须回答哪三个问题？为什么是这三个？

5. 选择一个同步"批准？"提示就足够的情况（不需要持久存储）。解释原因，并说出你接受的风险类别。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|---|---|---|
| Propose-then-commit（先提议后提交） | "两阶段批准" | 持久化提议 + 正面提交 + 验证 |
| Idempotency key（幂等键） | "重试安全令牌" | 每个提议唯一；第二次执行空操作 |
| Data lineage（数据沿袭） | "它从哪里来" | 导致提议的特定源内容 |
| Blast radius（爆炸半径） | "最坏情况" | 如果动作出错的影响范围 |
| Rubber-stamp（橡皮图章） | "快速批准" | 没有真正审查就点击"批准" |
| Challenge-and-response（挑战与响应） | "强制清单" | 审查者必须正面确认特定问题 |
| RequestInfoEvent | "MS 智能体框架原语" | 带结构化元数据的持久化人机协同请求 |
| `interrupt()` / `waitForApproval()` | "框架原语" | LangGraph / Cloudflare 等价的相同形状 |

## 延伸阅读

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`，持久化批准。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 和 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 人机协同作为长时域风险的缓解措施。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高风险系统的监管基线。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 围绕监督的宪法框架。
