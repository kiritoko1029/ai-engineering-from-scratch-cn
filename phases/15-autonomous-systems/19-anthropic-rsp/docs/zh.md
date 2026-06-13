# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 于 2026 年 2 月 24 日生效，替代了 2023 年的政策。两层缓解：Anthropic 单方面将做什么 vs 被框架为行业范围建议的内容（包括 RAND SL-4 安全标准）。新增前沿安全路线图和风险报告作为常设文档而非一次性交付物。取消了 2023 年的暂停承诺。引入 AI R&D-4 阈值：一旦跨越，Anthropic 必须发布识别不对齐风险和缓解措施的正面案例。Claude Opus 4.6 未跨越。Anthropic 在 v3.0 公告中表示"自信地排除这一点正变得困难"。SaferAI 将 2023 年 RSP 评为 2.2 分；他们将 v3.0 降级为 1.9 分，将 Anthropic 放入与 OpenAI 和 DeepMind 相同的"弱"RSP 类别。定性阈值取代了 2023 年的定量承诺；移除暂停条款是最尖锐的退步。

**类型：** 学习
**语言：** Python（标准库，RSP 阈值决策引擎）
**前置要求：** 第 15 阶段 · 06（AAR），第 15 阶段 · 07（RSI）
**所需时间：** 约45分钟

## 问题所在

前沿实验室发布的扩展政策部分是技术文档，部分是治理文档，部分是对监管者的信号。RSP v3.0 是 Anthropic 当前的文档。仔细阅读它很重要，不是因为遵守它是有约束力的（它不是），而是因为框架塑造了实验室如何构想灾难性风险以及如何向公众传达权衡。

v3.0 vs v2.0 的差异是有用的单元。新增了什么：前沿安全路线图、风险报告、AI R&D-4 阈值。移除了什么：2023 年的暂停承诺。重新框架了什么：两层缓解计划，分为 Anthropic 单方面和行业建议。外部审查——SaferAI——将分数从 2.2（v2）降级到 1.9（v3.0）。这就是扩展政策如何在看起来更精炼的同时变得更不严格。

## 概念说明

### 两层缓解计划

- **Anthropic 单方面行动**：无论其他实验室做什么 Anthropic 都会做的事情。阈值以上的训练停止、特定安全措施、特定部署门。
- **行业范围建议**：Anthropic 认为行业应集体做的事情。包括 RAND SL-4 安全标准。这些不是 Anthropic 的承诺；它们是政策倡导。

两层结构不在 v2 中。这意味着读者需要查看每个承诺在哪一列。"行业范围建议"列中的安全措施不是 Anthropic 的承诺；它是 Anthropic 的希望。

### AI R&D-4 阈值

这是 RSP v3.0 命名为重要下一阈值的能力级别。具体来说：一个能以有竞争力的成本自动化相当一部分 AI 研究的模型。一旦 Anthropic 相信模型跨越它，他们必须在继续扩展之前发布识别不对齐风险和缓解措施的正面案例。

根据 v3.0 公告，Claude Opus 4.6 未跨越。文档补充："自信地排除这一点正变得困难。"这种措辞很重要；它承认阈值足够接近，是一个活跃关注点，而非推测性限制。

第 6 课（自动对齐研究）和第 7 课（递归自我改进）直接输入到这个阈值。自动对齐研究智能体达到研究质量门槛是 AI R&D-4 阈值正在接近的证据。

### 前沿安全路线图和风险报告

v3.0 将两种制品提升为常设文档：

- **前沿安全路线图**：前瞻性文档，描述计划的安全工作、能力预期和缓解研究。
- **风险报告**：发布后特定模型的回顾性文档，描述观察到的能力和残余风险。

两者都是公开的。两者都按声明的节奏更新。效用是：读者可以跟踪 Anthropic 在路线图中说他们会做的事情与在风险报告中报告的内容的比较。

### 移除暂停条款

2023 年 RSP 包含明确的暂停承诺：如果模型跨越特定能力阈值，训练将暂停直到缓解措施到位。v3.0 用更温和的措辞替代了明确的暂停（发布正面案例，如果缓解措施充分则继续）。SaferAI 和其他分析师直接称这是新文件中最强的退步。

政策论证：2023 年的定量阈值到 2026 年代的能力基准下变得不可达到，因为基准本身被重新缩放。反对论点：扩展政策中的暂停条款是承诺装置；移除它移除了政策的可信度。

### SaferAI 的降级

SaferAI 是一个评估 RSP 风格文档的独立组织。他们的公开评级：2023 年 Anthropic RSP 得分 2.2（4.0 是当前最佳 RSP，1.0 是名义上的）。v3.0 得分 1.9。这使 Anthropic 从"中等"变为"弱"，与 OpenAI 和 DeepMind 一起处于弱类别。

根据 SaferAI 的降级因素：
- 定性阈值取代了定量阈值。
- 暂停承诺被移除。
- AI R&D-4 阈值缓解被描述为"正面案例"而非具体措施。
- 审查机制依赖 Anthropic 的安全咨询委员会，独立监督有限。

### 本课不是什么

这不是合规课。RSP v3.0 不是法规；没有什么强制 Anthropic 遵循它。本课在于以应得的特异性和怀疑态度阅读文档。扩展政策是前沿实验室发出的关于灾难性风险姿态的主要公开信号。阅读它们是一项实用技能，对于任何工作依赖前沿能力的人来说都是如此。

## 开始构建

`code/main.py` 实现了一个小决策引擎，反映了 RSP 阈值评估的形状：给定候选模型和一组能力测量，返回 AI R&D-4 阈值是否被跨越、所需的正面案例部分以及部署是否可以继续。它有意简单；重点是使文档的逻辑显式。

## 交付产出

`outputs/skill-scaling-policy-review.md` 根据 v3.0 参考审查扩展政策（Anthropic、OpenAI、DeepMind 或内部）：两层结构、阈值、暂停承诺、独立审查。

## 练习

1. 运行 `code/main.py`。输入三个不同能力级别的合成模型。确认阈值评估器按预期行为并产生正确的正面案例模板。

2. 完整阅读 RSP v3.0（32 页）。识别每个在"行业范围建议"层中的承诺。哪些在 v2 中是"Anthropic 单方面"？

3. 阅读 SaferAI 的 RSP 评分方法论。通过将他们的评分标准应用于文档来重现他们对 v3.0 的 1.9 分。哪一行评分标准最驱动了降级？

4. 2023 年暂停承诺被移除。提出一个替代承诺，在承认 2026 年基准重新缩放问题的同时保持政策的可信度。

5. 比较 RSP v3.0 与 OpenAI Preparedness Framework v2（第 20 课）。选择一个 v3.0 更强的领域。选择一个 Preparedness Framework 更强的领域。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|---|---|---|
| RSP | "Anthropic 的扩展政策" | Responsible Scaling Policy；v3.0 于 2026 年 2 月 24 日生效 |
| AI R&D-4 | "研究自动化阈值" | 以有竞争力的成本自动化相当 AI 研究的能力 |
| Affirmative case（正面案例） | "安全论证" | 发布的论点，识别风险且缓解措施充分 |
| Frontier Safety Roadmap（前沿安全路线图） | "前瞻计划" | 关于计划安全工作和预期能力的常设文档 |
| Risk Report（风险报告） | "模型的回顾" | 关于发布后观察到的能力和残余风险的常设文档 |
| Two-tier mitigation（两层缓解） | "单方面 vs 行业" | Anthropic 承诺 vs 行业建议，分离 |
| Pause commitment（暂停承诺） | "2023 年条款" | 暂停训练的明确承诺；在 v3.0 中移除 |
| SaferAI rating（SaferAI 评级） | "独立 RSP 评级" | 第三方评分标准；v3.0 得 1.9（v2 是 2.2） |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — 完整的 32 页政策。
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — v2 变更摘要。
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — 从 RSP v3.0 链接的常设文档。
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — 当前前沿模型的回顾。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 将 AI R&D-4 连接到测量的自主性。
