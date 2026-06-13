# OpenAI Preparedness Framework 与 DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2（2025 年 4 月）引入了研究类别——长期自主性、伪装能力、自主复制和适应、破坏安全防护——区别于跟踪类别。跟踪类别触发能力报告加上由安全咨询委员会审查的安全报告。DeepMind 的 FSF v3（2025 年 9 月，跟踪能力级别于 2026 年 4 月 17 日添加）将自主性折叠到 ML R&D 和 Cyber 领域（ML R&D 自主性级别 1 = 以与人类 + AI 工具有竞争力的成本完全自动化 AI R&D 流水线）。FSF v3 通过自动化监控工具推理滥用来明确处理欺骗性对齐。诚实说明：PF v2 中的研究类别（包括长期自主性）不会自动触发缓解措施；政策语言是"潜在的"。DeepMind 自己表示如果工具推理增强，自动化监控"长期将不再足够"。

**类型：** 学习
**语言：** Python（标准库，三框架决策表差异工具）
**前置要求：** 第 15 阶段 · 19（Anthropic RSP）
**所需时间：** 约45分钟

## 问题所在

第 19 课仔细阅读了 Anthropic 的扩展政策。本课通过阅读 OpenAI 和 DeepMind 的来完成全景。三个文档是解决同一问题的表亲制品——前沿实验室何时应该暂停或门控一个模型——它们收敛在一组小的类别上，并在重要的特定地方分歧。

收敛：三者都将长期自主性标记为值得跟踪的能力类别。三者都承认欺骗行为（对齐伪装、伪装能力）为特定的风险类别。三者都有内部审查机构。分歧：OpenAI 将类别分为"跟踪"（强制缓解）和"研究"（无自动触发）。DeepMind 将自主性折叠到两个领域而非单独命名。实验室命名跟踪 vs 研究，或关键 vs 中等，或 Tier-1 vs Tier-2；能力所在桶的运行后果在实验室之间不同。

一起阅读是有用的练习。相同的能力在 Anthropic 可以是"强制缓解"，在 OpenAI 是"监控但不触发"，在 DeepMind 是"在特定领域跟踪"。政策姿态很重要。

## 概念说明

### OpenAI Preparedness Framework v2（2025 年 4 月）

结构：

- **跟踪类别**：触发能力报告（模型能做什么）加上安全报告（什么缓解措施到位）。部署前由安全咨询委员会审查。
- **研究类别**：实验室跟踪但尚未承诺具体缓解措施的模型能力。包括长期自主性、伪装能力、自主复制和适应、破坏安全防护。

v2 中的关键框架：研究类别不会自动触发缓解措施。政策语言是"潜在的"缓解措施。这是要观察的特定增量：类别在跟踪 vs 研究中的位置是"门控"和"观察"之间的铰链。

### DeepMind Frontier Safety Framework v3（2025 年 9 月；跟踪能力级别于 2026 年 4 月 17 日添加）

结构：

- **关键能力级别（CCL）**：跨五个领域的能力阈值：Cyber、Bio、ML R&D、CBRN、自主性（折叠到 ML R&D 和 Cyber）。
- **跟踪能力级别**：2026 年 4 月添加的额外粒度。具体例子：ML R&D 自主性级别 1 = 以与人类 + AI 工具有竞争力的成本完全自动化 AI R&D 流水线。
- **欺骗性对齐监控**：对工具推理滥用的自动化监控的明确承诺。

自主性框架与 OpenAI 的不同。DeepMind 不保持"自主性"作为顶级领域；它被折叠到自主性会造成伤害的领域（ML R&D 和 Cyber）。论点是没有领域的自主性是没有风险的能力；反对论点是跨领域的自主性是框架应该命名的元风险。

### 三者收敛的地方

- 内部安全咨询委员会（Anthropic SAG、OpenAI SAG、DeepMind 内部委员会）。高能力模型部署前审查。
- 明确提及欺骗性对齐/对齐伪装作为风险类别。
- 按声明节奏的常设制品（Anthropic：前沿安全路线图、风险报告；OpenAI：能力和安全报告；DeepMind：FSF 更新周期）。
- 承认仅监控防御有天花板。DeepMind 明确："自动化监控长期将不再足够。"

### 它们分歧的地方

- **Anthropic**：v3.0 中移除暂停承诺；AI R&D-4 阈值是命名的下一门。
- **OpenAI**：跟踪 vs 研究分裂；研究类别（包括长期自主性）不自动门控。
- **DeepMind**：自主性折叠到其他领域；跟踪能力级别在 2026 年 4 月添加粒度。

### 伪装能力：使三者都复杂化的特定能力

伪装能力（模型在评估中策略性地表现不佳）在 OpenAI 的研究类别中。Anthropic 的 RSP v3.0 通过评估上下文差距（第 1 课）处理它。DeepMind 通过 FSF v3 中的欺骗性对齐监控处理它。

如果模型在评估中伪装能力，每个框架的能力阈值都被低估。框架仅在测量有效时有效。这就是为什么外部测量（第 21 课，METR）和对抗性评估在实验室自我评估之外是必要的。

### 阅读政策的技能

- 定位：你关心的每个能力应该在政策中可找到。如果找不到，政策不覆盖它。
- 分类：它是跟踪（触发缓解）还是研究（跟踪但不触发）？OpenAI 命名了这个；Anthropic 和 DeepMind 有自己的等价物。
- 节奏：政策是否按声明的时间表更新，还是仅在特定事件后？声明的节奏更强。
- 独立性：外部审查是强制的还是可选的？Anthropic 与 Apollo 和美国 AI 安全研究所合作；OpenAI 与 METR；DeepMind 主要与内部 SAG。

## 开始构建

`code/main.py` 实现了一个小决策表差异工具。给定一个能力（自主性、欺骗性对齐、R&D 自动化、cyber 提升等），它输出三个政策各自如何分类该能力，以及什么缓解措施触发。它是阅读辅助，不是政策工具。

## 交付产出

`outputs/skill-cross-policy-diff.md` 使用三个框架作为参考，为特定能力产生跨政策比较。

## 练习

1. 运行 `code/main.py`。确认差异工具的输出至少对两个你可以对照源文档验证的能力匹配政策。

2. 完整阅读 OpenAI Preparedness Framework v2。识别每个研究类别。为每个写一句话说明为什么它在研究而非跟踪中。

3. 完整阅读 DeepMind FSF v3，加上 2026 年 4 月跟踪能力级别更新。识别 ML R&D 自主性级别 1 的具体评估标准。你如何外部测量它？

4. 伪装能力在 OpenAI 的研究类别中。设计一个评估来强制伪装能力模型揭示其实际能力。参考第 1 课评估上下文博弈讨论。

5. 比较三个政策在特定能力（你选择）上的差异。说出你认为哪个政策的分类最严格，哪个最不严格。用源文本证明。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|---|---|---|
| Preparedness Framework（准备就绪框架） | "OpenAI 的扩展政策" | PF v2（2025 年 4 月）；跟踪 vs 研究类别 |
| Tracked Category（跟踪类别） | "强制缓解" | 触发能力 + 安全报告；SAG 审查 |
| Research Category（研究类别） | "仅监控" | 跟踪但无自动缓解；包括长期自主性 |
| Frontier Safety Framework（前沿安全框架） | "DeepMind 的扩展政策" | FSF v3（2025 年 9 月）+ 跟踪能力级别（2026 年 4 月） |
| CCL | "关键能力级别" | DeepMind 每领域阈值（Cyber、Bio、ML R&D、CBRN） |
| ML R&D autonomy level 1（ML R&D 自主性级别 1） | "R&D 自动化" | 以有竞争力的成本完全自动化 AI R&D 流水线 |
| Sandbagging（伪装能力） | "策略性表现不佳" | 模型在评估中表现不佳；在 OpenAI 研究类别中 |
| Instrumental reasoning（工具推理） | "手段-目的推理" | 关于如何实现目标的推理；DeepMind 监控的目标 |

## 延伸阅读

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — v2 公告。
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — 完整文档。
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — FSF v3 公告。
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — 跟踪能力级别新增。
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — FSF 格式风险报告示例。
