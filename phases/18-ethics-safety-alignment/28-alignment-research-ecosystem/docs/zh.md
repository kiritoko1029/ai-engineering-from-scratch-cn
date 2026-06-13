# 对齐研究生态系统 —— MATS、Redwood、Apollo、METR

> 五个组织定义了 2026 年非实验室对齐研究层。MATS（ML Alignment & Theory Scholars）：自 2021 年底以来培养 527+ 名研究员，180+ 篇论文，10K+ 引用，h-index 47；2024 年夏季队列以 501(c)(3) 注册，约 90 名学者和 40 名导师；80% 的 2025 年前校友从事安全/安保工作，200+ 人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo。Redwood Research：由 Buck Shlegeris 创立的应用对齐实验室；引入了 AI Control（第 10 课）；与英国 AISI 合作进行控制安全论证。Apollo Research：为前沿实验室进行部署前阴谋评估；撰写了 In-Context Scheming（第 8 课）和 Towards Safety Cases for AI Scheming。METR（Model Evaluation and Threat Research）：基于任务的能力评估、自主任务时间范围研究；"Common Elements of Frontier AI Safety Policies"比较了实验室框架。Eleos AI Research：模型福利部署前评估（第 19 课）；进行了 Claude Opus 4 福利评估。

**类型：** 学习
**语言：** 无
**前置要求：** 第 18 阶段 · 01-27（第 18 阶段之前的课程）
**所需时间：** 约45分钟

## 学习目标

- 识别非实验室对齐研究生态系统的五个组织及其核心产出。
- 描述 MATS 的规模（学者、论文、h-index）及其作为人才管道的角色。
- 描述 Redwood 的 AI Control 议程及其与英国 AISI 的合作。
- 描述 METR 基于任务的评估方法论。

## 问题所在

前沿实验室（第 18 课）在内部产生安全评估并发布选定结果。实验室外部的生态系统是评估被验证的地方、新的失败模式首次被发现的地方、以及人才被培养的地方。了解生态系统有助于解读哪些研究发现被谁信任。

## 概念说明

### MATS（ML Alignment & Theory Scholars）

始于 2021 年底。研究导师计划；学者与高级研究员一起花 10-12 周研究特定的对齐问题。

规模（2026 年）：
- 自成立以来 527+ 名研究员。
- 180+ 篇发表论文。
- 10K+ 引用。
- h-index 47。
- 2024 年夏季：90 名学者 + 40 名导师；以 501(c)(3) 注册。

职业成果：约 80% 的 2025 年前校友从事安全/安保工作。200+ 人在 Anthropic、DeepMind、OpenAI、英国 AISI、RAND、Redwood、METR、Apollo。

### Redwood Research

应用对齐实验室。由 Buck Shlegeris 创立。引入了 AI Control 议程（第 10 课）。与英国 AISI 合作进行控制安全论证。为 DeepMind 和 Anthropic 的评估设计提供建议。

经典论文：Greenblatt, Shlegeris et al., "AI Control"（arXiv:2312.06942, ICML 2024）；Alignment Faking（Greenblatt, Denison, Wright et al., arXiv:2412.14093，与 Anthropic 联合）。

风格：具体的威胁模型、最坏情况的对手、可压力测试的具体协议。

### Apollo Research

为前沿实验室进行部署前阴谋评估。撰写了 In-Context Scheming（第 8 课，arXiv:2412.04984）。2025 年 OpenAI 反阴谋训练合作的伙伴。产出 Towards Safety Cases for AI Scheming（2024 年）。

风格：在智能体设置中评估欺骗可能出现的场景；三支柱分解（不对齐、目标导向、情境感知）。

### METR（Model Evaluation and Threat Research）

基于任务的能力评估。自主任务完成时间范围研究。"Common Elements of Frontier AI Safety Policies"（metr.org/common-elements，2025 年）比较了实验室框架。

与 Apollo 联合撰写了 AI Scheming 安全论证草稿。

风格：长时间范围任务评估、经验能力测量、框架综合。

### Eleos AI Research

模型福利部署前评估。进行了系统卡片第 5.3 节记录的 Claude Opus 4 福利评估。为第 19 课的福利相关声明提供外部方法论检查。

### 人才流动

MATS 培养研究员。毕业生成为 Anthropic、DeepMind、OpenAI（实验室安全团队）或 Redwood、Apollo、METR、Eleos（外部评估）的成员。外部评估者与实验室以及英国 AISI / CAISI 合作。出版物将生态系统反馈给 MATS 用于下一届学员。

### 为什么这一层重要

单一来源的评估不可靠：实验室评估自己的模型存在结构性利益冲突。外部评估者可以提出和验证实验室可能低报的失败模式。2024 年 Sleeper Agents 论文（第 7 课）是 Anthropic + Redwood；Alignment Faking 是 Anthropic + Redwood；In-Context Scheming 是 Apollo；Anti-Scheming 是 Apollo + OpenAI。多组织结构就是质量控制。

### 本课在第 18 阶段中的位置

第 7-11 课引用了 Redwood 和 Apollo 的工作；第 18 课引用了 METR 的框架比较；第 19 课引用了 Eleos。第 28 课是第 18 阶段其余课程所依赖的生态系统的显式组织地图。

## 开始构建

没有代码。阅读 METR 的"Common Elements of Frontier AI Safety Policies"作为外部综合如何为实验室内部政策工作增加价值的示例。

## 交付成果

本课生成 `outputs/skill-ecosystem-map.md`。给定一个对齐声明或评估，它会识别组织、发表场所和方法论风格，并与已知的对应组织进行交叉检查。

## 练习

1. 从第 7-15 课中选择一篇论文，识别涉及的组织。将作者与 MATS 校友和当前生态系统隶属关系进行交叉检查。

2. 阅读 METR 的"Common Elements of Frontier AI Safety Policies"。找出他们强调的三个跨实验室趋同点和两个最大的分歧点。

3. MATS 的职业成果约 80% 是安全/安保。论证这种选择压力是适应性的（培养领域）还是有偏的（过滤掉非正统立场）。

4. Redwood 和 Apollo 都做控制/阴谋工作但风格不同。选择一个失败模式，描述每个组织会如何调查它。

5. Eleos AI 是唯一的纯模型福利组织。设计一个假设的第二个组织，专注于不同的福利相关问题（认知自由、机器人具身等），并阐明其方法论。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| MATS | "那个导师计划" | ML Alignment & Theory Scholars；自 2021 年以来 527+ 名研究员 |
| Redwood Research | "那个控制实验室" | 应用对齐；AI Control 作者；英国 AISI 合作伙伴 |
| Apollo Research | "那个阴谋评估" | 为前沿实验室进行部署前阴谋评估 |
| METR | "那个任务范围评估" | 基于任务的能力评估；框架综合 |
| Eleos AI | "那个福利实验室" | 模型福利部署前评估 |
| 人才管道 | "MATS -> 实验室" | MATS 毕业生流向 Anthropic、DM、OpenAI、Redwood、Apollo、METR |
| 外部评估 | "非实验室检查" | 非模型生产者进行的评估；增加可信度 |

## 延伸阅读

- [MATS（ML Alignment & Theory Scholars）](https://www.matsprogram.org/) —— 导师计划
- [Redwood Research](https://www.redwoodresearch.org/) —— AI Control 论文
- [Apollo Research](https://www.apolloresearch.ai/) —— 阴谋评估
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) —— 框架比较
- [Eleos AI Research](https://www.eleosai.org/research) —— 模型福利方法论
