# 前沿安全框架 —— RSP、PF、FSF

> 三大实验室的框架定义了 2026 年前沿能力的行业治理格局。Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）引入了分级的 AI 安全等级（ASL-1 到 ASL-5+），仿照生物安全等级设计，其中 ASL-3 于 2025 年 5 月针对 CBRN 相关模型激活。OpenAI Preparedness Framework v2（2025 年 4 月）为被追踪的能力定义了五项标准，并将能力报告与保障措施报告分离。DeepMind Frontier Safety Framework v3.0（2025 年 9 月）引入了关键能力等级（CCL），包括新增的有害操纵 CCL。三个框架现在都包含竞争对手调整条款，允许在同行实验室发布不具可比保障措施的产品时降低要求。跨实验室的对齐仍然是结构性的，而非术语性的："Capability Thresholds"、"High Capability thresholds"和"Critical Capability Levels"指的是类似的概念。

**类型：** 学习
**语言：** 无
**前置要求：** 第 18 阶段 · 17（WMDP），第 18 阶段 · 07-09（欺骗性失败）
**所需时间：** 约75分钟

## 学习目标

- 描述 Anthropic 的 ASL 层级结构以及 ASL-3 激活的原因。
- 列举 OpenAI Preparedness Framework v2 为被追踪能力定义的五项标准。
- 描述 DeepMind 的关键能力等级结构和有害操纵 CCL。
- 解释竞争对手调整条款及其对竞赛动态的影响。
- 定义安全论证并描述其三支柱结构（监控、不可读性、无能力）。

## 问题所在

第 7-17 课确立了欺骗是可能的、双重用途能力存在、评估有局限性。拥有前沿能力模型的实验室需要一个内部治理结构，它必须：
- 定义何时需要新保障措施的阈值。
- 定义扩展前的必要评估。
- 描述安全论证的样貌。
- 处理竞赛动态问题（如果竞争对手在没有保障措施的情况下发布产品，你该怎么办？）。

2025-2026 年的三大框架代表了当前的最佳实践——不完美、不断演进，且跨实验室的对齐程度足以使治理问题从"是否存在框架"转变为"框架是否足够"。

## 概念说明

### Anthropic Responsible Scaling Policy v3.0（2026 年 2 月）

ASL 层级结构：
- ASL-1：非前沿模型（被弱于前沿的基线覆盖）。
- ASL-2：当前前沿基线；以常规保障措施部署。
- ASL-3：灾难性滥用风险显著提高；CBRN 相关能力。2025 年 5 月激活。
- ASL-4：AI R&D-2 交叉阈值；能够自动化入门级 AI 研究的模型。
- ASL-5+：高级 AI R&D；能够显著加速有效扩展的模型。

v3.0 新增内容：
- 前沿安全路线图（以删减版公开发布）。
- 风险报告（季度发布，部分经过外部审查）。
- AI R&D 被细分为 AI R&D-2 和 AI R&D-4。
- 一旦跨越 AI R&D-4，需要提供积极的安全论证，识别模型追求不对齐目标所带来的不对齐风险。

### OpenAI Preparedness Framework v2（2025 年 4 月 15 日）

被追踪能力的五项标准：
- **合理可信。** 存在合理的威胁模型。
- **可测量。** 可以进行经验评估。
- **严重性。** 伤害规模大。
- **全新风险。** 不是已有风险的放大。
- **瞬时或不可逆。** 伤害发生迅速或不可撤销。

满足所有五项标准的能力将被追踪。其他则不被追踪。

PF v2 的其他结构：
- 将能力报告（模型能做什么）与保障措施报告（有哪些控制措施）分离。
- 安全咨询小组审查。
- 领导层批准；董事会安全委员会监督。
- "调整条款"：如果其他实验室在没有可比保障措施的情况下发布产品，OpenAI 可以降低要求。

### DeepMind Frontier Safety Framework v3.0（2025 年 9 月）

按领域划分的关键能力等级（CCL）：
- 生物武器提升
- 网络提升
- ML R&D 加速
- 有害操纵（v3.0 新增）：可能在高风险场景中显著改变信念/行为的模型。

v2.0（2025 年 2 月）增加了欺骗性对齐章节和 ML R&D CCL 的更高安全等级。

### 跨实验室对齐

- Anthropic 称之为"Capability Thresholds"。
- DeepMind 称之为"Critical Capability Levels"。
- OpenAI 称之为"High Capability thresholds"。

没有行业标准术语。结构上是对齐的：三层前沿能力，配有公开的评估标准。三个框架都包含竞争对手调整条款（2025 年起）。

### 安全论证

安全论证是在最坏情况假设下，证明部署可接受安全的书面论证。标准结构针对三个支柱：

- **监控。** 如果发生不良行为，我们能否检测到？
- **不可读性。** 模型是否缺乏执行连贯有害计划的能力？
- **无能力。** 模型是否缺乏造成相关伤害的能力？

不同的安全论证针对不同的支柱。对于 ASL-3 CBRN 案例，无能力（通过遗忘）是主要目标。对于欺骗性对齐，监控和不可读性是目标。对于网络提升，三者都相关。

### 竞赛动态问题

竞争对手调整条款颇具争议。批评者认为它会引发逐底竞争：如果三家实验室都会在竞争对手偏离时降低要求，均衡就会转向偏离。辩护者认为，替代方案（单方面保障措施）在偏离实验室安全意识更差的情况下会产生更糟的结果。

英国 AISI、美国 CAISI 和欧盟 AI 办公室（第 24 课）是外部治理的对应方。实验室框架是自愿性的；监管框架正在形成中。

### 本课在第 18 阶段中的位置

第 17-18 课是在欺骗和红队分析之上的测量与治理层。第 19-24 课涵盖福利、偏见、隐私、水印和监管结构。第 28 课映射了支撑这些评估的研究生态系统（MATS、Redwood、Apollo、METR）。

## 开始构建

本课没有代码。阅读三个主要来源：RSP v3.0、PF v2、FSF v3.0。将每个实验室的层级结构与其他实验室进行映射，并找出每个实验室定义的、其他实验室未定义的一个阈值。

## 交付成果

本课生成 `outputs/skill-framework-diff.md`。给定一个安全框架或发布说明，它会将该框架的阈值定义、所需评估和安全论证结构与 RSP v3.0、PF v2、FSF v3.0 进行比较，并标记跨实验室的差异。

## 练习

1. 阅读 RSP v3.0、PF v2 和 FSF v3.0。编制一个表格，列出每个实验室的 CBRN 阈值、AI R&D 阈值和部署前必需的评估。

2. 竞争对手调整条款存在于所有三个框架中（2025 年起）。写一段话支持它；再写一段话反对它。指出每种立场依赖的假设。

3. 为一个跨越 Anthropic AI R&D-4 阈值的模型设计安全论证。指出三个支柱（监控、不可读性、无能力）各自需要的证据。

4. DeepMind 的 FSF v3.0 引入了有害操纵 CCL。提出三项经验测量指标，用于判断模型是否已跨越该阈值。

5. 阅读 METR 的"Common Elements of Frontier AI Safety Policies"（2025 年）。指出三个最强的跨实验室趋同点和两个最大的分歧点。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| RSP | "Anthropic 的框架" | Responsible Scaling Policy；ASL 层级；v3.0 于 2026 年 2 月发布 |
| PF | "OpenAI 的框架" | Preparedness Framework；五项标准；v2 于 2025 年 4 月发布 |
| FSF | "DeepMind 的框架" | Frontier Safety Framework；CCL；v3.0 于 2025 年 9 月发布 |
| ASL-3 | "生物安全等级 3 的类比" | Anthropic 针对 CBRN 相关能力的层级；2025 年 5 月激活 |
| CCL | "关键能力等级" | DeepMind 的阈值概念；按领域划分 |
| 安全论证 | "正式论证" | 在最坏情况假设下证明部署可接受安全的书面论证 |
| 调整条款 | "竞争对手偏离允许" | 框架中关于在竞争对手无可比保障措施时降低要求的规定 |

## 延伸阅读

- [Anthropic — Responsible Scaling Policy v3.0（2026 年 2 月）](https://www.anthropic.com/responsible-scaling-policy) —— ASL 层级、路线图、AI R&D 细分
- [OpenAI — Updating the Preparedness Framework（2025 年 4 月 15 日）](https://openai.com/index/updating-our-preparedness-framework/) —— 五项标准、调整条款
- [DeepMind — Strengthening our Frontier Safety Framework（2025 年 9 月）](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) —— CCL v3.0、有害操纵
- [METR — Common Elements of Frontier AI Safety Policies（2025 年）](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) —— 跨实验室比较
