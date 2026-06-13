# LLM 功能的 A/B 测试 — GrowthBook、Statsig 与「感觉」问题

> 传统 A/B 测试并非为非确定性的 LLM 设计。关键区别在于：评估回答的是「模型能不能完成任务？」A/B 测试回答的是「用户在不在乎？」两者缺一不可；凭感觉上线的时代已经结束了。2026 年需要测试的方向：提示词工程（措辞）、模型选择（GPT-4 vs GPT-3.5 vs 开源模型；准确率 vs 成本 vs 延迟）、生成参数（temperature、top-p）。实际案例：聊天机器人的奖励模型变体带来了对话长度 +70%、留存率 +30% 的提升；Nextdoor 的 AI 主题行实验在奖励函数优化后实现了 +1% 的点击率提升；Khan Academy 的 Khanmigo 在延迟与数学准确性之间反复迭代权衡。平台对比：**Statsig**（2025 年 9 月被 OpenAI 以 11 亿美元收购）— 序贯检验、CUPED、一站式解决方案。**GrowthBook** — 开源、仓库原生、贝叶斯 + 频率派 + 序贯引擎、CUPED、SRM 检测、Benjamini-Hochberg + Bonferroni 校正。选择依据取决于你对仓库 SQL 的偏好以及「被 OpenAI 收购」这件事对你所在组织是否重要。

**类型：** 学习
**语言：** Python（标准库、简易序贯检验模拟器）
**前置要求：** Phase 17 · 13（可观测性）、Phase 17 · 20（渐进式部署）
**所需时间：** 约 60 分钟

## 学习目标

- 区分评估（「模型能不能完成任务」）与 A/B 测试（「用户在不在乎」）。
- 列举三个可测试的维度（提示词、模型、参数）并为每个维度选择合适的指标。
- 解释 CUPED、序贯检验和 Benjamini-Hochberg 多重比较校正。
- 根据仓库 SQL 偏好和企业收购立场选择 Statsig 或 GrowthBook。

## 问题所在

你手动调整了一个系统提示词，感觉效果更好了，于是直接上线。转化率的变化在噪声范围内。你要么归咎于指标本身，要么上线了新模型但转化率纹丝不动——到底是模型变差了，还是变化太小无法检测？你无从得知，因为你没有做 A/B 测试就上线了。

评估回答的是模型能否在标注数据集上完成任务，但无法回答用户是否更偏好新版本的输出。只有受控的在线实验才能回答这个问题，而且前提是实验具有足够的统计功效、控制了非确定性因素，并对多重比较做了校正。

## 概念说明

### 评估 vs A/B 测试

**评估** — 离线，标注数据集，评判者（评分标准或 LLM 评判或人工）。回答：「在这个固定分布上，输出是否正确/有用/安全？」

**A/B 测试** — 在线，真实用户，随机分组。回答：「新变体是否提升了用户层面的关键指标？」

两者缺一不可。评估在用户接触前捕获回归问题；A/B 测试在之后确认产品影响。

### 测试什么

1. **提示词工程** — 措辞、系统提示词结构、示例。指标：任务成功率、用户留存率、单次请求成本。
2. **模型选择** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS。指标：准确率（任务）+ 单次请求成本 + P99 延迟。多目标优化。
3. **生成参数** — temperature、top-p、max_tokens。指标：任务特定指标（输出多样性 vs 确定性）。

### CUPED — 方差缩减

利用实验前数据的受控实验。在比较实验后数据之前，回归去除实验前的方差。典型方差缩减幅度：30-70%。等效样本量免费提升。

实现方式：Statsig 和 GrowthBook 均已实现。

### 序贯检验

经典 A/B 测试假设固定样本量。序贯检验（「偷看然后决策」）在反复查看数据的情况下控制假阳性率。始终有效的序贯程序（mSPRT、Howard 置信序列）允许你在结果明确时提前停止。

### 多重比较校正

以 95% 置信度运行 20 个 A/B 测试，平均会产生一个假阳性。Bonferroni 校正收紧每个检验的 α 值；Benjamini-Hochberg 控制错误发现率。GrowthBook 两种都实现了。

### SRM — 样本比例失衡

分配哈希将用户随机分配到各变体。如果 50/50 的分配实际变成了 47/53，说明出了问题——SRM 检测会标记这种情况。两个平台都已实现。

### Statsig vs GrowthBook

**Statsig**：
- 2025 年 9 月被 OpenAI 以 11 亿美元收购。托管式 SaaS。
- 序贯检验、CUPED、留出人群。
- 一站式：功能标志 + 实验平台 + 可观测性。
- 最适合：团队本来就想用捆绑产品，不介意 OpenAI 所有权。

**GrowthBook**：
- 开源（MIT）；仓库原生（直接从 Snowflake/BigQuery/Redshift 读取数据）。
- 多引擎：贝叶斯、频率派、序贯。
- CUPED、SRM、Bonferroni、BH 校正。
- 自托管或托管云。
- 最适合：仓库 SQL 团队，数据团队控制指标层，偏好开源。

### 非确定性使功效计算复杂化

相同的提示词会产生不同的输出。传统功效计算假设观测值独立同分布（IID）。由于 LLM 的非确定性，有效样本量低于名义样本量。将所需样本量乘以约 1.3-1.5 倍作为安全余量。

### 实际案例结果

- 聊天机器人奖励模型变体：对话长度 +70%，留存率 +30%。
- Nextdoor 主题行：奖励函数优化后点击率 +1%。
- Khan Academy Khanmigo：延迟与数学准确性的迭代权衡。

### 反模式：凭感觉上线

每位资深工程师都能说出一个「感觉更好」就上线、没做 A/B 测试的功能。其中大多数都在团队没有注意到的情况下让产品指标退化了数月。A/B 测试就是那个强制机制。

### 需要记住的数字

- Statsig 被 OpenAI 收购：11 亿美元，2025 年 9 月。
- GrowthBook：MIT 开源；贝叶斯 + 频率派 + 序贯。
- CUPED 方差缩减：30-70%。
- LLM 非确定性 → +30-50% 样本量缓冲。

## 开始使用

`code/main.py` 模拟了一个具有固定边界的序贯 A/B 测试。展示了序贯检验如何让你提前停止。

## 部署产出

本课将生成 `outputs/skill-ab-plan.md`。给定功能变更、工作负载、基线，自动选择平台、门控策略和样本量。

## 练习

1. 运行 `code/main.py`。对于期望 5% 提升、基线 3% 转化率的情况，80% 功效需要多少样本量？
2. 为一个受医疗监管的本地部署客户选择 Statsig 或 GrowthBook。
3. 设计一个 A/B 测试，比较 GPT-4 和 GPT-3.5 在「每张解决工单成本」上的表现。主要指标、护栏指标、次要指标分别是什么？
4. 你的金丝雀发布通过了，但 A/B 测试显示转化率下降了 -1.2%。你还上线吗？写出升级标准。
5. 对一个实验前数据占实验后 60% 方差的场景应用 CUPED。计算等效样本量提升幅度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Eval | 「离线测试」 | 基于标注数据集的模型能力评估 |
| A/B test | 「实验」 | 基于真实用户的在线随机比较 |
| CUPED | 「方差缩减」 | 利用实验前数据回归以减少方差 |
| Sequential test | 「偷看也没关系的测试」 | 允许提前停止的始终有效程序 |
| Multiple comparison | 「族错误」 | 运行多个检验会膨胀假阳性率 |
| Bonferroni | 「严格校正」 | 将 α 除以检验次数 |
| Benjamini-Hochberg | 「BH FDR」 | 错误发现率控制，保守程度较低 |
| SRM | 「分配比例不对」 | 样本比例失衡；分配 bug |
| Statsig | 「OpenAI 旗下的」 | 商业一站式平台，2025 年被收购 |
| GrowthBook | 「开源那个」 | MIT 仓库原生平台 |
| mSPRT | 「序贯概率比检验」 | 经典序贯检验程序 |

## 延伸阅读

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
