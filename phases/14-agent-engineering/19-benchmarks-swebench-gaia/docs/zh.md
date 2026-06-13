# 基准测试：SWE-bench、GAIA、AgentBench

> 三个基准测试锚定了 2026 年的智能体评估。SWE-bench 测试代码补丁。GAIA 测试通用工具使用。AgentBench 测试多环境推理。了解它们的组成、污染故事和它们不衡量什么。

**类型：** 学习
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 06（工具使用）
**所需时间：** 约60分钟

## 学习目标

- 说出 SWE-bench 的测试工具（FAIL_TO_PASS）并解释它为什么以单元测试为门控。
- 解释 SWE-bench Verified（OpenAI，500 个任务）为何存在及其移除了什么。
- 描述 GAIA 的设计：对人类简单，对 AI 难；三个难度级别。
- 说出 AgentBench 的八个环境及其对开源 LLM 的主要障碍。
- 总结 SWE-bench+ 的污染发现及其影响。

## 问题所在

排行榜告诉你哪个模型在某个基准测试上获胜。它们不告诉你：

- 基准测试是否被污染（训练数据中的解决方案、测试泄露）。
- 基准测试是否衡量你关心的（代码 vs 浏览 vs 通用）。
- 评估器是否鲁棒（AST 匹配、状态检查、人工审查）。

在引用数字之前，了解三个锚定基准测试及其失败模式。

## 概念说明

### SWE-bench（Jimenez 等人，ICLR 2024 oral）

- 来自 12 个流行 Python 仓库的 2,294 个真实 GitHub issue。
- 智能体获得：修复前提交的代码库 + 自然语言 issue 描述。
- 智能体产出：一个补丁。
- 评估器：应用补丁，运行仓库的测试套件。补丁必须翻转 FAIL_TO_PASS 测试（之前失败，现在通过）且不破坏 PASS_TO_PASS 测试。

SWE-agent（Yang 等人，2024）发布时达到 12.5%，通过强调智能体-计算机接口（文件编辑器命令、模型理解的搜索语法）。

### SWE-bench Verified

OpenAI，2024 年 8 月。人工策划的 500 任务子集。移除模糊的 issue、不可靠的测试和修复不明确的任务。"你的智能体是否发布真实补丁？"的主要基准。

### 污染

- 超过 94% 的 SWE-bench issue 早于大多数模型截止日期。
- **SWE-bench+** 发现 32.67% 的成功补丁在 issue 文本中泄露了解决方案（模型在描述中看到了修复），31.08% 因弱测试覆盖而可疑。
- Verified 更干净但非无污染。

实际意义：在 SWE-bench 上得分 50% 的模型可能在 SWE-bench+ 上得分 35%。如果你声称 SWE-bench 性能，始终报告两者。

### GAIA（Mialon 等人，2023 年 11 月）

- 466 个问题；300 个保留在 huggingface.co/gaia-benchmark 的私有排行榜。
- 设计理念："对人类概念上简单（92%）但对 AI 难（GPT-4 带插件：15%）"。
- 测试推理、多模态、网页、工具使用。
- 三个难度级别；级别 3 需要跨模态的长工具链。

GAIA 是你用来衡量"通用能力"的。不要与代码特定的基准测试混淆。

### AgentBench（Liu 等人，ICLR 2024）

- 跨代码（Bash、DB、KG）、游戏（Alfworld、LTP）、网页（WebShop、Mind2Web）和开放式生成的 8 个环境。
- 多轮，每个分片约 4k-13k 轮。
- 主要发现：长期推理、决策和指令跟随是开源 LLM 追赶商业的障碍。

### 这些不衡量什么

- 真实世界的运营成本（token、挂钟时间）。
- 对抗条件下的安全行为。
- 你领域的性能（使用你自己的评估，第 30 课）。
- 尾部失败（基准测试取平均；生产运维关心最差的 1%）。

### 基准测试出错的地方

- **单一数字执念。** SWE-bench 50% 告诉你的少于 P50/P75/P95 成本 + 步骤分布。
- **污染的声明。** 报告 SWE-bench 而不提及 Verified 或 SWE-bench+ 是误导性的。
- **基准测试作为开发目标。** 为基准测试优化偏离了生产有用性。

## 开始构建

`code/main.py` 实现了一个类似 SWE-bench 的简易工具：

- 合成的 bug 修复任务（3 个任务）。
- 一个提出补丁的脚本化"智能体"。
- 检查 FAIL_TO_PASS（bug 现已修复）和 PASS_TO_PASS（没有破坏）的测试运行器。
- 基于问题分解深度的 GAIA 风格难度分类器。

运行它：

```
python3 code/main.py
```

输出展示了每个任务 + 每个难度的解决率，并使评估器规则具体化。

## 使用它

- **SWE-bench Verified** 用于代码智能体。始终报告 Verified 分数。
- **GAIA** 用于通用智能体。使用私有排行榜分片。
- **AgentBench** 用于多环境比较。
- **自定义评估**（第 30 课）用于你产品的实际形状。

## 交付它

`outputs/skill-benchmark-harness.md` 为任何代码库-任务对构建 SWE-bench 风格的工具，带 FAIL_TO_PASS / PASS_TO_PASS 门控。

## 练习

1. 将简易工具移植到真实仓库上运行（选一个你的）。为已知 bug 写 3 个 FAIL_TO_PASS 测试。
2. 添加步数指标。在你的 3 个任务上，每次解决需要多少智能体步骤？
3. 阅读 SWE-bench+ 论文。实现解决方案泄露检查（模式匹配 issue 文本与 diff）。
4. 从公开分片下载一个 GAIA 问题。追踪 GPT-4 级智能体会做什么。它需要什么工具？
5. 阅读 AgentBench 的每环境分解。哪个环境映射到你的产品表面？"SOTA"在那里是什么样的？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| SWE-bench | "代码智能体基准" | 2,294 个 GitHub issue；补丁必须翻转 FAIL_TO_PASS 测试 |
| SWE-bench Verified | "干净的 SWE-bench" | 500 个人工策划的任务，OpenAI |
| FAIL_TO_PASS | "修复门控" | 之前失败且补丁后必须通过的测试 |
| PASS_TO_PASS | "无回归门控" | 之前通过且必须仍然通过的测试 |
| GAIA | "通用基准" | 466 个人类简单/AI 难的多工具问题 |
| AgentBench | "多环境基准" | 8 个环境；长周期多轮 |
| 污染（Contamination） | "训练集泄露" | 模型训练中存在的基准任务 |
| SWE-bench+ | "污染审计" | 在成功的 SWE-bench 补丁中发现 32.67% 的解决方案泄露 |

## 延伸阅读

- [Jimenez 等人，SWE-bench (arXiv:2310.06770)](https://arxiv.org/abs/2310.06770)——原始基准
- [OpenAI，SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)——策划子集
- [Mialon 等人，GAIA (arXiv:2311.12983)](https://arxiv.org/abs/2311.12983)——通用基准
- [Liu 等人，AgentBench (arXiv:2308.03688)](https://arxiv.org/abs/2308.03688)——多环境套件
