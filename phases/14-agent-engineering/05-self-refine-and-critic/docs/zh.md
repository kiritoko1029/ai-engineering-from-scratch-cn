# Self-Refine 与 CRITIC：迭代式输出改进

> Self-Refine（Madaan 等人，2023）用一个 LLM 扮演三个角色——生成、反馈、优化——在循环中运行。7 个任务平均提升 20 个百分点。CRITIC（Gou 等人，2023）通过将验证路由到外部工具来加固反馈步骤。2026 年这个模式以"评估器-优化器"（Anthropic）或防护栏循环（OpenAI Agents SDK）的形式出现在每个框架中。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 03（Reflexion）
**所需时间：** 约60分钟

## 学习目标

- 说出 Self-Refine 的三个提示（生成、反馈、优化）并解释为什么历史记录对优化提示很重要。
- 解释 CRITIC 的关键洞见：LLM 在没有外部锚定时不可靠于自我验证。
- 用标准库实现一个带历史记录和可选外部验证器的 Self-Refine 循环。
- 将此模式映射到 Anthropic 的"评估器-优化器"工作流和 OpenAI Agents SDK 的输出防护栏。

## 问题所在

一个智能体产出了几乎正确的答案。也许一行代码有语法错误。也许摘要太长了。也许计划遗漏了一个边界情况。你想要的是：智能体批判自己的输出，然后修复它。

Self-Refine 表明用单个模型、无训练数据、无强化学习就能做到。但有一个问题：LLM 在硬事实上不善于自我验证。CRITIC 提出了修复方案——将验证步骤路由到外部工具（搜索引擎、代码解释器、计算器、测试运行器）。

这两篇论文共同定义了 2026 年迭代改进的默认做法：生成、验证（尽可能外部化）、优化、验证通过时停止。

## 概念说明

### Self-Refine（Madaan 等人，NeurIPS 2023）

一个 LLM，三个角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

关键细节：`refine` 看到完整历史——所有先前的输出和批判——所以不会重复错误。论文对此做了消融实验：去掉历史记录，质量急剧下降。

亮点：跨 7 个任务（数学、代码、缩写、对话）包括 GPT-4 在内，平均绝对提升 20 个百分点。无训练、无外部工具、单模型。

### CRITIC（Gou 等人，arXiv:2305.11738，v4 2024 年 2 月）

Self-Refine 的弱点：反馈步骤是 LLM 给自己打分。对于事实性声明这是不可靠的（幻觉对产出它的模型来说往往看起来很有说服力）。CRITIC 用 `verify(task, output, tools)` 替换了 `feedback(task, output)`，其中 `tools` 包括：

- 用于事实性声明的搜索引擎。
- 用于代码正确性的代码解释器。
- 用于算术的计算器。
- 领域特定的验证器（单元测试、类型检查器、代码检查器）。

验证器产出基于工具结果的结构化批判。优化器然后以此批判为条件进行优化。

亮点：CRITIC 在事实性任务上优于 Self-Refine，因为批判是有根据的。在没有外部验证器的任务（创意写作、格式化）上，CRITIC 退化为 Self-Refine。

### 停止条件

两种常见形态：

1. **验证器通过。** 外部测试返回成功。有条件时优先使用（单元测试、类型检查器、防护栏断言）。
2. **未发出反馈。** 模型说"输出没问题"。更便宜但不可靠；配合最大迭代次数上限使用。

2026 年默认做法：组合使用。"如果验证器通过则停止，或模型说没问题且迭代 >= 2，或迭代 >= 最大迭代次数。"

### 评估器-优化器（Anthropic，2024）

Anthropic 2024 年 12 月的文章将其命名为五种工作流模式之一。两个角色：

- 评估器（Evaluator）：评分输出并产出批判。
- 优化器（Optimizer）：根据批判修订输出。

循环直到评估器通过。这就是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 补充的关键工程细节：评估器和优化器的提示应该有显著差异，这样模型才不会只是橡皮图章。

### OpenAI Agents SDK 输出防护栏

OpenAI Agents SDK 以"输出防护栏"的形式提供此模式。防护栏是在智能体产出最终输出后运行的验证器。如果防护栏触发（抛出 `OutputGuardrailTripwireTriggered`），输出被拒绝，智能体可以重试。防护栏可以调用工具（CRITIC 风格）或纯函数（Self-Refine 风格）。

### 2026 陷阱

- **橡皮图章循环。** 同一模型用相同提示风格做生成和批判，会收敛到"看起来没问题"。使用结构不同的提示，或用更小更便宜的模型做批判。
- **过度优化。** 每次优化轮次增加延迟和 token。预算 1-3 轮；之后升级到人工审核。
- **CRITIC 用于琐碎任务。** 如果没有外部验证器，CRITIC 退化为 Self-Refine；不要为存根验证器支付延迟。

## 开始构建

`code/main.py` 在一个简易任务上实现了 Self-Refine 和 CRITIC：给定主题产出一个简短的要点列表。验证器检查格式（3 个要点，每个不超过 60 个字符）。CRITIC 添加了一个外部"事实验证器"来惩罚已知的幻觉。

组件：

- `generate`——脚本化生成器。
- `feedback`——LLM 风格的自我批判。
- `verify_external`——CRITIC 风格的锚定验证器。
- `refine`——根据历史重写输出。
- 停止条件——验证器通过或最多 4 次迭代。

运行它：

```
python3 code/main.py
```

比较 Self-Refine 和 CRITIC 的运行结果。CRITIC 捕获了 Self-Refine 遗漏的事实错误，因为外部验证器拥有自我批判所没有的锚定。

## 使用它

Anthropic 的评估器-优化器就是这个模式的 Claude 友好语言表述。OpenAI Agents SDK 的输出防护栏是 CRITIC 形态的（防护栏可以调用工具）。LangGraph 提供了一个读起来像 Self-Refine 的反思节点。Google 的 Gemini 2.5 Computer Use 添加了一个逐步安全评估器，是 CRITIC 的变体：每个行动在提交前都被验证。

## 交付它

`outputs/skill-refine-loop.md` 根据任务形状、验证器可用性和迭代预算配置一个评估器-优化器循环。为生成器、评估器/验证器和优化器发出提示，加上停止策略。

## 练习

1. 用 max_iterations=1 运行简易实现。CRITIC 还有帮助吗？
2. 用一个带噪声的验证器（随机 30% 误报）替换外部验证器。循环会怎样？这是 2026 年大多数防护栏栈的现实。
3. 实现"生成器和批判者使用不同模型"的变体：大模型生成，小模型批判。能打败同模型吗？
4. 阅读 CRITIC 第 3 节（arXiv:2305.11738 v4）。说出三类验证工具并各举一例。
5. 将 OpenAI Agents SDK 的 `output_guardrails` 映射到 CRITIC 的验证器角色。SDK 哪里做对了，哪里做错了？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Self-Refine | "自我修复的 LLM" | 单模型中的生成 -> 反馈 -> 优化循环，带历史记录 |
| CRITIC | "工具锚定验证" | 用外部验证器（搜索、代码、计算、测试）替换反馈 |
| 评估器-优化器（Evaluator-Optimizer） | "Anthropic 工作流模式" | 两个角色——评估器评分，优化器修订——循环至收敛 |
| 输出防护栏（Output guardrail） | "事后检查" | OpenAI Agents SDK 在智能体产出输出后运行的验证器 |
| 验证步骤（Verify step） | "批判阶段" | 承重的决策：锚定还是自评 |
| 优化历史（Refine history） | "模型已尝试的内容" | 先前输出 + 批判添加到优化提示中；去掉后质量崩溃 |
| 橡皮图章循环（Rubber-stamp loop） | "自我同意失败" | 相同提示的批判返回"看起来没问题"；用结构不同的提示修复 |
| 停止条件（Stop condition） | "收敛测试" | 验证器通过或无反馈且迭代上限；切勿单条件 |

## 延伸阅读

- [Madaan 等人，Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)——经典论文
- [Gou 等人，CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738)——工具锚定验证
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——评估器-优化器工作流模式
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/)——输出防护栏作为 CRITIC 形态的验证器
