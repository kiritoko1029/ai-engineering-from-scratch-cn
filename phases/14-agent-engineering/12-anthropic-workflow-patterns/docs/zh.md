# Anthropic 的工作流模式：简单优于复杂

> Schluntz 和 Zhang（Anthropic，2024 年 12 月）区分了工作流（预定义路径）和智能体（动态工具使用）。五种工作流模式覆盖大多数情况。从直接 API 调用开始。仅当步骤无法预测时才添加智能体。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）
**所需时间：** 约60分钟

## 学习目标

- 说出 Anthropic 的五种工作流模式：提示链、路由、并行化、编排器-工作者、评估器-优化器。
- 解释智能体与工作流的区别以及各自的工程成本。
- 识别何时选择工作流而非智能体（反之亦然）。
- 用标准库对脚本化 LLM 实现全部五种模式。

## 问题所在

团队为想要单个函数调用的问题引入多智能体框架。成本是真实的：框架添加了遮蔽提示、隐藏控制流并诱使过早复杂的层。Schluntz 和 Zhang 2024 年 12 月的文章是最常被引用的行业反驳：从简单开始，仅当复杂性能证明其成本时才添加。

## 概念说明

### 工作流 vs 智能体

- **工作流。** 通过预定义代码路径编排的 LLM 和工具。工程师拥有图。
- **智能体。** LLM 动态指挥自己的工具并采取自己的步骤。模型拥有图。

两者都有其位置。工作流更便宜、更快、更容易调试。智能体解锁开放式问题但使故障模式更难推理。

### 增强型 LLM

所有五种模式的基础：一个 LLM 加三种接入的能力——搜索（检索）、工具（行动）、记忆（持久化）。任何 API 调用都可以使用这些。

### 五种模式

1. **提示链。** 调用 1 的输出是调用 2 的输入。当任务有干净的线性分解时使用。步骤间可选的编程式门控。

2. **路由。** 分类器 LLM 选择调用哪个下游 LLM 或工具。当类别不同的输入需要不同处理时使用（一线支持 vs 退款 vs bug vs 销售）。

3. **并行化。** 并发运行 N 个 LLM 调用，聚合结果。两种形态：分段（不同块）和投票（相同提示，N 次运行，多数/合成）。

4. **编排器-工作者。** 编排器 LLM 动态决定运行哪些工作者（也是 LLM）并合成其输出。类似智能体循环但编排器不会无限循环。

5. **评估器-优化器。** 一个 LLM 提出答案，另一个 LLM 评估它。迭代直到评估器通过。这是 Self-Refine（第 05 课）的泛化。

### 工作流优于智能体的场景

- **可预测的任务。** 如果你能枚举步骤，就应该这样做。
- **成本约束的任务。** 工作流有界的步骤数；智能体可能失控。
- **合规约束的任务。** 审计员想要读取图，而非从轨迹推断。

### 智能体优于工作流的场景

- **开放式研究。** 当下一步取决于上一步返回的内容时。
- **可变长度任务。** 几分钟到几小时的工作，步数未知。
- **新领域。** 当你还不知道正确的工作流时——先探索，后固化。

### 上下文工程配套

"Effective context engineering for AI agents"（Anthropic 2025）形式化了相邻学科：200k 窗口是预算而非容器。包含什么、何时压缩、何时让上下文增长。在第 14 阶段关于上下文压缩的课程中有详细覆盖。

## 开始构建

`code/main.py` 对 `ScriptedLLM` 实现了全部五种工作流模式：

- `prompt_chain(input, steps)`——顺序执行。
- `route(input, classifier, handlers)`——分类 + 分派。
- `parallel_vote(prompt, n, aggregator)`——N 次运行，聚合。
- `orchestrator_workers(task, workers)`——编排器选择工作者。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)`——循环直到通过。

运行它：

```
python3 code/main.py
```

每个模式打印其轨迹。每个模式的总代码行数约 10-15 行；框架的成本以千行计。

## 使用它

- 大多数任务用直接 API 调用。
- 仅当模式真正需要持久状态（LangGraph）、Actor 模型并发（AutoGen v0.4）或角色模板（CrewAI）时才用框架。
- 当你想要 Claude Code 工具链形态但不想重建时使用 Claude Agent SDK。

## 交付它

`outputs/skill-workflow-picker.md` 为给定任务描述选择正确的模式，包括决策理由和工作流不足时到智能体的重构路径。

## 练习

1. 实现带置信度阈值的路由。低于阈值 -> 升级到人工。对于一线支持用例阈值设在哪里？
2. 给 `parallel_vote` 添加超时。当一个调用挂起时会怎样？如何在缺失投票的情况下聚合？
3. 将 `evaluator_optimizer` 变为赌博机：跨迭代保留 top-2 输出，使晚到的好结果不会被晚到的差结果覆盖。
4. 将提示链与路由结合：路由器选择三条链之一。衡量 token 成本与单大提示替代方案的对比。
5. 选择你的一项生产功能。画出工作流图。数步骤。智能体在这里真的更好吗？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 工作流（Workflow） | "预定义流程" | 工程师拥有的 LLM 和工具调用图 |
| 智能体（Agent） | "自主 AI" | 模型拥有的图；动态工具指挥 |
| 增强型 LLM（Augmented LLM） | "带工具的 LLM" | LLM + 搜索 + 工具 + 记忆；原子单元 |
| 提示链（Prompt chaining） | "顺序调用" | 调用 N 的输出是调用 N+1 的输入 |
| 路由（Routing） | "分类器分派" | 选择哪个链/模型处理输入 |
| 并行化（Parallelization） | "扇出" | N 个并发调用；按分段或投票聚合 |
| 编排器-工作者（Orchestrator-workers） | "分派器智能体" | 编排器 LLM 动态选择专家 LLM |
| 评估器-优化器（Evaluator-optimizer） | "提议者 + 裁判" | 迭代直到评估器通过；Self-Refine 泛化 |

## 延伸阅读

- [Anthropic，Building Effective Agents（2024 年 12 月）](https://www.anthropic.com/research/building-effective-agents)——五种工作流模式
- [Anthropic，Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——配套学科
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)——有状态图何时值得其成本
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)——编排器-工作者模式的产品化
