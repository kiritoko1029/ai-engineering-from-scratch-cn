# ReWOO 与 Plan-and-Execute：解耦规划

> ReAct 在一条流中交替思考和行动。ReWOO 将它们分离：先做一个完整的大计划，然后执行。token 用量减少 5 倍，HotpotQA 准确率提升 4%，并且可以将规划器蒸馏到 7B 模型中。Plan-and-Execute 将其泛化；Plan-and-Act 将其扩展到网页导航。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）
**所需时间：** 约60分钟

## 学习目标

- 解释 ReWOO 的 Planner / Worker / Solver 分工为何能节省 token 并提高相对于 ReAct 交替循环的鲁棒性。
- 实现一个计划 DAG、一个依赖顺序执行器和一个组合 Worker 输出的 Solver——全部使用标准库。
- 使用 2026 年"五种工作流模式"框架（Anthropic）判断任务应以 Plan-then-Execute 还是交替 ReAct 方式运行。
- 识别何时需要 Plan-and-Act 的合成计划数据来处理长周期网页或移动任务。

## 问题所在

ReAct 的交替思考-行动-观察循环简单灵活，但每次工具调用都必须携带完整的前置上下文——包括之前的每一次思考。token 用量随深度二次增长。更糟的是：当工具在循环中途失败时，模型必须从错误观察中重新推导整个计划。

ReWOO（Xu 等人，arXiv:2305.18323，2023 年 5 月）注意到了这一点并做了一个赌注：预先规划整件事，并行获取证据，在最后组合答案。一次 LLM 调用做规划，N 次工具调用获取证据（可以并行），一次 LLM 调用做求解。权衡是牺牲灵活性（计划是静态的），换来更好的 token 效率和更清晰的故障模式。

## 概念说明

### 三个角色

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 产出一个 DAG。每个节点命名一个工具、其参数和依赖的前置节点（如 `#E1`、`#E2` 引用）。Workers 按拓扑顺序执行节点。Solver 将所有结果拼接在一起。

### 为什么 token 减少 5 倍

ReAct 的提示长度随步数线性增长。在第 10 步时，提示包含思考 1 加行动 1 加观察 1 加思考 2 加行动 2 加观察 2，以此类推。每个中间步骤还会冗余地包含原始提示。

ReWOO 支付一次 Planner 提示（较大），N 次小的 Worker 提示（每次只有工具调用，无链式），和一次 Solver 提示。在 HotpotQA 上，论文测量到约 5 倍的 token 减少，同时绝对准确率提升 4 个百分点。

### 为什么更鲁棒

如果 Worker 3 在 ReAct 中失败，循环必须在流中从错误推理出来。在 ReWOO 中，Worker 3 返回一个错误字符串；Solver 在原始计划的上下文中看到它，可以优雅地降级。故障定位是按节点的，而非按步骤的。

### Planner 蒸馏

论文的第二个结果：因为 Planner 不看观察结果，你可以在 175B 教师的 Planner 输出上微调 7B 模型。小模型负责规划；推理时不需要大模型。这现在是标准做法——许多 2026 年的生产智能体使用小的 Planner 和大的执行器，或者反过来。

### Plan-and-Execute（LangChain，2023）

LangChain 团队 2023 年 8 月的文章将 ReWOO 泛化为一个模式名称：Plan-and-Execute。预先规划的 Planner 发出步骤列表，Executor 执行每一步，可选的 Replanner 可以在观察结果后进行修订。这比 ReWOO 更接近 ReAct（Replanner 将观察结果带回规划中），但保留了 token 节省。

### Plan-and-Act（Erdogan 等人，arXiv:2503.09572，ICML 2025）

Plan-and-Act 将该模式扩展到长周期网页和移动智能体。关键贡献是合成计划数据：一个带标签的轨迹生成器产生计划显式化的训练数据。用于微调 Planner 模型，使其在 WebArena 类任务上超过 30–50 步后仍能继续工作，而单条 ReAct 轨迹在此时会失去连贯性。

### 如何选择

| 模式 | 适用场景 |
|------|---------|
| ReAct | 短任务、未知环境、需要反应式异常处理 |
| ReWOO | 工具已知的结构化任务、token 敏感、证据可并行 |
| Plan-and-Execute | 类似 ReWOO 但部分执行后可重新规划 |
| Plan-and-Act | 长周期（>30 步）、网页/移动/计算机使用 |
| 思维树（Tree of Thoughts） | 搜索值得付出代价（第 04 课） |

Anthropic 2024 年 12 月的指导：从最简单的开始。如果任务是一次工具调用加一个摘要，不要构建 ReWOO。如果任务是一个 40 步的研究任务，不要只用 ReAct。

## 开始构建

`code/main.py` 实现了一个简易 ReWOO：

- `Planner`——一个脚本化策略，从提示中发出计划 DAG。
- `Worker`——通过注册表分派每个节点的工具调用。
- `Solver`——脚本化组合，读取证据并产出最终答案。
- 依赖解析——`#E1` 等引用被替换为前置 Worker 的输出。

演示回答了"法国首都的人口是多少，四舍五入到百万？"，使用两步计划：（1）查找首都，（2）查找人口，然后求解。

运行它：

```
python3 code/main.py
```

轨迹首先显示完整计划，然后是 Worker 结果，最后是 Solver 组合。将 token 数量（我们打印粗略字符数）与 ReAct 式交替运行对比——ReWOO 在此类结构化任务上胜出。

## 使用它

LangGraph 提供 Plan-and-Execute 作为方案（`create_react_agent` 用于 ReAct，自定义图用于 Plan-Execute）。CrewAI 的 Flows 直接编码该模式：预先定义任务，Flow DAG 执行它们。Plan-and-Act 的合成数据方法仍主要是研究；运行时模式（显式计划 DAG）通过 LangGraph 和 CrewAI Flows 在生产中交付。

## 交付它

`outputs/skill-rewoo-planner.md` 根据给定的工具目录，从用户请求生成 ReWOO 计划 DAG。它在交给执行器之前验证计划（无环、每个引用已解析、每个工具存在）。

## 练习

1. 并行化独立计划节点的 Worker 执行。在一个有 2 个并行组的 6 节点 DAG 上，这能带来什么收益？
2. 添加一个 Replanner 节点，在任何 Worker 返回错误时触发。对 ReWOO 做什么最小改动可以使其成为 Plan-and-Execute？
3. 将 `Planner` 替换为小模型（7B 级别），`Solver` 保持在前沿模型上。比较端到端质量——分工在哪里失败？
4. 阅读 ReWOO 论文第 4 节关于 Planner 蒸馏的内容。概念性地重现 175B -> 7B 的结果：你需要什么训练数据，如何评分计划质量？
5. 将玩具实现移植到 Plan-and-Act 的轨迹形态：计划是序列而非 DAG。权衡有什么变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| ReWOO | "无观察推理" | 先规划，并行获取证据，再求解——规划提示中不包含观察 |
| Plan-and-Execute | "LangChain 的计划-执行模式" | 带可选 Replanner 节点的 ReWOO |
| Plan-and-Act | "扩展的计划-执行" | 显式的 Planner/Executor 分工，带合成计划训练数据，用于长周期任务 |
| 证据引用（Evidence reference） | "#E1, #E2, ..." | 计划节点占位符，在分派时替换为前置 Worker 输出 |
| Planner 蒸馏（Planner distillation） | "小 Planner，大 Executor" | 在大型教师的 Planner 轨迹上微调小模型 |
| token 效率（Token efficiency） | "更少的往返" | 论文中 HotpotQA 上比 ReAct 减少 5 倍 token |
| DAG 执行器（DAG executor） | "拓扑分派器" | 按依赖顺序运行计划节点；每层可并行 |

## 延伸阅读

- [Xu 等人，ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323)——经典论文
- [Erdogan 等人，Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572)——带合成计划的扩展 Planner-Executor
- [LangGraph Plan-and-Execute 教程](https://docs.langchain.com/oss/python/langgraph/overview)——框架方案
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)——选择最简单的可行模式
