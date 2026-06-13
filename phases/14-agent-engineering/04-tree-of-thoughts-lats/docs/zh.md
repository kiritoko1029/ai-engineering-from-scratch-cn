# 思维树与 LATS：深思熟虑的搜索

> 单条思维链轨迹没有回溯的余地。ToT（Yao 等人，2023）将推理转化为一棵树，每个节点有自评估。LATS（Zhou 等人，2024）在蒙特卡洛树搜索下统一了 ToT、ReAct 和 Reflexion。Game of 24 从 4%（CoT）提升到 74%（ToT）；LATS 在 HumanEval 上达到 92.7% pass@1。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 03（Reflexion）
**所需时间：** 约75分钟

## 学习目标

- 将推理框定为搜索：节点是"思考"，边是"扩展"，值是"前景如何"。
- 用标准库实现一个 ToT 风格的 BFS 树搜索，带自评估评分。
- 扩展为一个简易的 LATS MCTS 循环，包含选择/扩展/模拟/回传。
- 判断何时搜索值得付出 token 倍增的代价（Game of 24、代码生成），何时单条轨迹就够了（简单问答）。

## 问题所在

思维链是一次线性遍历。如果第一步错了，后续每一步都在错误前提上工作。在 Game of 24（用四个数字通过 + - x / 得到 24）上，GPT-4 CoT 只有 4% 的准确率。模型早期选错了子表达式，无法恢复。

推理需要的能力是：提出多个候选方案、评估它们、选择有前景的、在出现死胡同时回溯。这就是搜索。思维树和 LATS 是两种经典表述。

## 概念说明

### 思维树（Yao 等人，NeurIPS 2023）

每个节点是一个连贯的中间步骤（"一个思考"）。每个节点可以扩展为 K 个子思考。LLM 用评分提示自评估每个节点。搜索探索整棵树——BFS、DFS 或 beam search。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自评估是承重的部分。论文展示了三种变体：`sure / likely / impossible` 分类、`1..10` 数值评分和候选间投票。三种方法在 Game of 24 上都大幅超越 CoT（GPT-4 从 4% 到 74%）。

### LATS（Zhou 等人，ICML 2024）

LATS 在 MCTS 下统一了 ToT、ReAct 和 Reflexion。LLM 扮演三个角色：

- **策略（Policy）**：提出候选下一步行动（ReAct 风格）。
- **价值函数（Value function）**：对部分轨迹评分（ToT 风格自评估）。
- **自反思器（Self-reflector）**：失败时写一段自然语言反思（Reflexion 风格），并用它重新种子未来的展开。

环境反馈（观察）混入价值函数，使搜索基于真实的工具结果而非仅仅是模型意见。论文发表时的结果：HumanEval pass@1 92.7%（GPT-4，SOTA），WebShop 平均 75.9（GPT-3.5，接近基于梯度的微调）。

### MCTS 简述

每次迭代四个阶段：

1. **选择（Select）**——从根到叶使用 UCT（树的上置信界）遍历。
2. **扩展（Expand）**——通过策略生成 K 个子节点。
3. **模拟（Simulate）**——从子节点使用策略展开，用价值函数（或环境奖励）评分叶节点。
4. **回传（Backpropagate）**——沿路径向上更新访问次数和价值估计。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一项是利用；第二项是探索。按任务调整 `c`。

### 成本现实

搜索会爆炸式消耗 token。ToT 在 Game of 24 上使用 CoT 的 100–1000 倍 token。LATS 类似。这不是免费的；将搜索保留给：

- 单条轨迹明显不足的任务（Game of 24、复杂代码）。
- 挂钟时间不如正确性重要的任务。
- 有廉价可靠价值函数的任务（代码的单元测试、数学的显式目标）。

如果你的任务只有一个正确答案且评估器很吵，搜索往往使事情更糟——它会找到一个"高分"的错误答案。

### 2026 年定位

大多数生产智能体不运行 LATS。它们运行带工具锚定验证的 ReAct（CRITIC，第 05 课）。搜索出现在专业细分领域：

- 以测试作为价值函数的编码智能体（HumanEval 风格）。
- 探索多条查询路径的深度研究智能体。
- LangGraph 子图内的重规划工作流。

AlphaEvolve（第 11 课）是 2025 年的极端：代码上的进化搜索、机器可检查的适应度、前沿收益（56 年来首次 4x4 矩阵乘法改进）。

## 开始构建

`code/main.py` 实现了：

- 一个风格化的"选择算术运算"任务上的小型 ToT BFS。
- 同一任务上的简易 LATS MCTS 循环（选择/扩展/模拟/回传），带 UCT 选择。
- 一个组合符号分数和自评估分数的价值函数。

运行它：

```
python3 code/main.py
```

轨迹显示 ToT 以 BFS 每节点扩展三个候选，对比 LATS 通过 MCTS 收敛到最佳展开。两种方法都打印了 token 计数。

## 使用它

LangGraph 提供 ToT 风格的探索作为子图模式；LangChain 团队关于 LATS 的博客（2024 年 5 月）是参考教程。LlamaIndex 提供 `TreeOfThoughts` 智能体。对于大多数 2026 年的生产智能体，这个模式存在于 `if task_complexity > threshold: use_search()` 门控之后——参见第 05 课的评估器-优化器模式。

## 交付它

`outputs/skill-search-policy.md` 根据任务形状、预算和评估器保真度，在线性 ReAct、ToT、LATS 和进化搜索之间选择。

## 练习

1. 用 UCT c=0.1 和 c=2.0 运行简易 LATS。轨迹有什么变化？
2. 将价值函数替换为更吵的评分器（添加随机抖动）。MCTS 还能找到最佳叶节点吗？它能容忍的最低信噪比是多少？
3. 实现 beam-search ToT（每层保留 top-k）并与 BFS 比较。在紧张的 token 预算下哪个更好？
4. 阅读 LATS 第 5.1 节。重现 HumanEval 轨迹计数：需要多少次展开才能达到报告的 pass@1？
5. 阅读 LATS 论文关于"LATS 何时帮助较少"的讨论。写一段决策规则，将任务形状映射到搜索策略。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 思维树（Tree of Thoughts） | "分支 CoT" | Yao 等人——带自评估的思考节点树 |
| LATS | "LLM 的 MCTS" | Zhou 等人——在 MCTS 下统一 ToT + ReAct + Reflexion |
| UCT | "上置信界" | 平衡利用（Q）和探索（ln N / n）的选择公式 |
| 价值函数（Value function） | "这个状态有多好" | 提示式 LLM 评分或环境奖励；驱动回传 |
| 策略（Policy） | "行动提议器" | ReAct 风格的生成器；发出候选下一步思考/行动 |
| 展开（Rollout） | "模拟轨迹" | 使用策略从节点走到叶节点，用价值函数评分 |
| 回传（Backpropagate） | "更新祖先" | 将叶节点的奖励沿路径推上去，更新访问次数和 Q |
| 搜索成本（Search cost） | "Token 爆炸" | Game of 24 上 CoT 的 100-1000 倍；采用前做好预算 |

## 延伸阅读

- [Yao 等人，Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601)——经典论文
- [Zhou 等人，LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406)——带 Reflexion 反馈的 MCTS
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)——搜索的子图模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131)——带编程式评估器的进化搜索
