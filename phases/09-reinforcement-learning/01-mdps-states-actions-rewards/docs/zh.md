# MDP、状态、动作与奖励

> 马尔可夫决策过程由五部分组成：状态、动作、转移概率、奖励以及折扣因子。强化学习中的所有算法——Q学习、PPO、DPO、GRPO——都是针对这一结构进行优化的。掌握它之后，即可免费阅读强化学习的其余内容。

**类型：** 学习
**语言：** Python
**先修知识：** 第1阶段 · 06（概率与分布）、第2阶段 · 01（机器学习分类）
**时长：** 约45分钟

## 问题所在

你正在编写一个国际象棋机器人、库存规划系统、交易代理，或是用于训练推理模型的 PPO 循环。尽管属于四个完全不同的领域，但有一个令人惊讶的事实：它们最终都可以归结为同一个数学对象。

监督学习会提供 `(x, y)` 对数据，并要求你拟合一个函数。而强化学习则不提供任何标签——仅有状态流、你所采取的动作以及一个标量奖励值。这一步是否赢得了比赛？补货决策是否节省了成本？这笔交易是否盈利？大语言模型刚刚生成的文本是否为评估者带来了更高的奖励？

在将这些数据结构化之前，你是无法从中学习的。“我看到了什么”、“我做了什么”、“接下来发生了什么”、“效果如何”——所有这些都需要转化为可以进行分析的对象。这种结构化形式就是马尔可夫决策过程。此阶段的所有强化学习算法，包括最后的 RLHF 和 GRPO 循环，都是针对这一结构进行优化的。

## 概念概述

![马尔可夫决策过程：状态、动作、转移、奖励与折扣因子](../assets/mdp.svg)

**五个核心要素。**

- **状态** `S`。智能体做出决策所需了解的所有信息。在 GridWorld 中为网格单元；在国际象棋中为棋盘；在大型语言模型中则为上下文窗口及所有记忆内容。
- **动作** `A`。可供选择的操作。如上下左右移动、执行某一步棋或输出一个标记。
- **转移概率** `P(s' | s, a)`。给定状态 `s` 和动作 `a` 时，下一个状态的分布情况。在国际象棋中为确定性过程，在库存管理中为随机过程，在大型语言模型的解码过程中则为近似确定性过程。
- **奖励** `R(s, a, s')`。用于量化表现的标量信号。获胜时为 +1，失败时为 -1；也可表示收入减去成本。在 GRPO 算法中则表现为对数似然比项。
- **折扣因子** `γ ∈ [0, 1)`。用于衡量未来奖励相对于当前奖励的重要性。`γ = 0.99` 时，时间范围约为 100 步；`γ = 0.9` 时，时间范围约为 10 步。

**马尔可夫性质** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`。即未来状态仅取决于当前状态。若不满足此性质，则说明状态表示不够完整——这是状态本身的缺陷，而非该方法的失败。

**策略与回报值。** 策略 `π(a | s)` 将状态映射到动作的分布上。回报值 `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` 是未来所有奖励的折现之和。在策略 `π` 下，从状态 `s` 开始的预期回报值为 `V^π(s) = E[G_t | s_t = s]`。而特定动作下的预期回报值为 Q 值 `Q^π(s, a) = E[G_t | s_t = s, a_t = a]`。所有的强化学习算法都会先估计这两个值中的一个，再据此优化策略 `π`。

**贝尔曼方程。** 本阶段所有方法所依赖的不动点方程：

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

这些方程将预期回报拆分为“当前步骤的奖励”与“到达下一状态后的折现价值”。这类问题具有递归特性。第 9 阶段的所有算法要么通过迭代该方程直至收敛（动态规划），要么从中采样（蒙特卡洛方法），或是仅推进一步进行估算（时差分法）。

```figure
discount-horizon
```

## 构建它

### 步骤 1：一个极小的确定性马尔可夫决策过程

一个 4×4 的 GridWorld 环境。智能体从左上角出发，终端位于右下角，每步奖励为 -1，可用动作包括 `{up, down, left, right}`。详情参见 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行代码。这就是整个环境模型。包含确定性状态转移、恒定的步长惩罚，以及一个吸收态的终止状态。

### 步骤 2：部署策略

策略是一个从状态到动作分布的函数。最简单的形式为均匀随机选择。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

运行该随机策略 1000 次。在 4×4 的棋盘上，其平均收益约为 -60 至 -80。最优收益为 -6（沿右下方向直线行进）。缩小这一差距是第 9 阶段的核心任务。

### 步骤 3：通过贝尔曼方程精确计算 `V^π`

对于规模较小的MDP，贝尔曼方程表现为一个线性系统。需枚举所有状态，应用期望值计算，并不断迭代直至状态价值不再发生变化。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

这就是迭代策略评估。它是 Sutton 与 Barto 提出的首个算法，也是此后所有强化学习方法的理论基础。

### 步骤 4：`γ` 是一个具有物理意义的超参数

有效视野长度大致为 `1 / (1 - γ)`。当 `γ = 0.9` 时，对应 10 步；`γ = 0.99` 时，对应 100 步；`γ = 0.999` 时，对应 1000 步。

该值过低会导致智能体行为短视；过高则会导致奖励分配出现噪声，因为许多早期步骤需要共同承担对遥远未来奖励的贡献。由于训练集时长较短且有限，大语言模型强化学习对齐通常采用 `γ = 1` 的设置。控制任务多使用 `0.95–0.99` 的值，而长视野策略游戏则常用 `0.999`。

## 常见陷阱

- **非马尔可夫状态。** 如果需要结合最近三次观测结果来做出决策，那么“状态”就不仅仅是当前的单个观测值。解决方案：使用堆叠帧结构（Atari游戏中的DQN会堆叠4层帧）或采用循环状态结构（在观测值上应用LSTM/GRU）。
- **稀疏奖励。** 仅基于胜负的奖励机制在状态空间较大的情况下几乎会导致学习失败。需要对奖励进行整形处理（生成中间信号），或者通过模仿学习来进行辅助训练（第9阶段·09）。
- **奖励操纵问题。** 最优化代理奖励往往会产生异常行为。OpenAI的赛船智能体为了不断收集能量道具而持续绕圈，始终无法完成比赛。务必根据目标结果而非代理指标来定义奖励。
- **折扣因子设定错误。** 在无限时间跨度任务中将`γ = 1`会导致所有状态的价值都变为无穷大。必须通过设置有限的时间跨度或保证`γ < 1`的方式来对数值进行限制。
- **奖励量级问题。** {+100, -100}与{+1, -1}这样的奖励尺度虽然能产生相同的最优策略，但梯度幅值却存在巨大差异。在将数据输入PPO/DQN算法之前，应将其归一化到接近`[-1, 1]`的范围内。

## 使用它

2026年技术栈要求在处理代码之前，先将所有强化学习流程转化为MDP：

| 情景 | 状态 | 动作 | 奖励 | γ |
|-----------|-------|--------|--------|---|
| 控制（移动、操作） | 关节角度 + 速度 | 连续扭矩 | 针对特定任务的奖励函数 | 0.99 |
| 游戏（国际象棋、围棋、扑克） | 棋盘状态 + 历史记录 | 合法走法 | 获胜=+1 / 失利=-1 | 1.0（有限值） |
| 库存/定价管理 | 存货量 + 需求量 | 订购数量 | 收入 - 成本 | 0.95 |
| 大语言模型的RLHF训练 | 上下文标记 | 下一个标记 | 结尾处的奖励模型得分 | 1.0（单集约200个标记） |
| 推理任务的GRPO方法 | 提示语 + 部分响应 | 下一个标记 | 结尾时的验证器评分0/1 | 1.0 |

在编写任何训练循环之前，需先定义这五个元组。大多数“强化学习无法正常工作”的错误报告，其根源都在于理论层面的MDP建模存在缺陷。

## 发布它

保存为 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 练习题

1. **简单。** 在 `code/main.py` 中实现 4×4 的 GridWorld 环境以及随机策略的滚动评估方法。运行 10,000 轮实验，输出收益的均值与标准差，并将其与最优收益值（-6）进行比较。
2. **中等难度。** 对于均匀随机策略，使用 `γ ∈ {0.5, 0.9, 0.99}` 分别运行 `policy_evaluation` 函数，并将计算得到的状态价值 `V` 以 4×4 网格的形式打印出来。解释为何当 `γ` 的值越大时，靠近终点的状态价值增长得越快。
3. **高难度。** 将 GridWorld 环境改为随机性版本：每个动作有概率 `p = 0.1` 转移到相邻方向。重新对均匀策略进行价值评估。请问 `V[start]` 的值会变好还是变差？原因是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MDP | “强化学习框架” | 满足马尔可夫性质的元组 `(S, A, P, R, γ)`。 |
| 状态 | “智能体所观察的内容” | 在选定策略类下，用于描述未来动态的充分统计量。 |
| 策略 | “智能体的行为” | 条件分布 `π(a \| s)` 或确定性映射 `s → a`。 |
| 回报 | “总奖励” | 从当前步骤开始，经过折现后的累加和 `Σ γ^t r_t`。 |
| 价值 | “状态的好坏程度” | 在策略 `π` 下，从状态 `s` 出发所能获得的期望回报。 |
| Q值 | “动作的好坏程度” | 在策略 `π` 下，从状态 `s` 出发并首先执行动作 `a` 所能获得的期望回报。 |
| 贝尔曼方程 | “动态规划递归关系” | 将价值/Q值分解为一步奖励与折现后的后续状态价值的不动点形式。 |
| 折现因子 `γ` | “未来与现在的权重” | 对远期奖励的几何加权系数；有效时间范围约为 `~1/(1-γ)`。 |

## 延伸阅读

- [Sutton & Barto (2018). 《强化学习：导论》，第2版。](http://incompleteideas.net/book/RLbook2020.pdf) —— 相关教材。第3章介绍了MDP与贝尔曼方程；第1章阐述了作为后续所有内容基础的奖励假设。
- [Bellman (1957). 《动态规划》](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) —— 贝尔曼方程的起源文献。
- [OpenAI Spinning Up — 第1部分：核心概念](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) —— 从深度强化学习角度出发的简明MDP入门指南。
- [Puterman (2005). 《马尔可夫决策过程》](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) —— 关于MDP及精确求解方法的运筹学参考书。
- [Littman (1996). 《序列决策算法》（博士论文）](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) —— 将MDP作为动态规划特例进行最清晰推导的文献。
