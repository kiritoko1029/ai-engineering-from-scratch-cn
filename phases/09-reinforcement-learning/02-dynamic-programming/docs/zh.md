# 动态规划——策略迭代与价值迭代

> 动态规划实际上就是带有“作弊”机制的强化学习。你已经知道了状态转移函数和奖励函数，只需不断迭代贝尔曼方程，直到 `V` 或 `π` 的值不再发生变化为止。它是所有基于采样的方法都试图逼近的基准。

**类型：** 构建
**语言：** Python
**先修课程：** 第9阶段 · 01（马尔可夫决策过程）
**时长：** 约75分钟

## 问题所在

你拥有一个模型已知的MDP：你可以查询任意状态-动作对对应的`P(s' | s, a)`和`R(s, a, s')`。库存管理者知晓需求分布，棋盘游戏的状态转换是确定性的，而Gridworld则仅需四行Python代码即可实现——这些都属于拥有“模型”的情况。

无模型强化学习（Q-learning、PPO、REINFORCE）是为没有模型、只能从环境中采样而设计的算法。但当你拥有模型时，存在更快且更优的方法：动态规划。贝尔曼在1957年设计了这些方法，它们依然决定了最优性的定义——当人们提到“该MDP的最优策略”时，指的正是动态规划会返回的策略。

在2026年，你有三个理由需要使用动态规划。首先，强化学习研究中的所有表格型环境（GridWorld、FrozenLake、CliffWalking）都是通过动态规划求解以获得公认最优的策略。其次，精确的值函数有助于你*调试*采样方法：如果Q-learning对`V*(s_0)`的估计与动态规划的答案相差30%，说明该Q-learning算法存在缺陷。第三，现代离线强化学习与规划方法（MCTS、AlphaZero的搜索机制，以及第9·10阶段基于模型的强化学习）都在所学模型或给定模型之上迭代贝尔曼回溯过程。

## 概念概述

![策略迭代与价值迭代并列图](../assets/dp.svg)

**两种算法均为基于贝尔曼方程的不动点迭代。**

**策略迭代。** 通过交替执行两个步骤，直至策略不再发生变化。

1. *评估阶段：* 给定策略 `π`，通过反复应用公式 `V(s) ← Σ_a π(a|s) Σ_{s',r} P(s',r|s,a) [r + γ V(s')]` 直至其收敛，从而计算出 `V^π`。
2. *改进阶段：* 给定 `V^π`，根据 `V^π` 以贪婪方式更新策略 `π`：`π(s) ← argmax_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`。

收敛性得以保证，原因在于：(a) 每次改进步骤要么保持 `π` 不变，要么使某个状态的 `V^π` 严格增加；(b) 确定性策略的空间是有限的。即便状态空间较大，通常也需要约5–20次外层迭代即可收敛。

**价值迭代。** 将评估与改进合并为一次遍历，直接应用贝尔曼*最优性*方程：

`V(s) ← max_a Σ_{s',r} P(s',r|s,a) [r + γ V(s')]`

重复该过程，直至满足条件 `max_s |V_{new}(s) - V(s)| < ε`。最后通过选择贪婪动作来提取策略。由于没有内部评估循环，每次迭代的计算速度更快，但通常需要更多迭代次数才能收敛。

**广义策略迭代（GPI）。** 一种统一的框架。价值函数与策略被置于一个双向改进循环中；任何能够促使二者相互一致的方法（如异步价值迭代、改进型策略迭代、Q学习、演员-评论家算法、PPO）都属于GPI的实例。

**为何 `γ < 1` 至关重要。** 贝尔曼算子是在上范数意义下的 `γ`-压缩算子：`||T V - T V'||_∞ ≤ γ ||V - V'||_∞`。这种压缩特性保证了唯一不动点的存在以及几何收敛性。若放弃 `γ < 1` 的条件，这一保证即不复存在——此时需要引入有限时间步长或吸收态来确保收敛。

```figure
value-iteration-gamma
```

## 构建它

### 步骤 1：构建 GridWorld MDP 模型

继续使用第 01 课中的那个 4×4 GridWorld 环境。我们在此基础上增加了一个随机变体：代理有 `0.1` 的概率会向一个随机的垂直方向滑移。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 会返回一个包含 `(s', r, p)` 元组的列表。这就是整个模型。

### 步骤 2：策略评估

给定策略 `π(s) = {action: prob}`，不断迭代贝尔曼方程，直到价值函数 `V` 的值不再发生变化：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 步骤 3：策略优化

用关于 `V` 的贪婪策略替换 `π`。如果 `π` 没有变化，则返回 —— 此时已达到最优解。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 步骤 4：将它们拼接在一起

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

4×4 网格的典型收敛情况：需要 4–6 次外部迭代。最终输出 `V*(0,0) ≈ -6`，同时会得到一个能严格减少步数的策略。

### 第 5 步：值迭代（单循环版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

相同的不动点，更少的代码行数。

## 常见陷阱

- **忽略终端状态的处理。** 若将贝尔曼算法应用于吸收态，它仍会选出一种“最优动作”，但该动作实际上不会改变任何状态。应通过 `if s == terminal: V[s] = 0` 进行防护。
- **上确界范数与 L2 范数的收敛性差异。** 应使用 `max |V_new - V|` 而非平均值进行判断。理论上的收敛保证是基于上确界范数的。
- **就地更新与同步更新的对比。** 采用就地方式（高斯-赛德尔法）更新 `V[s]` 的收敛速度比使用独立的 `V_new` 字典（雅可比法）更快。实际生产代码中通常采用就地更新方式。
- **策略的平局处理问题。** 若两个动作的 Q 值相等，`argmax` 函数在每次迭代中可能以不同的方式打破平局，从而导致“策略稳定性”检测结果出现波动。应使用稳定的平局判定规则（例如按固定顺序选取第一个动作）。
- **状态空间爆炸问题。** 动态规划算法在每次迭代中的时间复杂度为 `O(|S| · |A|)`。该方法适用于最多约 10⁷ 个状态的情况。超过此数量级后，就需要采用函数近似方法（从第 9·05 阶段开始学习）。

## 使用它

2026年，动态规划（DP）已成为规划器正确性的基准以及其内部循环的核心：

| 应用场景 | 方法 |
|----------|--------|
| 精确求解小型表格型MDP | 价值迭代法（更为简单）或策略迭代法（外部迭代步数更少） |
| 验证Q学习/PPO实现 | 在简易环境中将其与DP最优解V*进行对比 |
| 基于模型的强化学习（第9、10阶段） | 基于所学状态转移模型进行贝尔曼回溯 |
| AlphaZero/MuZero中的规划 | 蒙特卡洛树搜索 = 异步贝尔曼回溯 |
| 离线强化学习（CQL、IQL） | 保守型Q迭代法——在DP基础上对异常行为动作施加惩罚 |

每当有人提到“最优价值函数”时，实际上指的便是“动态规划的不动点”。在论文中看到`V*`或`Q*`时，可将其视为该循环的结果。

## 发布它

保存为 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 练习题

1. **简单。** 在 4×4 的 GridWorld 环境中，使用 `γ ∈ {0.9, 0.99}` 运行价值迭代算法。需要执行多少次迭代才能使得 `max |ΔV| < 1e-6`？将最终的 `V*` 值以 4×4 的网格形式输出。
2. **中等难度。** 在具有随机滑移概率（`0.1`）的 *随机* GridWorld 环境中，比较策略迭代与价值迭代算法的性能。统计指标包括：迭代次数、实际运行时间以及最终得到的 `V*(0,0)` 值。哪种算法在迭代步数上收敛更快？在实际运行时间上又如何？
3. **高难度。** 构建改进版的策略迭代算法：在评估阶段，不再进行直至收敛的迭代，而是仅执行 `k` 次迭代。针对 `k ∈ {1, 2, 5, 10, 50}`，绘制 `V*(0,0)` 的误差值与 `k` 值之间的关系曲线。该曲线反映了评估过程与算法改进之间的权衡关系是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 策略迭代 | “动态规划算法” | 交替进行评估（`V^π`）与改进（基于 `V^π` 的贪婪策略 `π`），直至策略不再变化。 |
| 价值迭代 | “更快的动态规划” | 一次性应用贝尔曼最优性回溯；以几何级数收敛至 `V*`。 |
| 贝尔曼算子 | “递归关系” | `(T V)(s) = max_a Σ P (r + γ V(s'))`；在上范数下为 `γ`-压缩算子。 |
| 压缩性 | “动态规划收敛的原因” | 任何满足 `\|\|T x - T y\|\| ≤ γ \|\|x - y\|\|` 的算子 `T` 都存在唯一的不动点。 |
| GPI | “一切都是动态规划” | 广义策略迭代：任何使 `V` 和 `π` 达到相互一致性的方法。 |
| 同步更新 | “雅可比式” | 在整个计算过程中始终使用旧的 `V` 值；易于分析，但速度较慢。 |
| 内置更新 | “高斯-赛德尔式” | 使用正在被更新的 `V` 值；实际应用中收敛速度更快。 |

## 延伸阅读

- [Sutton & Barto (2018). 第4章 — 动态规划](http://incompleteideas.net/book/RLbook2020.pdf) — 政策迭代与价值迭代的经典阐述。
- [Bertsekas (2019). 强化学习与最优控制](http://www.athenasc.com/rlbook.html) — 对压缩映射论证的严谨分析。
- [Puterman (2005). 马尔可夫决策过程](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 修改后的政策迭代及其收敛性分析。
- [Howard (1960). 动态规划与马尔可夫过程](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 政策迭代的原始论文。
- [Bertsekas & Tsitsiklis (1996). 神经动态规划](http://www.athenasc.com/ndpbook.html) — 构建了从传统动态规划到近似动态规划/深度强化学习的桥梁，为后续所有课程奠定基础。
