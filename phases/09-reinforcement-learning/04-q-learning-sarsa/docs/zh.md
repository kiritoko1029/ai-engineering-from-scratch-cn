# 时序差分法 —— Q学习与SARSA

> Monte Carlo 方法会一直运行到整个训练回合结束。TD 学习会在每一步之后，通过自举法更新下一个状态的价值估计值。Q 学习属于离线且乐观的学习策略；SARSA 则属于在线且谨慎的学习策略。这两种方法都仅需一行代码即可实现，并且是本阶段所有深度强化学习方法的基石。

**类型：** 构建
**语言：** Python
**先修知识：** 第 9 阶段 · 01（马尔可夫决策过程）、第 9 阶段 · 02（动态规划）、第 9 阶段 · 03（蒙特卡洛方法）
**时长：** 约 75 分钟

## 问题所在

蒙特卡洛方法虽然有效，但存在两个高昂的代价。其一是需要有能够终止的训练回合，且只有在获得最终回报后才会进行更新。如果一个训练回合包含1,000步，那么蒙特卡洛方法就需要等待整整1,000步才能进行任何更新。该方法在实际应用中方差较大、偏差较小，但速度较慢。

动态规划则呈现出相反的特点——具有零方差的自举式备份机制——但其前提是必须拥有已知的模型。

时差学习则在两者之间取得了平衡。它从单个状态转移 `(s, a, r, s')` 出发，生成一步目标值 `r + γ V(s')`，并使 `V(s)` 向该目标值靠拢。该方法无需模型，也不需要完整的训练回合。虽然右侧使用近似值 `V` 会导致一定偏差，但其方差远低于蒙特卡洛方法，且能够从第一步开始就进行在线更新。

这正是所有现代强化学习算法——如 DQN、A2C、PPO、SAC——所依赖的核心原理。第9阶段的内容将围绕函数逼近技术以及基于本课所学的单步时差更新机制而展开的各类技巧进行深入讲解。

## 概念概述

![Q-learning与SARSA：离线策略最大值法与在线策略Q(s', a')](../assets/td.svg)

**V值的TD(0)更新公式：**

`V(s) ← V(s) + α [r + γ V(s') - V(s)]`

括号内的量即为TD误差 `δ = r + γ V(s') - V(s)`。它相当于马尔可夫蒙特卡洛方法中 `G_t - V(s_t)` 的在线版本。要实现收敛，需满足罗宾斯-蒙罗条件（`Σ α = ∞`，`Σ α² < ∞`），且所有状态均会被无限次访问。

**Q-learning。** 一种用于控制的离线策略TD方法：

`Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]`

其中的 `max` 假设从状态 `s'` 开始将始终遵循*贪婪*策略，而无需考虑智能体实际采取的动作。正是这种解耦机制使得Q-learning能够在智能体通过ε-贪婪策略进行探索的同时学习到 `Q*`。Mnih等人（2015年）将其应用于Atari游戏，发展出了深度Q-learning（第05课内容）。

**SARSA。** 一种在线策略TD方法：

`Q(s, a) ← Q(s, a) + α [r + γ Q(s', a') - Q(s, a)]`

其名称来源于元组 `(s, a, r, s', a')`。SARSA使用智能体*实际*采取的下一步动作 `a'`，而非贪婪策略下的 `argmax` 结果。无论当前运行的ε-贪婪策略 `π` 为何，该方法最终都会收敛到 `Q^π`；当 `ε → 0` 时，该值即为 `Q*`。

**悬崖行走任务的差异。** 在经典的悬崖行走任务中（从悬崖上掉下会获得奖励 -100），Q-learning能够学习到沿着悬崖边缘的最优路径，但在探索过程中偶尔仍会遭遇惩罚。而SARSA由于将探索带来的噪声纳入Q值计算，因此能学习到距离悬崖一步之遥的更安全路径。经过训练后，当 `ε → 0` 时，两种方法都能达到最优解。在实际应用中这一点很重要：因为在部署环境中确实存在探索行为，SARSA的表现更为保守。

**期望SARSA。** 用策略 `π` 下 `Q(s', a')` 的期望值替换原公式中的该表达式：

`Q(s, a) ← Q(s, a) + α [r + γ Σ_{a'} π(a'|s') Q(s', a') - Q(s, a)]`

其方差低于SARSA（无需获取 `a'` 的样本），但目标函数仍为在线策略形式。在现代教材中，这通常是默认的模型。

**n步TD与TD(λ)。** 通过在自举之前等待 `n` 步来在TD(0)与马尔可夫蒙特卡洛方法之间进行折衷。当 `n=1` 时即为TD算法，`n=∞` 时则为MC算法。TD(λ)则通过对所有 `n` 值按几何权重 `(1-λ)λ^{n-1}` 进行平均来计算。大多数深度强化学习方法采用的 `n` 值在3到20之间。

```figure
qlearning-gridworld
```

## 构建它

### 步骤 1：基于ε-贪婪策略的SARSA算法

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。其与Q学习的*唯一*区别在于目标行。

### 步骤 2：Q 学习

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 将目标函数与具体行为解耦。正是这个符号区分了在策略内优化与在策略外优化的方法。

### 步骤 3：学习曲线

记录每100个回合的平均回报。Q学习在简单的确定性GridWorld环境中收敛速度更快；而SARSA在处理“悬崖行走”问题时则更为保守。在`code/main.py`中的4×4 GridWorld环境中，当`α=0.1, ε=0.1`时，经过约2,000个回合后，这两种算法的表现均接近最优。

### 步骤 4：与动态规划真值进行比较

运行值迭代算法（第02课）以获得 `Q*` 值。检查 `max_{s,a} |Q_learned(s,a) - Q*(s,a)|` 的数值。一个状态良好的表格型TD智能体在经过10,000个训练回合后，在4×4的GridWorld环境中的该值应控制在`~0.5`左右。

## 常见陷阱

- **初始 Q 值至关重要。** 乐观的初始化值（对于负奖励任务，设 `Q = 0`）有助于鼓励探索行为；而悲观的初始化值则可能使贪婪策略永远陷入困境。
- **α 调度策略。** 对于非平稳问题，使用恒定的 `α` 即可。虽然理论上衰减的 `α_n = 1/n` 可以实现收敛，但实际应用中速度过慢——建议将 `α` 固定在 `[0.05, 0.3]` 范围内，并持续监控学习曲线。
- **ε 调度策略。** 初始值设为较高水平（如 `ε=1.0`），随后逐渐衰减至 `ε=0.05`。“GLIE”条件即表示在无限探索的极限情况下算法仍能收敛。
- **Q 学习中的最大偏差问题。** 当 Q 值存在噪声时，`max` 运算符会导致向上的偏差，进而造成估计值过高——Hasselt 的双 Q 学习方法（被第 05 课中的 DDQN 所采用）通过使用两个 Q 表来解决此问题。
- **非终止性剧集。** 即使没有终止条件，TD 学习依然可以进行，但需要要么限制步数上限，要么在达到上限时正确处理自举过程。标准做法是将上限视为非终止状态，并继续进行自举操作。
- **状态哈希化。** 若状态为元组或张量形式，则应使用可哈希的键（优先选择元组而非列表；对于浮点数元组，需先进行四舍五入处理，而非直接使用原始值）。

## 使用它

2026年的TD学习框架概览：

| 任务类型 | 方法 | 原因 |
|------|------|------|
| 小型表格结构环境 | Q学习 | 能直接学习最优策略。 |
| 在线场景下的安全关键应用 | SARSA / 期望SARSA | 探索阶段更为保守。 |
| 高维状态空间 | DQN（第9阶段·05） | 基于神经网络的Q函数，结合重放机制与目标网络。 |
| 连续动作空间 | SAC / TD3（第9阶段·07） | 在Q网络上进行TD更新；策略网络负责生成动作。 |
| 基于大语言模型的强化学习（奖励模型驱动） | PPO / GRPO（第9阶段·08、12） | 采用演员-评论家架构，通过GAE实现类似TD的学习优势估计。 |
| 离线强化学习 | CQL / IQL（第9阶段·08） | 基于Q学习的算法，并加入保守的正则化处理。 |

2026年的相关论文中，90%的“强化学习”内容其实都是对Q学习或SARSA的进一步拓展。在深入研究之前，请先熟练掌握表格结构的更新机制。

## 发布它

保存为 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 练习题

1. **简单。** 在 4×4 的 GridWorld 环境中实现 Q 学习和 SARSA 算法。对 2,000 个训练轮次进行记录，绘制每 100 轮次的平均回报曲线。哪种算法的收敛速度更快？
2. **中等难度。** 构建一个“走悬崖”环境（尺寸为 4×12，最后一行是奖励为 -100 的悬崖，角色触碰后会重置到起点）。比较 Q 学习和 SARSA 算法最终得到的策略，并截取两者行走路径的截图。哪种算法的角色更接近悬崖？
3. **高难度。** 实现双 Q 学习算法。在带有噪声奖励的 GridWorld 环境中（每步奖励上叠加标准差为 σ=5 的高斯噪声），展示 Q 学习会显著高估 `V*(0,0)` 的值，而双 Q 学习则不会出现这种情况。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| TD误差 | “更新信号” | `δ = r + γ V(s') - V(s)`，即自举残差。 |
| TD(0) | “单步TD” | 仅使用下一状态的估计值，在每次状态转移后进行更新。 |
| Q学习 | “离线策略强化学习基础” | 使用对下一状态动作的`max`运算进行TD更新；无论行为策略如何，都能学习到`Q*`。 |
| SARSA | “在线策略Q学习” | 使用实际的下一动作进行TD更新；针对当前的ε-贪婪策略π，学习`Q^π`。 |
| 期望SARSA | “低方差SARSA” | 用动作`a'`在策略π下的期望值替换随机采样的`a'`。 |
| GLIE | “正确的探索调度” | 具有无限探索能力的极限贪婪策略；是Q学习收敛所必需的。 |
| 自举法 | “在目标函数中使用当前估计值” | 这是TD学习与蒙特卡洛学习的区别所在。虽会产生偏差，但能大幅降低方差。 |
| 最大化偏差 | “Q学习会高估价值” | 对噪声较大的估计值取`max`会导致上偏；可通过双Q学习来消除该偏差。 |

## 延伸阅读

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始论文及收敛性证明。  
- [Sutton & Barto (2018). 第 6 章 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。  
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 针对最大化偏差的修正方法。  
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — Expected SARSA 的提出动机。  
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 首次提出 SARSA（当时称为“改进型连接主义 Q-learning”）的论文。  
- [Sutton & Barto (2018). 第 7 章 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 将 TD(0) 推广至 TD(n)，阐述从 Q-learning 到资格轨迹的演进过程，以及后来在 PPO 中出现的 GAE。
