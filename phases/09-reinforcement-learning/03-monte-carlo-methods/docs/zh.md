# 蒙特卡洛方法——从完整训练集学习

> 动态规划需要一个模型。而蒙特卡洛方法仅需实验序列即可。运行策略，观察回报值，然后对它们取平均值。这是强化学习中最简单的理念——也是开启后续所有技术的基础。

**类型：** 构建
**语言：** Python
**先修课程：** 第9阶段 · 01（MDP），第9阶段 · 02（动态规划）
**时长：** 约75分钟

## 问题所在

动态规划虽然优雅，但它假设我们可以为每一个状态和动作查询 `P(s' | s, a)` 的值。而在现实世界中，几乎不存在如此理想的情况。机器人无法通过解析计算出关节扭矩作用后相机像素的分布情况；定价算法也无法对所有可能的客户反应进行积分处理；大型语言模型也无法枚举出某个标记之后的所有可能延续内容。

我们需要一种仅需具备从环境中*采样*能力的方法。运行策略，获取一条轨迹 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`，并利用它来估算数值。这就是蒙特卡洛方法。

从动态规划转向蒙特卡洛在哲学层面具有重要意义：我们不再依赖*已知的模型 + 精确的回溯值*，而是采用*采样得到的行为序列 + 平均回报值*。虽然方差会显著增加，但其适用范围却大幅扩展。本课程之后的所有强化学习算法——包括时序差分法、Q学习、REINFORCE、PPO以及GRPO——本质上都属于蒙特卡洛估计器，有时还会在其基础上叠加自举技术。

## 概念概述

![Monte Carlo：滚动采样、计算回报并取平均值；首次访问法与每次访问法对比](../assets/monte-carlo.svg)

**一句话概括核心思想：** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`，其中`G^{(i)}(s)`表示在策略`π`下访问状态`s`时观测到的回报序列。

**首次访问法与每次访问法的区别。** 对于一个多次访问状态`s`的训练集，首次访问法仅统计第一次访问的回报；而每次访问法则统计所有访问的回报。在极限情况下，这两种方法均无偏。由于样本独立同分布，首次访问法更易于分析。每次访问法每轮需要更多数据，但在实际应用中通常收敛速度更快。

**增量均值法。** 无需存储所有回报，只需更新当前平均值：

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

重新整理可得：`V_new = V_old + α · (target - V_old)`，其中`α = 1/n`。若用常数步长`α ∈ (0, 1)`替代`1/n`，即可得到一个非平稳的蒙特卡洛估计器，用于追踪策略`π`的变化。这一转变正是从蒙特卡洛方法到时间差分方法，乃至所有现代强化学习算法的起点。

**探索问题随之出现。** 动态规划通过穷举方式遍历了所有状态，而蒙特卡洛方法仅能处理策略实际访问过的状态。如果策略`π`是确定性的，状态空间中的某些区域将永远无法被采样，其价值估计也将始终为零。历史上出现过三种解决方案，按出现顺序如下：

1. **开始探索。** 每个训练集从随机状态`(s, a)`对开始。虽然能保证状态覆盖，但在实际中并不现实（无法让机器人“重置”到任意状态）。
2. **ε-贪婪策略。** 根据当前的Q值采取贪婪行动，但以概率`ε`选择随机动作。这样所有状态-动作对最终都会被采样到。
3. **离线蒙特卡洛方法。** 在行为策略`μ`下收集数据，再通过重要性采样来学习目标策略`π`的信息。该方法方差较大，但为后续的DQN等基于回放缓冲区的算法奠定了基础。

**蒙特卡洛控制。** 其流程与策略迭代类似，即先评估、再改进、再评估，只不过评估环节是基于采样的：

1. 运行策略`π`，获取一个训练集。
2. 根据观测到的回报更新`Q(s, a)`的值。
3. 使策略`π`对当前的Q值采用ε-贪婪策略。
4. 重复上述步骤。

在适度条件下（所有状态对都会被无限次访问，且步长`α`满足Robbins-Monro条件），该方法必然以概率1收敛到最优的`Q*`和`π*`。

```figure
epsilon-greedy
```

## 构建它

### 步骤 1：发布 → (s, a, r) 列表

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

没有模型，仅有 `env.reset()` 和 `env.step(s, a)`。接口与 Gym 环境相同，但更为精简。

### 步骤 2：计算收益（反向扫描）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

单次遍历，时间复杂度为 `O(T)`。通过反向递推公式 `G_t = r_{t+1} + γ G_{t+1}` 可以避免重复求和。

### 步骤 3：首次访问时的 MC 评估

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

仅需三行代码即可完成工作：将状态标记为首次访问，递增计数器，并更新滑动平均值。

### 步骤 4：ε-贪婪 MC 控制（在策略内）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 步骤 5：与 DP 金标准进行对比

当剧集数趋向于无穷大时，您对 `V^π` 的蒙特卡洛估计值应与第 02 课中的动态规划结果一致。实际测试中：在 4×4 的 GridWorld 环境中进行 50,000 次迭代后，其结果与动态规划法的答案的误差通常在 `~0.1` 左右。

## 常见陷阱

- **无限回合数问题。** 最大值算法（MC）要求回合必须*终止*。如果策略可以无限循环，需限制 `max_steps` 的数值，并将该限制视为隐式失败。使用随机策略的 GridWorld 通常会出现超时现象——这是正常现象，只需确保正确统计超时次数即可。
- **方差问题。** 最大值算法使用完整的回报值。在长回合数场景下，方差会非常大——仅末尾一次不利的奖励就会使 `V(s_0)` 发生同等幅度的变化。时间差分方法（第 04 课）通过自举法降低了这种方差。
- **状态覆盖问题。** 在 Q 值表完全未知且存在多个相同值的情况下，贪婪最大值算法只会尝试某一个动作。此时必须采用探索策略，如 ε-贪婪、基于起始点的探索或 UCB 算法。
- **非平稳策略问题。** 如果策略 `π` 发生变化（如在 MC 控制中），旧的回报值实际上来自不同的策略。常数 α 最大值算法可以处理这种情况，而样本平均最大值算法则无法处理。
- **离线策略重要性采样问题。** 权重 `π(a|s)/μ(a|s)` 会在整个轨迹上相乘。随着时间范围的扩大，方差会急剧上升。可通过基于每次决策的加权重要性采样来限制方差，或改用时间差分方法。

## 使用它

2026年蒙特卡洛方法的应用前景：

| 应用场景 | 选择蒙特卡洛方法的理由 |
|----------|------------------------|
| 短周期博弈（二十一点、扑克） | 对局会自然结束，收益数据较为清晰。 |
| 对已记录策略的离线评估 | 基于存储的轨迹计算平均折现回报。 |
| 蒙特卡洛树搜索（AlphaZero） | 从树叶节点出发的蒙特卡洛模拟用于指导节点选择。 |
| 大语言模型强化学习评估 | 计算给定策略在采样完成任务中的平均奖励。 |
| PPO算法中的基线估计 | 优势目标 `A_t = G_t - V(s_t)` 中使用的 `G_t` 即为蒙特卡洛方法计算的结果。 |
| 强化学习教学 | 最简单且实际可行的算法——去除自举机制即可理解其核心原理。 |

现代深度强化学习算法（PPO、SAC）通过 `n` 步回报或GAE，在纯蒙特卡洛方法（完整回报）与纯TD方法（单步自举）之间进行插值。这两种极端情况实际上都属于同一类估计器。

## 发布它

保存为 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 练习题

1. **简单。** 在 4×4 GridWorld 环境中实现基于均匀随机策略的第一次访问蒙特卡洛评估方法。运行 10,000 个回合，将 `V(0,0)` 的值作为回合数的函数绘制出来，并与动态规划方法的计算结果进行对比。
2. **中等难度。** 实现 ε-贪婪蒙特卡洛控制策略，其中 `ε` 的取值为 `{0.01, 0.1, 0.3}`。在运行 20,000 个回合后比较平均回报值。该曲线呈现何种形态？偏差-方差权衡出现在哪个区间？
3. **高难度。** 实现基于重要性采样的*离策略*蒙特卡洛方法：在均匀随机策略 `μ` 下收集数据，进而估计确定性最优策略 `π` 的 `V^π` 值。比较普通的重要性采样、逐次决策的重要性采样以及加权重要性采样三种方法的方差大小，哪种方法的方差最低？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 蒙特卡洛方法 | “随机抽样” | 通过对该分布的独立同分布样本进行平均，来估计期望值。 |
| 返回值 `G_t` | “未来奖励” | 从步骤 `t` 到剧集结束的所有折现奖励之和：`Σ_{k≥0} γ^k r_{t+k+1}`。 |
| 首次访问蒙特卡洛方法 | “每个状态只计数一次” | 仅剧集中首次访问某个状态时的信息会被用于价值估计。 |
| 每次访问蒙特卡洛方法 | “利用所有访问记录” | 每次访问都会被纳入计算；虽然存在轻微偏差，但样本效率更高。 |
| ε-贪婪策略 | “探索噪声” | 以概率 `1-ε` 选择贪婪动作，以概率 `ε` 选择随机动作。 |
| 重要性采样 | “纠正从错误分布中抽样带来的问题” | 通过 `π(a\|s)/μ(a\|s)` 的乘积对回报值进行重新加权，从而利用来自 `μ` 数据的信息来估计 `V^π`。 |
| 在策略学习 | “利用自身的数据学习” | 目标策略等于行为策略。典型代表包括普通蒙特卡洛方法、PPO、SARSA。 |
| 异策略学习 | “利用他人的数据学习” | 目标策略不等于行为策略。典型代表包括基于重要性采样的蒙特卡洛方法、Q学习、DQN。 |

## 延伸阅读

- [Sutton & Barto (2018). 第 5 章 — 蒙特卡洛方法](http://incompleteideas.net/book/RLbook2020.pdf) — 该领域的经典论述。
- [Singh & Sutton (1996). 基于替换资格轨迹的强化学习](https://link.springer.com/article/10.1007/BF00114726) — 首次访问与每次访问分析。
- [Precup, Sutton, Singh (2000). 用于离线策略评估的资格轨迹](http://incompleteideas.net/papers/PSS-00.pdf) — 离线蒙特卡洛方法及方差控制。
- [Mahmood et al. (2014). 用于离线学习的加权重要性采样](https://arxiv.org/abs/1404.6362) — 现代低方差的重要性采样估计器。
- [Tesauro (1995). TD-Gammon：一个自我教学的西洋双陆棋程序](https://dl.acm.org/doi/10.1145/203330.203343) — 首次大规模实证表明蒙特卡洛/时间差分自玩方法能够达到超人类水平；为本阶段后半部分所有内容提供了概念基础。
