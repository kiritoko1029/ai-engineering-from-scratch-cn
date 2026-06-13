# 策略梯度——从零实现 REINFORCE

> 停止估算价值。直接对策略进行参数化，计算期望回报的梯度，然后沿上升方向迭代优化。Williams（1992）在单个定理中便阐述了这一原理，这也是 PPO、GRPO 以及所有大语言模型强化学习循环存在的基础。

**类型：** 构建
**编程语言：** Python
**先修知识：** 第3阶段 · 03（反向传播）、第9阶段 · 03（蒙特卡洛方法）、第9阶段 · 04（时序差分学习）
**所需时间：** 约75分钟

## 问题所在

Q学习与DQN通过参数化*价值函数*来决定动作选择，通常采用`argmax Q`的方式选取最优动作。这种方法在动作和状态均为离散值时表现良好，但当动作为连续型（比如如何在10维扭矩空间中使用`argmax`？）或需要随机策略时就会失效，因为`argmax`本质上属于确定性算法。

而策略梯度则改为参数化*策略*。`π_θ(a | s)`是一个神经网络，用于输出动作的分布概率，可从中采样以决定实际执行的动作。随后计算期望回报关于参数`θ`的梯度，并沿该梯度方向进行优化，整个过程无需使用`argmax`，也不涉及贝尔曼递归，仅需对目标函数`J(θ) = E_{π_θ}[G]`执行梯度上升即可。

REINFORCE定理（Williams, 1992）证明了该梯度是可计算的：`∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`。具体实现步骤为：运行一个训练回合，计算该回合的回报值，在每一步将该回报乘以`∇ log π_θ(a | s)`的结果，最后对所有步长的结果取平均值并继续进行梯度上升优化，如此循环即可。

2026年所有的LLM-RL算法——包括PPO、DPO和GRPO——本质上都是REINFORCE的改进版本。彻底掌握REINFORCE是理解后续内容以及第10·07阶段（RLHF实现）和第10·08阶段（DPO）的基础前提。

## 概念概述

![策略梯度：softmax 策略、log-π 梯度以及基于回报的更新](../assets/policy-gradient.svg)

**策略梯度定理。** 对于任何由参数 `θ` 参数化的策略 `π_θ`：

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

其中，`G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` 表示从步骤 `t` 开始的折现回报。该期望值是在从 `π_θ` 中采样的完整轨迹 `τ` 上计算的。

**证明过程很简短。** 对期望值下的表达式 `J(θ) = Σ_τ P(τ; θ) G(τ)` 求导。利用公式 `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)`（对数微分技巧）。将 `log P(τ; θ)` 分解为 `Σ log π_θ(a_t | s_t)` 以及不依赖于 `θ` 的环境项。由于环境项会消失，经过两行代数运算即可得到该定理。

**降低方差的技巧。** 原始的 REINFORCE 方法方差极大——回报值存在噪声，`∇ log π` 也存在噪声，二者的乘积噪声更为严重。有两种常用的解决方法：

1. **基线减法。** 用 `G_t - b(s_t)` 替换 `G_t`，其中 `b(s_t)` 是任何不依赖于 `a_t` 的基线值。由于 `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`，因此该方法是无偏的。常见的选择是由评价器学习得到的 `b(s_t) = V̂(s_t)`，即演员-评价器框架（第 07 节）。
2. **未来回报法。** 用 `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)` 替换 `Σ_t G_t · ∇ log π_θ(a_t | s_t)`。对于某个给定动作而言，只有未来的回报才起作用——过去的奖励只会带来均值为零的噪声。

将这两种方法结合后，可得：

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

这就是加入了基线的 REINFORCE 方法，它也是 A2C（第 07 节）和 PPO（第 08 节）的直接前身。

**Softmax 策略参数化。** 对于离散动作，常用的表达式为：

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

其中 `f_θ` 是任意能够输出每个动作得分的神经网络。其梯度形式较为简洁：

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

即，所选动作的得分减去其在该策略下的期望值。

**连续动作的高斯策略。** 其表达式为 `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`。`∇ log N(a; μ, σ)` 具有封闭形式的解。这就是第 9 · 07 阶段中 SAC 方法所需的所有内容。

```figure
policy-gradient-landscape
```

## 构建它

### 步骤 1：softmax 策略网络

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

对于表格型环境，可使用线性策略（每个动作对应一个权重向量）。在处理Atari游戏时，则替换为卷积神经网络，并保留softmax输出层。

### 步骤 2：采样与对数概率

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 步骤 3：在捕获日志概率的情况下进行部署

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 步骤 4：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

梯度 `∇ log π(a|s) = e_a - π(·|s)`（即 `a` 的 onehot 表示减去各概率值）是 softmax 策略梯度的核心。请将其牢记在心。

### 步骤 5：基线设定

对最近几轮实验中的 `G` 值计算滑动平均，即可有效降低方差，从而使 4×4 的 GridWorld 环境能够正常运行；该算法通常需要约 500 轮实验才能收敛。若将基准模型升级为通过学习得到的 `V̂(s)`，则可构建出演员-评论家架构。

## 常见陷阱

- **梯度爆炸。** 模型输出值可能变得极其巨大。在将 `G` 与 `∇ log π` 相乘之前，务必先对整个批次中的 `G` 进行归一化，使其接近 `~N(0, 1)`。
- **熵崩溃。** 策略过早收敛为近乎确定性的动作，停止探索并陷入僵局。解决方法：在目标函数中加入熵奖励项 `β · H(π(·|s))`。
- **方差过高。** 基本的 REINFORCE 方法需要数千个训练回合才能达到效果。标准的解决方法是使用评价器基线（第 07 课）或 TRPO/PPO 的信赖域方法（第 08 课）。
- **样本效率低。** 在策略内学习方法中，每次更新后都会丢弃所有历史状态转移数据。而通过重要性采样进行的策略外校正虽然能重新利用这些数据，但会带来方差问题（PPO 方法中的比率项实际上是对 IS 权重的截断处理）。
- **非平稳梯度。** 100 个回合之前的相同梯度所使用的仍是旧版本的 `π`。正因如此，策略内学习方法需要每隔几次训练就进行更新。
- **功劳分配问题。** 若不使用累积奖励，历史奖励会引入噪声。因此应始终使用累积奖励。

## 使用它

在2026年，REINFORCE已很少被直接使用，但其梯度公式却无处不在：

| 应用场景 | 推导出的方法 |
|----------|---------------|
| 连续控制 | 基于高斯策略的PPO / SAC |
| LLM强化学习对齐 | 带有KL惩罚项、在词元级策略上运行的PPO |
| LLM推理（DeepSeek） | GRPO —— 带有群组相对基线的REINFORCE，无评价器 |
| 多智能体系统 | 集中式评价器的REINFORCE（MADDPG, COMA） |
| 离散动作机器人 | A2C、A3C、PPO |
| 仅基于偏好的设置 | DPO —— 将REINFORCE重写为偏好似然损失函数，无需采样 |

当你在2026年的训练脚本中看到`loss = -advantage * log_prob`时，这就是带有基线的REINFORCE。诸如DPO、GRPO、RLOO等整个论文，其实都是基于这一行公式所衍生的降低方差技巧。

## 发布它

另存为 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 练习题

1. **简单。** 在 4×4 的 GridWorld 环境中实现基于线性 softmax 策略的 REINFORCE 算法。不设置基准线，训练 1,000 个回合。绘制学习曲线，并计算方差（回报的标准差）。
2. **中等。** 添加一个滑动平均基准线，再次进行训练。比较该基准线版本与原始版本的样本效率及方差。基准线能将收敛所需的步数减少多少？
3. **困难。** 增加熵奖励项 `β · H(π)`。遍历 `β ∈ {0, 0.01, 0.1, 1.0}` 的不同取值。绘制最终回报值与策略熵的值。在此任务中，最优的 `β` 值是多少？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 策略梯度 | “直接训练策略” | `∇J(θ) = E[G · ∇ log π_θ(a\|s)]`；基于对数导数技巧推导得出。 |
| REINFORCE | “最初的策略梯度算法” | Williams（1992）提出；蒙特卡洛回报值乘以对数策略梯度。 |
| 对数导数技巧 | “得分函数估计器” | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`；使期望值的梯度更易处理。 |
| 基线 | “降低方差” | 从 `G` 中减去的任意函数 `b(s)`；由于 `E[b · ∇ log π] = 0`，因此是无偏的。 |
| 未来回报 | “仅考虑未来的回报” | 使用 `G_t^{from t}` 而非完整的 `G_0`；结果更准确且方差更低。 |
| 熵奖励 | “鼓励探索行为” | `+β · H(π(·\|s))` 这一项可防止策略过度收敛。 |
| 在线学习 | “基于最新观测数据进行训练” | 梯度期望是针对当前策略计算的——无法直接复用旧数据。 |
| 优势函数 | “比平均值好多少” | `A(s, a) = G(s, a) - V(s)`；REINFORCE-with-baseline 算法会将其作为乘数使用。 |

## 延伸阅读

- [Williams (1992). 连通主义强化学习中的简单统计梯度跟踪算法](https://link.springer.com/article/10.1007/BF00992696) —— REINFORCE方法的原始论文。
- [Sutton等人 (2000). 带函数逼近的强化学习策略梯度方法](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) —— 结合函数逼近的现代策略梯度定理。
- [Sutton & Barto (2018). 第13章 —— 策略梯度方法](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书形式的阐述。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) —— 配有PyTorch代码的清晰教学性说明。
- [Peters & Schaal (2008). 利用策略梯度进行运动技能的强化学习](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) —— 方差降低方法以及将REINFORCE与信赖域系列算法（TRPO、PPO）联系起来的自然梯度视角。
