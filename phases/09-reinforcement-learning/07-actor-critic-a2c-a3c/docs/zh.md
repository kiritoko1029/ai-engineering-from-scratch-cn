# 演员-评论家算法——A2C与A3C

> REINFORCE 算法存在噪声问题。通过加入一个用于学习 `V̂(s)` 的评价器，并将评估值从回报中减去，即可得到期望值相同但方差大幅降低的优势函数。这就是演员-评论家算法。A2C 以同步方式运行该算法；而 A3C 则在多个线程间并行运行。这两种方法都是所有现代深度强化学习技术的核心思想模型。

**类型：** 构建
**语言：** Python
**先修课程：** 第 9 阶段 · 04（时序差分学习），第 9 阶段 · 06（REINFORCE）
**时长：** 约 75 分钟

## 问题所在

原生 REINFORCE 方法虽然可行，但其方差极大。蒙特卡洛方法返回的 `G_t` 值在不同训练轮次之间的波动幅度可达 10 倍以上。将该噪声与 `∇ log π` 相乘后再取平均，得到的梯度估计器需要数千个训练轮次才能使策略发生与较少次数的 DQN 更新所能达到的相同变化量。

这种高方差源于直接使用原始回报值。如果减去一个基线值 `b(s_t)` —— 任何状态函数，包括学习得到的价值函数 —— 则期望值保持不变，方差则会下降。最易处理的基线值为 `V̂(s_t)`。此时，与 `∇ log π` 相乘的量即为*优势函数*：

`A(s, a) = G - V̂(s)`

若某个动作产生的回报高于平均值，则该动作良好；反之则较差。结合了学习型评价器的 REINFORCE 方法被称为*演员-评论家*架构。评价器为演员提供了一个方差较低的“教师”模型。2015 年之后的所有深度策略方法（A2C、A3C、PPO、SAC、IMPALA）均属于此类架构。

## 概念概述

![Actor-critic：策略网络与价值网络，TD残差即优势函数](../assets/actor-critic.svg)

**两个网络，一个共享损失函数：**

- **策略网络** `π_θ(a | s)`：用于决定动作。通过采样选择动作，并采用策略梯度法进行训练。
- **价值网络** `V_φ(s)`：用于估计状态下的期望回报。通过最小化 `(V_φ(s) - target)²` 进行训练。

**优势函数。** 其有两种常见形式：

- *蒙特卡洛优势*：`A_t = G_t - V_φ(s_t)`。无偏，但方差较高。
- *TD优势*：`A_t = r_{t+1} + γ V_φ(s_{t+1}) - V_φ(s_t)`。存在偏差（使用 `V_φ` 计算），但方差极低。也被称为*TD残差* `δ_t`。

**n步优势函数。** 在上述两种形式之间进行插值：

`A_t^{(n)} = r_{t+1} + γ r_{t+2} + … + γ^{n-1} r_{t+n} + γ^n V_φ(s_{t+n}) - V_φ(s_t)`

当 `n = 1` 时即为纯TD优势；当 `n = ∞` 时即为蒙特卡洛优势。在实际实现中，针对Atari游戏通常使用 `n = 5`，而在MuJoCo平台上使用PPO算法时则常用 `n = 2048`。

**广义优势估计（GAE）。** Schulman等人（2016年）提出了一种对所有n步优势函数进行指数加权平均的方法：

`A_t^{GAE} = Σ_{l=0}^{∞} (γλ)^l δ_{t+l}`

其中 `λ ∈ [0, 1]`。当 `λ = 0` 时为TD方法（方差低，偏差高）；当 `λ = 1` 时为蒙特卡洛方法（方差高，无偏）；2026年的默认值为 `λ = 0.95`——需通过调整该参数使偏差与方差达到理想平衡。

**A2C：同步优势的演员-评论家算法。** 在 `N` 个并行环境中收集共 `T` 步的数据，为每一步计算优势函数，然后基于合并后的批量数据同时更新策略网络和价值网络，重复此过程。它是A3C的简化版，更具可扩展性。

**A3C：异步优势的演员-评论家算法。** 由Mnih等人（2016年）提出。该方法会启动 `N` 个工作线程，每个线程运行一个环境。每个工作线程在其自身的数据序列上独立计算梯度，随后以异步方式将这些梯度上传至共享的参数服务器。由于各工作线程处理的是不同的数据轨迹，因此无需回放缓冲区。A3C证明了在CPU上也能实现大规模训练。到了2026年，基于GPU的A2C（批量并行环境）因GPU更擅长处理大批量数据而占据主导地位。

**综合损失函数。**

`L(θ, φ) = -E[ A_t · log π_θ(a_t | s_t) ]  +  c_v · E[(V_φ(s_t) - G_t)²]  -  c_e · E[H(π_θ(·|s_t))]`

该损失函数由三部分组成：策略梯度损失、价值回归项以及熵奖励项。常见的初始参数值为 `c_v ~ 0.5`，`c_e ~ 0.01`。

## 构建它

### 步骤 1：批评者

线性评价函数 `V_φ(s) = w · features(s)` 通过均方误差进行更新：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在表格环境下，该评价函数仅需几百个训练轮次即可收敛。在Atari环境中，则需用共享的CNN主干网络与价值头来替代线性评价函数。

### 步骤 2：n 步优势

给定长度为 `T` 的滚动序列以及自举得到的最终值 `V(s_T)`：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是评估目标。`advantages` 则是用于乘以 `∇ log π` 的值。

### 步骤 3：合并更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

在线策略梯度，每次更新仅进行一次迭代，演员网络与评论家网络使用不同的学习率。

### 第 4 步：并行化（A3C 与 A2C）

- **A3C：** 启动 `N` 个线程。每个线程运行独立的训练环境并执行各自的前向传播。定期将梯度更新推送到共享的主节点上。主节点无需加锁——即使出现竞争也会产生噪声，但不会影响整体性能。
- **A2C：** 在单个进程内运行 `N` 个环境实例，将观测值堆叠为 `[N, obs_dim]` 形式的批次，然后进行批量前向传播和批量反向传播。这种方式能提高 GPU 利用率，结果具有确定性，且更易于分析。它是 2026 年的默认算法。

为便于理解，我们的示例代码采用单线程实现；若要改写为批量处理的 A2C，仅需三行 NumPy 代码即可完成。

## 常见陷阱

- **先有批评器偏差，后有演员梯度问题。** 若批评器是随机的，其基准值将毫无信息量，训练过程实际上是在处理纯噪声数据。在启用策略梯度之前，应让批评器运行几百步进行预热，或使用较低的演员学习率。
- **优势归一化。** 每个批次都将优势值归一化为零均值、单位标准差的形式。这能在几乎不增加成本的情况下大幅稳定训练过程。
- **共享主干网络。** 对于图像输入，为演员和批评器使用相同的特征提取器，并设置独立的输出头。这些共享的特征可以同时从两种损失函数中获益。
- **在线策略约束。** A2C算法每次更新仅重复使用一次数据。若重复次数过多，梯度会出现偏差（PPO正是通过重要性采样校正来解决这一问题）。
- **熵值坍缩现象。** 若未设置 `c_e > 0`，策略在几百次更新后就会变得近乎确定性，从而停止探索行为。
- **奖励尺度问题。** 优势值的大小取决于奖励的尺度。需要对奖励进行归一化处理（例如通过运行标准差除法），以确保不同任务间的梯度幅度保持一致。

## 使用它

在2026年的研究中，A2C/A3C极少作为最终选择，但它们是后续所有算法优化的基础架构：

| 方法 | 与A2C的关系 |
|--------|--------------|
| PPO | A2C结合了用于多轮迭代的截断重要性比率机制 |
| IMPALA | A3C加上V-trace离线策略校正技术 |
| SAC（第9阶段·07） | 带有软价值评估器的离线A2C算法（见下一课） |
| GRPO（第9阶段·12） | 不使用评估器、基于群体相对优势的A2C算法 |
| DPO | 将A2C转化为偏好排序损失函数，且不进行采样 |
| AlphaStar / OpenAI Five | 结合联赛训练与模仿预训练的A2C算法 |

如果在2026年的论文中看到“优势”这一概念，通常指的是演员-评论家架构。

## 发布它

另存为 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 练习题

1. **简单。** 在 4×4 GridWorld 环境中使用基于 MC 优势函数（`G_t - V(s_t)`）的方法训练演员-评论家模型。并将该方法的样本效率与第 06 课中的 REINFORCE-with-running-mean-baseline 方法进行比较。
2. **中等难度。** 改为使用 TD 残差优势函数（`r + γ V(s') - V(s)`）。测量优势值批次的方差，其数值会下降多少？
3. **高难度。** 实现 GAE(λ) 算法。遍历 `λ ∈ {0, 0.5, 0.9, 0.95, 1.0}` 的不同取值。绘制最终回报与样本效率的关系图。针对该任务，偏差/方差的最佳平衡点位于何处？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Actor | “策略网络” | `π_θ(a\|s)`，通过策略梯度进行更新。 |
| Critic | “价值网络” | `V_φ(s)`，通过针对回报/TD目标的MSE回归进行更新。 |
| Advantage | “比平均值好多少” | `A(s, a) = Q(s, a) - V(s)`或其估计值。用于计算`∇ log π`的乘数。 |
| TD residual | “δ” | `δ_t = r + γ V(s') - V(s)`；单步优势估计值。 |
| GAE | “插值调节参数” | 由`λ`参数化的n步优势值的指数加权之和。 |
| A2C | “同步式演员-评论家算法” | 在多个环境中批量处理；每次滚动仅进行一次梯度更新。 |
| A3C | “异步式演员-评论家算法” | 工作线程将梯度推送到共享的参数服务器。源自原始论文；在2026年已较少使用。 |
| Bootstrap | “使用终局时的价值值” | 截断滚动过程，通过添加`γ^n V(s_{t+n})`来闭合求和。 |

## 延伸阅读

- [Mnih等人（2016）。深度强化学习的异步方法](https://arxiv.org/abs/1602.01783) —— A3C，最初的异步演员-评论家算法论文。
- [Schulman等人（2016）。基于广义优势估计的高维连续控制](https://arxiv.org/abs/1506.02438) —— GAE。
- [Sutton与Barto（2018）。第13章 —— 演员-评论家方法](http://incompleteideas.net/book/RLbook2020.pdf) —— 相关基础理论；当评论家为神经网络时，可结合第9章关于函数逼近的内容一起学习。
- [Espeholt等人（2018）。IMPALA](https://arxiv.org/abs/1802.01561) —— 基于V-trace离线校正的可扩展分布式演员-评论家算法。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) —— 值得研读的实用A2C/PPO实现方案。
- [Konda与Tsitsiklis（2000）。演员-评论家算法](https://papers.nips.cc/paper/1786-actor-critic-algorithms) —— 用于双时间尺度演员-评论家分解的基础收敛性结果。
