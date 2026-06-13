# 深度Q网络（DQN）

> 2013年：Mnih使用原始像素训练了一个Q学习网络，在七款Atari游戏中击败了所有传统强化学习智能体。2015年：该技术扩展至49款游戏，并发表在《自然》杂志上，由此开启了深度强化学习时代。DQN是在Q学习基础上加入了三种能够提升函数逼近稳定性的技巧。

**类型：** 构建
**语言：** Python
**先修知识：** 第3阶段·03（反向传播）、第9阶段·04（Q学习、SARSA）
**耗时：** 约75分钟

## 问题所在

表格Q学习需要为每一对（状态，动作）单独存储一个Q值。国际象棋棋盘的状态数量约为10⁴³。Atari游戏的一帧图像具有210×160×3 = 100,800个特征维度。即便只有数千个状态，表格式强化学习也会失效，更不用说数十亿个状态了。

事后看来，解决方案显而易见：用神经网络`Q(s, a; θ)`来替代Q表。但这种“事后才明白的”方案却经历了数十年的探索。在“致命三重威胁”——函数近似、自举采样以及离策略学习——的共同作用下，基于Q学习的简单函数近似方法会发散。Mnih等人（2013年、2015年）提出了三种能够稳定学习过程的工程技巧：

1. **经验回放**可降低状态转移之间的相关性。
2. **目标网络**用于固定自举采样时的目标值。
3. **奖励裁剪**可用于规范梯度幅值。

DQN在Atari游戏上的应用，标志着首个仅通过单一架构与一组超参数即可解决数十个基于原始像素的控制问题的案例。此后所有“深度强化学习”算法——如DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57等——都是在这一包含三种技巧的基础之上发展而来的。

## 概念概述

![DQN训练循环：环境、回放缓冲区、在线网络、目标网络、贝尔曼TD损失](../assets/dqn.svg)

**目标。** DQN旨在最小化基于神经网络Q函数的单步TD损失：

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

其中，`θ`为在线网络，通过梯度下降在每一步中进行更新；`θ^-`为目标网络，每隔约10,000步从`θ`复制一次权重；`D`为存储历史状态转移的回放缓冲区。

**按重要性排序的三种技巧：**

**经验回放。** 使用容量约为`10⁶`个状态转移的环形缓冲区。在每个训练步骤中，会均匀随机抽取一个小批量数据。该方法能够打破时间相关性（连续帧几乎相同），使网络能够多次从稀有的高奖励状态转移中学习，并降低连续梯度更新之间的关联度。若不采用此方法，基于神经网络的在线TD算法在Atari游戏上会出现发散现象。

**目标网络。** 在贝尔曼方程的两侧使用相同的`Q(·; θ)`网络会导致目标值在每次更新时都发生变动——即“追逐自己的尾巴”。解决办法是保留另一个权重固定的网络`Q(·; θ^-)`。每隔`C`步将`θ`的权重复制到`θ^-`中。这样就能为数千次梯度更新提供一个稳定的回归目标。软更新方式`θ^- ← τ θ + (1-τ) θ^-`（用于DDPG、SAC算法）则是更为平滑的改进方案。

**奖励裁剪。** Atari游戏的奖励幅度在1到1000以上不等。将奖励裁剪为`{-1, 0, +1}`可以防止某一个游戏单独主导梯度计算。当奖励的绝对值具有重要性时此方法不适用，但对于仅需判断正负的Atari游戏而言效果良好。

**双DQN算法。** Hasselt（2016）通过调整策略解决了最大化偏差问题：使用在线网络来*选择*动作，而使用目标网络来*评估*该动作。

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ^-))`

该算法可直接替代原有方法且性能更稳定，建议作为默认方案使用。

**其他改进方案（Rainbow，2017）：** 优先级回放（更多地采样TD误差较大的状态转移）、对抗式架构（分别设置`V(s)`和优势值计算模块）、噪声网络（用于实现探索行为）、n步回报机制、分布式Q函数模型（C51/QR-DQN）以及多步自举法。这些方法各自能提升少量性能，其收益大致具有可加性。

## 构建它

此处的代码仅使用标准库，未依赖 NumPy——我们在一个极小的连续型 GridWorld 环境中使用了自行实现的单隐藏层多层感知机，因此每个训练步骤的耗时均在微秒级。该算法在规模上与 Atari DQN 完全相同。

### 步骤 1：重放缓冲区

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 环境的容量约为 50,000；我们的玩具级测试环境仅需 5,000 即可。

### 步骤 2：一个小型 Q 网络（手动构建的多层感知机）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

前向传播：线性层 → ReLU层 → 线性层。这就是整个神经网络结构。

### 步骤 3：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

该算法形式与第04课中的Q学习相同，仅有两点不同：(a) 我们通过可微的`Q(·; θ)`进行反向传播，而非从表中查值；(b) 目标值使用的是`Q(·; θ^-)`。

### 步骤 4：外部循环

对于每一集，均基于 `Q(·; θ)` 采用 ε-贪婪策略进行决策，将状态转移信息存入缓冲区，抽取一个小批量数据，执行一次梯度更新，并定期执行 `θ^- ← θ` 的同步操作。具体流程如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我们这个状态为16维独热编码的微型GridWorld环境中，智能体仅需约500个训练回合即可学习到近乎最优的策略。而在Atari游戏环境中，需将训练规模扩展至2亿帧，并引入卷积神经网络作为特征提取器。

## 常见陷阱

- **致命三重威胁。** 函数逼近、离线策略与自举方法结合使用时可能会出现发散现象。DQN通过目标网络和回放机制来缓解这一问题，二者均不可移除。
- **探索行为。** ε值必须逐渐衰减，通常在训练开始的前约10%时间内从1.0降至0.01。若早期探索不足，Q网络会收敛到局部极小值区域。
- **高估问题。** 对噪声较大的Q值取`max`操作会导致估计结果向上偏移。在实际应用中应始终使用双DQN算法。
- **奖励尺度调整。** 需对奖励值进行裁剪或归一化处理，因为梯度大小与奖励幅值成正比。
- **回放缓冲区的冷启动问题。** 在缓冲区中积累数千条样本之前不应开始训练。仅基于约20个样本计算的初始梯度容易导致过拟合。
- **目标网络同步频率。** 同步频率过高相当于没有目标网络；频率过低则会导致目标数据陈旧。Atari DQN采用每10,000次环境步骤进行一次同步的策略。经验法则为：大约在训练总时长的1/100处进行同步。
- **观测值预处理。** Atari DQN通过堆叠4帧图像来构建马尔可夫状态。任何包含速度信息的环境都需要通过帧堆叠或使用循环状态结构来处理。

## 使用它

2026年，DQN已很少被视为最先进的算法，但仍作为离线策略学习的参考算法：

| 任务类型 | 首选方法 | 为何不选择DQN？ |
|----------|----------|----------------|
| 类Atari的离散动作任务 | Rainbow DQN或Muesli | 同一框架，但具备更多优化技巧。 |
| 连续控制任务 | SAC / TD3（第9阶段·07） | DQN没有策略网络。 |
| 在线策略学习/高吞吐量场景 | PPO（第9阶段·08） | 无需回放缓冲区；更易于扩展。 |
| 离线强化学习 | CQL / IQL / Decision Transformer | 采用保守的Q值目标函数，不会出现自举导致的数值爆炸问题。 |
| 大型离散动作空间任务（如推荐系统） | 带有动作嵌入的DQN或IMPALA | 效果良好；细节处理至关重要。 |
| 大语言模型强化学习 | PPO / GRPO | 以序列级而非步级进行训练；损失函数也有所不同。 |

这些核心理念依然适用。回放网络与目标网络广泛存在于SAC、TD3、DDPG、SAC-X以及AlphaZero的自玩缓冲区中，所有离线强化学习方法也都采用类似机制。奖励裁剪的概念则以优势归一化的形式延续于PPO中。架构始终是设计的蓝图。

## 发布它

保存为 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 练习题

1. **简单。** 运行 `code/main.py`，绘制每轮的回报曲线。需要多少轮才能使运行均值超过 -10？
2. **中等难度。** 禁用目标网络（在贝尔曼目标的双方都使用在线网络）。检测训练稳定性——回报值是会出现振荡还是发散？
3. **高难度。** 添加双 DQN：使用在线网络来选择 `argmax a'`，使用目标网络来进行评估。在具有噪声奖励的 GridWorld 环境中，分别运行包含与不包含双 DQN 的版本，比较 1,000 轮后的 `Q(s_0, best_a)` 偏差与真实的 `V*(s_0)` 值。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| DQN | “深度 Q 学习” | 采用神经网络形式的 Q 函数、回放缓冲区以及目标网络的 Q 学习算法。 |
| 经验回放 | “随机排序的转换数据” | 在每个梯度更新步骤中均匀从环形缓冲区中采样数据；用于降低数据之间的相关性。 |
| 目标网络 | “冻结的自举网络” | 定期复制 Q 值并用作贝尔曼目标函数，从而稳定训练过程。 |
| 致命三要素 | “导致强化学习发散的原因” | 函数逼近 + 自举机制 + 非策略性采样 = 无法保证收敛。 |
| 双重 DQN | “解决最大化偏差的方案” | 在线网络负责选择动作，目标网络则负责评估该动作的价值。 |
| 对战型 DQN | “V 头与 A 头结构” | 将 Q 值分解为 V + A - mean(A) 的形式；虽然输出相同，但能提升梯度流的质量。 |
| Rainbow 算法 | “集所有技巧于一身” | 结合了 DDQN、PER、对战型结构、多步预测、噪声注入以及分布式训练等多种技术。 |
| PER | “优先级回放” | 按照 TD 错误的大小比例来采样转换数据。 |

## 延伸阅读

- [Mnih等人（2013）。《利用深度强化学习玩Atari游戏》](https://arxiv.org/abs/1312.5602) —— 2013年NeurIPS研讨会论文，开启了深度强化学习的发展。
- [Mnih等人（2015）。《通过深度强化学习实现人类水平的控制能力》](https://www.nature.com/articles/nature14236) —— 发表在《自然》杂志上的论文，展示了基于DQN的49种游戏解决方案。
- [Hasselt、Guez、Silver（2016）。《利用双Q学习进行深度强化学习》](https://arxiv.org/abs/1509.06461) —— DDQN算法的相关研究。
- [Wang等人（2016）。《对抗网络架构》](https://arxiv.org/abs/1511.06581) —— 对抗式DQN架构的研究。
- [Hessel等人（2018）。《Rainbow：深度强化学习中的多项改进整合》](https://arxiv.org/abs/1710.02298) —— 该论文汇总了多种用于提升深度强化学习性能的技巧。
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) —— 清晰现代化的DQN技术阐述文档。
- [Sutton与Barto（2018）。第9章 —— 《基于近似的在线预测》](http://incompleteideas.net/book/RLbook2020.pdf) —— 教科书中对“致命三要素”（函数近似 + 自举法 + 离线策略）的讲解，DQN的目标网络与回放缓冲区正是为了解决这些问题而设计的。
- [CleanRL DQN实现代码](https://docs.cleanrl.dev/rl-algorithms/dqn/) —— 用于消融实验的单文件版DQN实现；可结合本课程自实现的版本一同学习。
