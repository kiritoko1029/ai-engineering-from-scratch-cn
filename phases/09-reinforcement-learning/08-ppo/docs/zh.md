# 近端策略优化（PPO）

> A2C 在每次更新后就丢弃整批 rollout。PPO 用一个裁剪过的重要性比率把策略梯度包裹起来，让你能在同一批数据上做 10 个以上的训练周期而不会让策略崩溃。出自 Schulman 等人（2017）。在 2026 年它仍然是默认的策略梯度算法。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 9 · 06（REINFORCE）、Phase 9 · 07（Actor-Critic）
**所需时间：** 约 75 分钟

## 问题所在

A2C（第 07 课）是 on-policy 的：梯度 `E_{π_θ}[A · ∇ log π_θ]` 要求数据采样自*当前*的 `π_θ`。一旦做了一次更新，`π_θ` 就变了；你刚才用过的数据如今变成了 off-policy 的。再拿来复用，你的梯度就有偏差。

Rollout 的代价很高。在 Atari 上，跨 8 个环境 × 128 步 = 1024 个 transition 的一次 rollout 要花掉十几秒的环境时间。在一次梯度步后就把它丢掉太浪费了。

信赖域策略优化（TRPO，Schulman 2015）是第一个修复方案：约束每次更新，使新旧策略之间的 KL 散度保持在 `δ` 以下。理论上很干净，但每次更新都需要做一次共轭梯度求解。2026 年没人再跑 TRPO 了。

PPO（Schulman 等人 2017）用一个简单的裁剪目标取代了硬性的信赖域约束。只多了一行代码。每批 rollout 训练十个周期。不用共轭梯度。理论保证足够好。九年过去了，从 MuJoCo 到 RLHF，它依然是一切场景下默认的策略梯度算法。

## 概念说明

![PPO 裁剪代理目标：比率在 1 ± ε 处被裁剪](../assets/ppo.svg)

**重要性比率。**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

这是新策略相对于采集数据的那个策略的似然比。`r_t = 1` 表示没有变化。`r_t = 2` 表示新策略采取 `a_t` 的概率是旧策略的两倍。

**裁剪代理目标。**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

两项：

- 如果优势 `A_t > 0` 且比率试图增长超过 `1 + ε`，裁剪会把梯度压平——不要把一个好动作推到比旧概率高出 `+ε` 以上。
- 如果优势 `A_t < 0` 且比率试图增长超过 `1 - ε`（意味着相对于其裁剪后的下降，我们会让一个坏动作变得更可能），裁剪会给梯度封顶——不要把一个坏动作压到低于 `-ε`。

`min` 处理另一个方向：如果比率朝*有利*方向移动，你仍然能拿到梯度（在会伤害你的那一侧才裁剪）。

典型取值 `ε = 0.2`。把目标画成关于 `r_t` 的函数：一个分段线性函数，在"好的一侧"有一个平顶，在"坏的一侧"有一个平底。

**完整的 PPO 损失。**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

与 A2C 相同的 actor-critic 结构。三个系数，通常取 `c_v = 0.5`、`c_e = 0.01`、`ε = 0.2`。

**训练循环。**

1. 在 `N` 个并行环境上各采集 `T` 步，得到 `N × T` 个 transition。
2. 计算优势（GAE），把它们冻结为常量。
3. 把 `π_{θ_old}` 冻结为当前 `π_θ` 的一份快照。
4. 进行 `K` 个周期，对每个 `(s, a, A, V_target, log π_old(a|s))` 的小批量：
   - 计算 `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`。
   - 应用 `L^{CLIP}` + 价值损失 + 熵。
   - 做一步梯度更新。
5. 丢弃这批 rollout。返回第 1 步。

`K = 10`、小批量为 64 是一组标准的超参数。PPO 很鲁棒：在 ±50% 范围内，具体数值很少会有影响。

**KL 惩罚变体。** 原论文提出了一个使用自适应 KL 惩罚的替代方案：`L = L^{PG} - β · KL(π_θ || π_old)`，其中 `β` 根据观测到的 KL 进行调整。裁剪版本成为主流；而 KL 变体在 RLHF 中得以存续（那里相对于参考策略的 KL 本就是一个你总是想要的独立约束）。

## 开始构建

### 第 1 步：在 rollout 时刻捕获 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照在 rollout 时刻只取一次。它在更新周期期间不会改变。

### 第 2 步：计算 GAE 优势（第 07 课）

与 A2C 相同。在整个批次上做归一化。

### 第 3 步：裁剪代理目标更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

"被裁剪 → 零梯度"这一模式是 PPO 的核心。如果新策略在有利方向上已经漂移得太远，更新就停止。

### 第 4 步：价值与熵

与 A2C 一样，对 critic 目标加上标准的 MSE，对 actor 加上一个熵奖励项。

### 第 5 步：诊断指标

每次更新都要关注三件事：

- **平均 KL** `E[log π_old - log π_θ]`。应保持在 `[0, 0.02]`。如果它冲过 `0.1`，就降低 `K_EPOCHS` 或 `LR`。
- **裁剪比例**——比率落在 `[1-ε, 1+ε]` 之外的样本所占的比例。应为 `~0.1-0.3`。若为 `~0`，说明裁剪从未触发 → 提高 `LR` 或 `K_EPOCHS`。若为 `~0.5+`，说明你在过拟合这批 rollout → 把它们调低。
- **解释方差** `1 - Var(V_target - V_pred) / Var(V_target)`。Critic 质量的度量。随着 critic 学习，它应朝 1 攀升。

## 常见陷阱

- **裁剪系数调错。** `ε = 0.2` 是事实标准。调到 `0.1` 会让更新过于胆怯；`0.3+` 则招致不稳定。
- **周期数过多。** `K > 20` 经常导致不稳定，因为策略偏离 `π_old` 太远。要给周期数设上限，对大网络尤其如此。
- **没有奖励归一化。** 大尺度的奖励会侵蚀裁剪范围。在计算优势之前先对奖励做归一化（运行标准差）。
- **忘记优势归一化。** 按批次做零均值/单位标准差归一化是标准做法。在大多数基准上跳过它会毁掉 PPO。
- **学习率未衰减。** PPO 受益于线性衰减到零的学习率。恒定学习率往往更糟。
- **重要性比率算错。** 为了数值稳定，始终用 `exp(log_new - log_old)`，而不是 `new / old`。
- **梯度符号错误。** 最大化代理目标 = *最小化* `-L^{CLIP}`。符号写反是最常见的 PPO bug。

## 实际应用

PPO 是 2026 年默认的 RL 算法，覆盖的领域多得令人惊讶：

| 用例 | PPO 变体 |
|----------|-------------|
| MuJoCo / 机器人控制 | 采用高斯策略的 PPO，GAE(0.95) |
| Atari / 离散游戏 | 采用类别分布策略的 PPO，滚动 128 步的 rollout |
| LLM 的 RLHF | 带有对参考模型 KL 惩罚的 PPO，奖励来自响应末尾的 RM |
| 大规模游戏智能体 | IMPALA + PPO（AlphaStar、OpenAI Five） |
| 推理型 LLM | GRPO（第 12 课）——不带 critic 的 PPO 变体 |
| 仅有偏好数据 | DPO——PPO+KL 的闭式坍缩，无需在线采样 |

PPO 的*损失形态*——裁剪代理目标 + 价值 + 熵——是 DPO、GRPO 以及几乎所有 RLHF 流水线的脚手架。

## 交付产物

保存为 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 练习

1. **简单。** 在 4×4 GridWorld 上用 `ε=0.2, K=4` 运行 PPO。在相同环境步数下，将其样本效率与 A2C（每批 rollout 只训练一个周期）作比较。
2. **中等。** 扫描 `K ∈ {1, 4, 10, 30}`。绘制回报随环境步数变化的曲线，并跟踪每次更新的平均 KL。在这个任务上，`K` 取到多少时 KL 会爆炸？
3. **困难。** 用一个自适应 KL 惩罚替换裁剪代理目标（若 `KL > 2·target` 则 `β` 翻倍，若 `KL < target/2` 则减半）。比较最终回报、稳定性以及无裁剪程度。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 重要性比率 | "r_t(θ)" | `π_θ(a\|s) / π_old(a\|s)`；相对于采集数据的策略的偏离量。 |
| 裁剪代理目标 | "PPO 的主要技巧" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`；在有利一侧越过裁剪点后梯度变平。 |
| 信赖域 | "TRPO / PPO 的意图" | 限制每次更新的 KL，以保证单调改进。 |
| KL 惩罚 | "软信赖域" | PPO 的另一种形式：`L - β · KL(π_θ \|\| π_old)`。自适应的 `β`。 |
| 裁剪比例 | "裁剪触发的频率" | 诊断指标——应为 0.1-0.3；超出范围说明调参不当。 |
| 多周期训练 | "数据复用" | 对每批 rollout 训练 K 个周期；用方差代价换取样本效率。 |
| 近似 on-policy | "大体上 on-policy" | PPO 名义上是 on-policy 的，但 K>1 个周期会安全地使用略微 off-policy 的数据。 |
| PPO-KL | "另一种 PPO" | KL 惩罚变体；用于 RLHF，那里对参考策略的 KL 本就是一个约束。 |

## 延伸阅读

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) —— 原论文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) —— TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) —— 对每个 PPO 超参数做了消融实验。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) —— InstructGPT；RLHF 中的 PPO 配方。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) —— 配 PyTorch 的现代清晰讲解。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) —— 许多论文使用的单文件 PPO 参考实现。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) —— 在语言模型上跑 PPO 的生产级配方；与第 09 课（RLHF）一同阅读。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) —— "37 个代码层面优化"那篇论文；哪些 PPO 技巧是关键的、哪些只是民间传说。
