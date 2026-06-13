# 仿真到现实迁移（Sim-to-Real）

> 一个在仿真器里训练却在硬件上失败的策略，是一个记住了仿真器的策略。域随机化、域适应和系统辨识，是让学到的控制器跨越现实鸿沟的三个工具。

**类型：** 学习
**语言：** Python
**前置要求：** Phase 9 · 08（PPO）、Phase 2 · 10（偏差/方差）
**所需时间：** 约 45 分钟

## 问题所在

训练一个真实机器人既慢、又危险、又昂贵。一个双足机器人要花数百万个训练回合才学会走路；一个真实的双足机器人哪怕只摔倒一次都会损坏硬件。仿真给你无限次的重置、确定性的可复现性、并行的环境，以及零物理损伤。

但仿真器是错的。轴承的摩擦比 MuJoCo 模型里的更多。摄像头有仿真器未纳入的镜头畸变。电机有 99% 的仿真模型都跳过的延迟、回程间隙和饱和。风、灰尘和变化的光照会破坏一个在无菌渲染下训练出的策略。**现实鸿沟**——仿真分布与真实分布之间的系统性差异——是部署型机器人 RL 的核心问题。

你需要一个对*仿真到现实分布偏移鲁棒*的策略。历史上有三种方法：随机化仿真器（域随机化）、用少量真实数据适应策略（域适应 / 微调），或辨识真实系统的参数并使之匹配（系统辨识）。在 2026 年，主流配方将这三者与大规模并行仿真（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）结合起来。

## 概念说明

![三种仿真到现实范式：域随机化、适应、系统辨识](../assets/sim-to-real.svg)

**域随机化（DR）。** Tobin 等人 2017，Peng 等人 2018。在训练期间，随机化每一个在真实机器人上可能不同的仿真参数：质量、摩擦系数、电机 PD 增益、传感器噪声、摄像头位置、光照、纹理、接触模型。策略学到的是一个关于"它今天身处哪个仿真"的条件分布，并在整个跨度上泛化。如果真实机器人落在训练包络内，策略就能奏效。

- **优点：** 无需真实数据。一个配方，适用于多种机器人。
- **缺点：** 过度随机化的训练会产出一个"通用"但过分谨慎的策略。噪声太多 ≈ 正则化太多。

**系统辨识（SI）。** 在训练之前把仿真器的参数拟合到真实世界数据上。如果你能测量真实机器人上的臂关节摩擦，就把它代入仿真。然后训练一个期望这些数值的策略。需要访问真实系统，但能直接缩小现实鸿沟。

- **优点：** 精确、低噪声的训练目标。
- **缺点：** 残余的模型误差对策略不可见；未被辨识的小效应（例如电机死区）仍会破坏部署。

**域适应。** 在仿真中训练，再用少量真实数据微调。两种风味：

- **Real2Sim2Real：** 用真实 rollout 学一个残差仿真器 `f(s, a, z) - f_sim(s, a)`，在校正后的仿真中训练。不需要太多真实数据就能弥合鸿沟。
- **观测适应：** 训练一个策略，通过一个学到的特征提取器（例如 GAN 像素到像素）把真实观测映射为类仿真的观测。控制器留在仿真里。

**特权学习 / 师生法。** Miki 等人 2022（ANYmal 四足机器人）。在仿真中训练一个能访问特权信息（真实摩擦、地形高度、IMU 漂移）的*教师*。蒸馏出一个只看到真实传感器观测的*学生*。学生学会从历史中推断特权特征，对物理参数变化鲁棒。

**大规模并行仿真。** 2024–2026 年。Isaac Lab、Mujoco MJX、Brax 都能在单个 GPU 上运行数千个并行机器人。配 4,096 个并行人形机器人的 PPO 能在数小时内采集到相当于数年的经验。随着训练分布的拓宽，"现实鸿沟"在收缩；当这 4,096 个环境各自拥有不同的随机化参数时，DR 几乎成了免费的。

**真实世界的 2026 年配方（四足行走示例）：**

1. 配以域随机化的重力、摩擦、电机增益、负载的大规模并行仿真。
2. 用特权信息（地形图、机体速度真值）训练的教师策略。
3. 仅用本体感受（腿关节编码器）从教师蒸馏出的学生策略。
4. 可选：通过对真实 IMU 的自编码器做观测适应。
5. 部署。在 10 多个环境中零样本运行。如果失败，就用安全约束的 PPO 做几分钟的真实世界微调。

## 开始构建

本课的代码是域随机化在一个带*噪声*转移的 GridWorld 上的微型演示。我们训练一个策略，让它在"仿真"中体验随机化的打滑概率，并在一个它训练时从未见过的打滑水平的"现实"上评估。其形态直接对应于 MuJoCo 到硬件的迁移。

### 第 1 步：参数化仿真

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是仿真器暴露出的一个参数。在真实机器人里它可能是摩擦、质量、电机增益——任何在仿真与现实之间发生偏移的东西。

### 第 2 步：用 DR 训练

在每个回合开始时，采样 `slip ~ Uniform[0.0, 0.4]`。训练 PPO / Q-learning / 任何算法。这样做许多回合。

### 第 3 步：在"真实"打滑上做零样本评估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上评估。前四个在训练支撑集内；`0.5` 和 `0.7` 在外。一个 DR 训练的策略应在支撑集内保持接近最优，在支撑集外优雅地退化。一个固定打滑训练的策略在其训练打滑之外会很脆弱。

### 第 4 步：与窄域训练作比较

训练第二个策略，只用 `slip = 0.0`。在相同的 `slip` 扫描上评估。一旦真实打滑 > 0，你应当看到灾难性的下降。

## 常见陷阱

- **随机化太多。** 在 `slip ∈ [0, 0.9]` 上训练，你的策略会过分规避风险以至于从不尝试最优路径。要匹配*预期*的真实世界分布，而不是"什么都可能发生"。
- **随机化太少。** 在一个窄切片上训练，策略根本无法泛化。使用自适应课程（自动域随机化），随着策略改进而拓宽分布。
- **参数空间辨识错误。** 随机化了错误的东西（在真实鸿沟是电机延迟时却去随机化摄像头色调），DR 就帮不上忙。先对真实机器人做剖析。
- **特权信息泄漏。** 一个用全局状态而非仅用观测来决定动作的教师，会产出一个学生无法追赶的策略。要确保给定观测历史，教师的策略对学生是可实现的。
- **仿真到仿真迁移失败。** 如果你的策略对一个更难的仿真变体不鲁棒，它对真实世界也不会鲁棒。部署前务必在一个留出的仿真变体上测试。
- **没有真实世界的安全包络。** 一个在仿真里能用、在现实里也"能用"但没有底层安全护盾的策略，仍可能损坏硬件。在一个非学习型控制器中加入速率限制、扭矩限制、关节限制。

## 实际应用

2026 年的仿真到现实技术栈：

| 领域 | 技术栈 |
|--------|-------|
| 足式运动（ANYmal、Spot、人形） | Isaac Lab + DR + 特权教师 / 学生 |
| 操作（灵巧手、抓取放置） | Isaac Lab + DR + 用于视觉的 DR-GAN |
| 自动驾驶 | CARLA / NVIDIA DRIVE Sim + DR + 真实微调 |
| 无人机竞速 | RotorS / Flightmare + DR + 在线适应 |
| 手指/手内操作 | OpenAI Dactyl（前所未有规模的 DR） |
| 工业机械臂 | MuJoCo-Warp + SI + 少量真实微调 |

对各种规模的控制而言，工作流是一致的：尽你所能拟合仿真，随机化你拟合不了的部分，训练庞大的策略，蒸馏，配安全护盾部署。

## 交付产物

保存为 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 练习

1. **简单。** 在固定打滑的 GridWorld（slip=0.0）上训练一个 Q-learning 智能体。在 slip ∈ {0.0, 0.1, 0.3, 0.5} 上评估。绘制回报随 slip 变化的曲线。
2. **中等。** 训练一个 DR Q-learning 智能体，采样 `slip ~ Uniform[0, 0.3]`。在相同扫描上评估。在 slip=0.5（分布外）处 DR 带来了多大收益？
3. **困难。** 实现一个课程：从 slip=0.0 开始，每当策略达到最优的 90% 时就拓宽 DR 范围。测量达到 slip=0.3 零样本所需的总环境步数，并与一个固定 DR 基线作比较。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| 现实鸿沟 | "仿真到现实的差异" | 训练与部署的物理/传感之间的分布偏移。 |
| 域随机化（DR） | "跨随机仿真训练" | 在训练期间随机化仿真参数，使策略泛化。 |
| 系统辨识（SI） | "测量真实并拟合仿真" | 估计真实物理参数；把仿真设置成匹配的值。 |
| 域适应 | "在真实数据上微调" | 仿真训练后的少量真实世界微调；可适应观测或动力学。 |
| 特权信息 | "给教师的真值" | 只有仿真才有的信息；学生必须从观测历史中推断它。 |
| 师生法 | "把特权蒸馏成可观测" | 教师用捷径训练；学生学着在没有捷径的情况下模仿。 |
| ADR | "自动域随机化" | 随策略改进而拓宽 DR 范围的课程。 |
| Real2Sim | "用真实数据弥合鸿沟" | 学一个残差，让仿真模仿真实 rollout。 |

## 延伸阅读

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) —— 最初的 DR 论文（机器人视觉）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) —— 用于动力学、四足运动的 DR。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) —— Dactyl，大规模的 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) —— 用于 ANYmal 的师生法。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) —— 驱动 2025–2026 年部署的大规模并行仿真。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) —— ADR 课程方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) —— Dyna 框架（用模型做规划 + rollout），它支撑着现代仿真到现实流水线。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) —— 仿真到现实方法的分类法，附基准结果。
