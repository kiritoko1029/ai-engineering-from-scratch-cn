# 直接偏好优化家族

> Rafailov 等人（2023）证明 RLHF 的最优解在偏好数据上具有封闭形式，因此你可以跳过显式奖励模型，直接优化策略。这一洞见催生了一个家族——IPO、KTO、SimPO、ORPO、BPO——每一个都修复了 DPO 的一种失败模式。到 2026 年，直接对齐算法在前沿后训练运行中的使用量超过了 PPO。但第 2 课中的过度优化曲线仍然适用：DAA 并没有逃脱古德哈特，只是改变了它咬的位置。

**类型：** 学习
**语言：** Python（标准库，六变体偏好损失比较器）
**前置要求：** 第 18 阶段 · 01（InstructGPT），第 18 阶段 · 02（奖励黑客），第 10 阶段 · 08（DPO 基础）
**所需时间：** 约75分钟

## 学习目标

- 从带 KL 的 RLHF 最优解推导 DPO 的封闭形式。
- 说明 IPO、KTO、SimPO、ORPO、BPO 各自修复了 DPO 的哪种失败模式。
- 区分"隐式奖励差距"和"偏好强度"，解释为什么 IPO 的恒等映射很重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）证明 DAA 尽管没有显式 RM 仍会过度优化。

## 问题所在

RLHF 目标（第 1 课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此奖励由最优策略与参考策略的比率隐式定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将其代入 Bradley-Terry 偏好似然，配分函数 `Z(x)` 消去了，因为它只依赖于 `x`。剩下的就是仅关于策略参数的损失——不需要奖励模型。这就是 DPO。

问题是：推导假设最优解可达、偏好数据在分布内、参考策略是真实的模态锚点。这些都不完全成立。每个家族成员修复一个不同的被违反假设。

## 概念说明

### DPO（Rafailov 等，2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出错的地方：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。微小的偏好可以产生任意大的差距。
- 损失驱动被选择和被拒绝的对数概率向相反方向移动。它可以将被选择的绝对对数概率压低，只要被拒绝的下降更快。这就是"退化的被选择回答"现象。
- 分布外偏好（罕见对 vs 罕见对）产生任意的隐式奖励。

### IPO（Azar 等，2024）

恒等偏好优化用偏好概率上的恒等映射替代了 log-sigmoid。损失变成了有界目标上的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边际被 `1/(2 beta)` 约束。偏好强度和隐式奖励差距成正比。不会爆炸。

### KTO（Ethayarajh 等，2024）

Kahneman-Tversky 优化完全放弃了成对结构。给定单个标注输出和二元的"理想"或"非理想"信号，它映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失使用不同的权重（损失厌恶）。好处：你可以使用未配对的数据，这种数据丰富得多。

### SimPO（Meng 等，2024）

简单偏好优化将训练信号与生成对齐。完全移除参考策略，并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

带有稳定边际 `gamma`。长度归一化消除了利用 DPO 长度偏差失败模式的动机（更长的 `y_w` 在构造上给出更大的对数概率差距）。

### ORPO（Hong 等，2024）

赔率比偏好优化在标准 SFT 负对数似然上添加一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略——SFT 项就是正则化器。从基础模型到对齐模型进行单阶段训练。不需要单独的 SFT 检查点。

### BPO（ICLR 2026 投稿，OpenReview id=b97EwMUWu7）

识别出"退化的被选择回答"问题：DPO 保留了排序 `y_w > y_l`，但 `y_w` 的绝对对数概率可能下降。BPO 添加了一行修正，惩罚被选择回答的对数概率下降。报告在 Llama-3.1-8B-Instruct 上数学推理相比 DPO 提升了 10.1% 准确率。

### 普遍结论：DAA 仍然会过度优化

Rafailov 等人的"Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms"（NeurIPS 2024）在多个数据集和 KL 预算上使用 DPO、IPO、SLiC 训练策略。真实奖励与 KL 的曲线有与 Gao 等人相同的峰值然后崩溃形状。隐式奖励在训练期间查询分布外样本；KL 正则化无法稳定这一点。

DAA 并没有逃脱古德哈特。它们将古德哈特咬的表面从"奖励模型过度优化"变为"参考策略比率过度优化"。通用的修复方法——更好的数据、集成、早停——对两者都适用。

### 如何选择（2026 年）

- 如果你有大量配对偏好数据：使用保守 beta 的 DPO，如果长度偏差明显则用 SimPO。
- 如果你有未配对的二元反馈：KTO。
- 如果你想要从基础模型的单阶段流程：ORPO。
- 如果你在 DPO 日志中看到退化的被选择对数概率：BPO。
- 如果偏好强度变化很大且 DPO 在饱和：IPO。

每个实验室在一组任务上运行所有五个方法，然后为每个任务选择赢家。没有理由认为数学推理和安全的最优选择是相同的。

```figure
dpo-margin
```

## 开始构建

`code/main.py` 在玩具偏好数据集上比较六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO），其中真实偏好强度因对而异。每个损失使用相同的小型 softmax 策略在相同的 500 对样本上优化。绘制每种方法的最终胜率、被选择对数概率漂移和隐式奖励分布。

## 交付产出

本课程生成 `outputs/skill-preference-loss-selector.md`。给定数据集统计（配对 vs 未配对、可变 vs 均匀偏好强度、长度分布）和目标（单阶段或 SFT 后接偏好），推荐一个偏好损失并报告它所防护的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终被选择对数概率下降。BPO 应该保留更高的被选择绝对概率——验证这一点。

2. 修改偏好数据使所有对具有相等的强度。六种方法中最鲁棒的是哪种？哪种退化了？解释 IPO 在此处的优势。

3. 使被拒绝的回答平均比被选择的长 2 倍。不做其他更改，用数值展示 DPO 的长度利用和 SimPO 的修复。

4. Rafailov 等人（NeurIPS 2024）声称 DAA 会过度优化。复现单点版本：绘制被选择减被拒绝的 KL 散度，观察 DPO 在大 beta 下的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 对 DPO 添加的一行修正。对照 `code/main.py` 中的实现确认。

## 关键术语

| 术语 | 人们的说法 | 实际含义 |
|------|-----------|---------|
| DPO | "没有奖励模型的 RLHF" | 从 RLHF 封闭形式最优解推导的损失；仅涉及策略参数 |
| 隐式奖励 | "对数比率" | `beta * log(pi(y\|x) / pi_ref(y\|x))`——DPO 隐含的奖励 |
| IPO | "有界的 DPO" | 用恒等映射替代 log-sigmoid；隐式奖励差距被 `1/(2 beta)` 约束 |
| KTO | "未配对的 DPO" | 前景理论效用上的单标签，带有损失厌恶 |
| SimPO | "无参考的 DPO" | 长度归一化的对数似然 + 边际；没有参考策略 |
| ORPO | "单阶段 DPO" | NLL + 赔率比偏好项；从基础模型一次训练完成 |
| BPO | "保护被选择的 DPO" | DPO 加上对降低被选择回答绝对对数概率的惩罚 |
| 退化的被选择 | "被选择的下降了" | DPO 降低被选择的对数概率，只要被拒绝的下降更快 |
| DAA | "直接对齐算法" | 任何跳过显式 RM 的偏好损失方法 |

## 延伸阅读

- [Rafailov 等 — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar 等 — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036) — IPO
- [Ethayarajh 等 — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等 — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
