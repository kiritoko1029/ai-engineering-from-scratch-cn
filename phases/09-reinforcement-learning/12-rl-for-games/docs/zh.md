# 游戏中的 RL —— AlphaZero、MuZero 与 LLM 推理时代

> 1992 年：TD-Gammon 用纯 TD 在西洋双陆棋上击败了人类冠军。2016 年：AlphaGo 击败李世石。2017 年：AlphaZero 从零开始在国际象棋、将棋和围棋上称霸。2024 年：DeepSeek-R1 证明了同一个配方——用 GRPO 取代 PPO——在推理上也行得通。游戏是驱动本阶段每一次突破的基准。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 9 · 05（DQN）、Phase 9 · 08（PPO）、Phase 9 · 09（RLHF）、Phase 9 · 10（MARL）
**所需时间：** 约 120 分钟

## 问题所在

游戏拥有 RL 想要的一切。干净的奖励（输/赢）。无限的回合（自我对弈可重置）。完美的仿真（游戏*本身*就是仿真器）。离散或小型连续动作空间。强迫对抗鲁棒性的多智能体结构。

而游戏正是每一次重大 RL 突破被检验的方式。TD-Gammon（西洋双陆棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（《星际争霸 II》，2019）。MuZero（学到的模型，2019）。AlphaTensor（矩阵乘法，2022）。AlphaDev（排序算法，2023）。DeepSeek-R1（数学推理，2025）——最新一例证明，游戏 RL 技术在文本上也奏效。

这个 capstone 通过一个统一的视角——**自我对弈 + 搜索 + 策略改进**——纵览三个里程碑式的架构：AlphaZero、MuZero 和 GRPO。每一个都对前一个做了推广；尤其是 GRPO，它是 AlphaZero 配方应用于 LLM 推理的产物，以 token 作为动作、以数学验证作为获胜信号。

## 概念说明

![AlphaZero ↔ MuZero ↔ GRPO：相同的循环，不同的环境](../assets/rl-games.svg)

**统一的循环。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017）。** Silver 等人。给定一个规则已知的游戏（国际象棋、将棋、围棋）：

- 策略-价值网络：一座塔 `f_θ(s) → (p, v)`。`p` 是合法走子上的先验。`v` 是期望的对局结果。
- 蒙特卡洛树搜索（MCTS）：在每一步走子时，展开一棵可能延续的树。用 `(p, v)` 作为先验 + bootstrap。按 UCB（PUCT）选择节点：`a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`。
- 自我对弈：让智能体对阵智能体下棋。在第 `t` 步走子时，MCTS 的访问分布 `π_t` 成为策略的训练目标。
- 损失：`L = (v - z)² - π · log p + c · ||θ||²`。`z` 是对局结果（+1 / 0 / -1）。

零人类知识。零手工启发式。一个单一的配方，在各下了几千万局自我对弈后掌握了国际象棋、将棋和围棋。

**MuZero（2019）。** Schrittwieser 等人。移除了规则必须已知这一要求。

- 不再使用固定环境，而是学一个*潜在动力学模型* `(h, g, f)`：
  - `h(s)`：把观测编码为潜在状态。
  - `g(s_latent, a)`：预测下一个潜在状态 + 奖励。
  - `f(s_latent)`：预测策略先验 + 价值。
- MCTS 在*学到的潜在空间*中运行。相同的搜索，相同的训练循环。
- 在围棋、国际象棋、将棋*以及* Atari 上都奏效——一个算法，无需规则知识。

**随机 MuZero（2022）。** 加入了随机动力学和机会节点；扩展到西洋双陆棋这一类游戏。

**Muesli、Gumbel MuZero（2022-2024）。** 在样本效率和确定性搜索上的改进。

**GRPO（2024-2025）。** DeepSeek-R1 配方。同样的 AlphaZero 形态循环，应用于语言模型推理：

- "游戏"：回答一道数学 / 编程 / 推理题。"获胜" = 验证器（测试用例通过、数值答案匹配）返回 1。
- 策略：LLM。动作：token。状态：提示 + 已生成的回答。
- 没有 critic（PPO 风格的 V_φ）。而是对每个提示从策略中采样 `G` 个补全。为每个计算奖励。用**组相对优势** `A_i = (r_i - mean_r) / std_r` 作为 REINFORCE 风格更新的信号。
- 对参考策略的 KL 惩罚以防漂移（类似 RLHF）。
- 完整损失：

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

无奖励模型，无 critic，无 MCTS。组相对基线取代了这三者。在推理基准上，它以 PPO-RLHF 一小部分的算力达到或超越其质量。

**完整的 R1 配方。** DeepSeek-R1（DeepSeek 2025）在一篇论文里给出了两个模型：

- **R1-Zero。** 从 DeepSeek-V3 基座模型出发。无 SFT。直接用 GRPO，配两个奖励分量：*准确性奖励*（基于规则——最终答案是否解析为正确数字 / 代码是否通过单元测试）和*格式奖励*（补全是否把思维链包裹在 `<think>…</think>` 标签中）。经过数千步，平均回答长度从约 100 增长到约 10,000 个 token，数学基准分数攀升至接近 o1-preview 水平。模型从零开始学会推理。缺点：它的思维链常常不可读、混杂语言，且缺乏文风上的打磨。
- **R1。** 用一个四阶段流水线修复 R1-Zero 的可读性问题：
  1. **冷启动 SFT。** 收集几千条格式干净的长思维链示范。在它们上对基座模型做监督微调。这给出一个可读的起点。
  2. **面向推理的 GRPO。** 应用 GRPO，配以准确性+格式奖励，外加一个*语言一致性*奖励以防语码转换。
  3. **拒绝采样 + 第二轮 SFT。** 从 RL 检查点采样约 60 万条推理轨迹，只保留那些最终答案正确且思维链可读的，并与约 20 万条非推理 SFT 样本（写作、问答、自我认知）结合。再次微调基座。
  4. **全谱 GRPO。** 再进行一轮 RL，同时覆盖推理（基于规则的奖励）和通用对齐（有帮助性/无害性的基于偏好的奖励）。

其结果在 AIME 和 MATH-500 上以开放权重匹配 o1，且小到足以蒸馏。同一篇论文还通过在 R1 的推理轨迹上做 SFT，发布了六个蒸馏出的稠密模型（从 Qwen-1.5B 到 Llama-70B）——学生端没有 RL。在学生的规模上，蒸馏一个强大的 RL 教师，始终胜过从零开始做 RL。

**推理为何用 GRPO 而非 PPO。** DeepSeekMath 论文（2024 年 2 月）给出三个理由：(1) 无价值网络需要训练，内存减半；(2) 组基线天然处理推理任务产生的稀疏的轨迹末端奖励；(3) 按提示归一化使得难度迥异的题目之间的优势可以相互比较，而 PPO 单一的 critic 做不到。

**无搜索 vs 基于搜索。** 游戏出现了分叉：

- *长程的完美信息游戏*（围棋、国际象棋）：仍然基于搜索。AlphaZero / MuZero 占主导。
- *LLM 推理*：生产中尚无 MCTS；在完整 rollout 上用 GRPO，推理时用 best-of-N 算力。过程奖励模型（PRM）暗示着步骤级搜索正被重新加回来。

## 开始构建

`code/main.py` 中的代码实现了**微缩版 GRPO**——一个带多组样本的赌博机（bandit）。算法与在 LLM 上的相同；只是策略和环境更简单。它教的是*损失*和*组相对优势*，这正是 2025 年的创新点。

### 第 1 步：一个微型验证器环境

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真实的 GRPO 中，验证器会运行单元测试或检查数学相等性。

### 第 2 步：策略：对每个提示的 K 个答案 token 做 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等价于一个以提示为条件的 LLM 的最后一层输出。

### 第 3 步：组采样与组相对优势

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

组相对优势是 2024 年 DeepSeek 的技巧。无需 critic。"基线"是组均值，归一化用的是组标准差。

### 第 4 步：与 REINFORCE 基线（无价值）作比较

相同设置，相同算力，普通 REINFORCE。GRPO 收敛得更快、更稳定。

### 第 5 步：观察熵和 KL

与 RLHF 相同的诊断指标：对参考的平均 KL、策略熵、奖励随时间的变化。一旦这些稳定下来，训练就完成了。

## 常见陷阱

- **通过愚弄验证器进行奖励作弊。** GRPO 继承了 RLHF 的风险：如果验证器有错或可被利用，LLM 就会找到那个漏洞。鲁棒的验证器（多个测试用例、形式化证明）很重要。
- **组规模太小。** 组基线的方差随 `1/√G` 变化。低于 `G = 4` 时，优势信号是嘈杂的；标准选择是 `G = 8` 到 `64`。
- **长度偏差。** 不同长度的 LLM 补全有不同的对数概率。按 token 数归一化，或用序列级对数概率，或截断到最大长度。
- **纯自我对弈循环。** AlphaZero 风格的训练在一般和博弈上可能卡在支配循环里。通过多样化的对手池（联赛对战，第 10 课）来缓解。
- **搜索-策略失配。** AlphaZero 训练策略去模仿搜索输出。如果策略网络太小、无法表示搜索的分布，训练就会停滞。
- **算力门槛。** MuZero / AlphaZero 需要巨量算力。单次消融实验常常要数百 GPU 小时。存在用于学习的微缩演示（例如四子棋上的 AlphaZero）。
- **验证器覆盖度。** 对一个有 bug 的解也能通过的单元测试，会强化那个 bug。设计能捕捉边界情况的验证器。

## 实际应用

按领域划分的 2026 年游戏 RL 图景：

| 领域 | 主导方法 |
|--------|-----------------|
| 双人零和棋类（围棋、国际象棋、将棋） | AlphaZero / MuZero / KataGo |
| 不完美信息纸牌游戏（扑克） | CFR + 深度学习（DeepStack、Libratus、Pluribus） |
| Atari / 像素游戏 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略（Dota、星际争霸） | PPO + 自我对弈 + 联赛（OpenAI Five、AlphaStar） |
| LLM 数学/代码推理 | GRPO（DeepSeek-R1、Qwen-RL、开源复现） |
| LLM 对齐 | DPO / RLHF-PPO（非 GRPO；验证器是偏好而非可验证的） |
| 机器人 | PPO + DR（不算游戏 RL，但用相同的策略梯度工具） |
| 组合问题 | AlphaZero 变体（AlphaTensor、AlphaDev） |

这个*配方*——自我对弈、搜索增强的改进、策略蒸馏——横跨文本、像素和物理控制。GRPO 是最年轻的实例；还会有更多。

## 交付产物

保存为 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 练习

1. **简单。** 实现 `code/main.py` 中的 GRPO 赌博机。在 2 个提示 × 各 4 个答案 token 上训练。用 `G=8` 在 1,000 次更新内收敛。
2. **中等。** 接入 PPO（裁剪）和普通 REINFORCE。在相同的赌博机上将它们的样本效率和奖励方差与 GRPO 作比较。
3. **困难。** 扩展到一个长度为 2 的"推理链"：智能体发出两个 token，验证器奖励这一对。测量 GRPO 如何处理跨两步序列的信用分配。（提示：按*完整序列*计算组优势，再传播到两个 token 位置。）

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------------|-----------------------|
| MCTS | "带学习网络的树搜索" | 蒙特卡洛树搜索；用学到的 `(p, v)` 先验做 UCB1/PUCT 选择。 |
| AlphaZero | "自我对弈 + MCTS" | 训练成匹配 MCTS 访问次数和对局结果的策略-价值网络。 |
| MuZero | "学到模型的 AlphaZero" | 相同循环，但通过学到的动力学在潜在空间中进行。 |
| GRPO | "无 critic 的 PPO" | 组相对策略优化；带组均值基线 + KL 的 REINFORCE。 |
| PUCT | "AlphaZero 的 UCB" | `Q + c · p · √N / (1 + N_a)` —— 平衡价值估计与先验。 |
| 自我对弈 | "智能体对阵过去的自己" | 零和的标准做法；对称的训练信号。 |
| 联赛对战 | "基于种群的自我对弈" | 过去的 + 当前的 + 剥削者被采样作为对手。 |
| 验证器奖励 | "可验证 RL" | 奖励来自一个确定性的检查器（测试通过、答案匹配）。 |
| 过程奖励 | "PRM" | 对每个推理步骤打分，而不只是最终答案。 |

## 延伸阅读

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) —— 引入 GRPO 和组相对基线的论文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) —— 完整的四阶段 R1 配方外加 R1-Zero 消融。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) —— 大规模的 CFR + 深度学习。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) —— 开创这一切的论文。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) —— 用自定义奖励函数应用 GRPO 的生产级参考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) —— 在多个规模上对 R1 配方的开源复现。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) —— 教材对自我对弈、搜索和"设计奖励"的框架，R1 在 LLM 规模上将其实例化。
