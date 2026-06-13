# 异步 Hogwild! 推理

> 推测解码（第 10 阶段 · 15）在一个序列内并行化 token。多智能体框架跨整个序列并行化，但强制显式协调（投票、子任务分割）。Hogwild! 推理（Rodionov 等，arXiv:2504.06261）做了不同的事情：在共享的键值缓存上并行运行 N 个同一 LLM 的实例。每个 worker 立即看到其他 worker 生成的 token。现代推理模型——QwQ、DeepSeek-R1——可以通过该共享缓存自我协调，无需任何微调。该方法是实验性的，但它打开了一个全新的推理并行维度，与推测解码正交。本课程在标准库 Python 中实现双 worker Hogwild! 模拟器，并解释为什么共享缓存协作来自现有模型的推理能力。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 10 阶段 · 12（推理优化），第 10 阶段 · 15（推测解码）
**所需时间：** 约60分钟

## 学习目标

- 描述三种常见的并行 LLM 拓扑（投票、子任务、Hogwild!），并指出每种针对的问题。
- 说出核心 Hogwild! 设置：多个 worker、一个共享 KV 缓存、通过自提示的涌现协调。
- 从 worker 数 `N`、任务级并行度 `p` 和协调开销 `c` 的函数关系计算 Hogwild! 的时钟时间加速。
- 在玩具问题上实现双 worker Hogwild! 模拟器，观察涌现的任务分工。

## 问题所在

现代 LLM 通过产生长推理链来解决难题——5000 token 的逐步逻辑很常见，深度数学问题上出现数万 token。在 70B 模型上 35 token/秒的解码下，50k token 是 24 分钟。交互性就不存在了。

推测解码（第 10 阶段 · 15）通过在一个序列内并行化给你 3-5 倍加速。除此之外，自回归解码的顺序依赖是硬天花板。每个新 token 依赖所有先前 token。

显而易见的问题：我们能跨序列并行化吗？在同一问题上运行同一模型的多个副本，让它们合作，分配工作？

之前的投票集成（运行 N 个模型，选择多数答案）、思维树（分支推理路径并重组）和多智能体框架（给每个智能体分配子任务，使用协调器）。这些在特定任务领域都有帮助。但它们都引入了显式协调机制——投票规则、分支剪枝逻辑、智能体间消息协议。

Hogwild! 推理采用了不同的方法。N 个 worker 共享一个 KV 缓存。每个 worker 立即看到其他 worker 生成的 token，就像它们是自己的上下文一样。Worker——无需任何训练或微调——就能弄清楚如何分工。现代推理模型（QwQ、DeepSeek-R1、Claude 家族推理模式）可以读取共享缓存并说出类似"我看到 worker 2 已经处理了基础情况，所以我来做归纳步骤"的话。

加速是工作负载依赖的，截至 2026 年 4 月仍是实验性的。但这个想法值得了解，因为它打开了推理并行的新维度。

## 概念说明

### 设置

初始化 N 个 worker 进程，都运行同一 LLM。不使用每 worker 的 KV 缓存，而是维护一个共享缓存。当 worker i 生成 token `t_j` 时，token 被写入共享缓存的下一个位置。当 worker k 执行下一步时，它读取缓存的当前状态（包括所有 N 个 worker 已生成的一切）。

在步骤时间，worker 竞争写入 token。没有每 worker 的位置索引——缓存是单个增长序列。顺序由写入到达时间决定。

### 为什么协调会涌现

Worker 共享一个提示。通常是类似"你是共同解决此问题的 N 个实例之一。每个实例读取共享内存并能看到其他实例写的内容。避免重复工作。"提示加共享缓存就够了。推理模型读取缓存，注意到问题的哪些部分已被尝试，并且（通常但不总是）转向未探索的部分。

Hogwild! 论文（Rodionov 等，2025）报告了如下观察：

- Worker 制定计划并通过缓存传达给其他 worker。
- Worker 注意到其他 worker 推理中的错误并指出。
- Worker 在计划失败时适应并提出替代方案。
- 当提示检查冗余时，Worker 检测到并转向。

这些都不需要微调。涌现行为来自模型已有的推理能力。

### 命名

论文的名字戏仿了 Hogwild! SGD（Recht 等，2011），一种异步更新优化器。类比：SGD 的异步 worker 都写入共享参数向量；Hogwild! 推理的 worker 都写入共享 KV 缓存。两者都依赖经验收敛而非同步保证。

### RoPE 使这可行

旋转位置编码（RoPE，Su 等 2021）通过 Q 和 K 向量中的旋转编码位置信息。因为位置是旋转而非固定偏移，token 的位置可以移动而无需重算 KV 缓存条目。当 worker i 在位置 `p` 写入共享缓存时，读取该位置的其他 worker 可以直接使用缓存条目——无需重新旋转。

在学习式位置或绝对位置模型中，Hogwild! 每次并发写入都需要缓存失效。RoPE 让缓存保持稳定。

### 时钟时间计算

令 `T_serial` 为一个 worker 单独解决问题的时间。令 `p` 为任务级可并行比例。令 `c` 为每步协调开销（读取扩展缓存，决定写什么）。

单 worker 时间：`T_serial`。
N worker Hogwild! 时间，如果协调免费：`T_serial * ((1 - p) + p / N)`。经典 Amdahl。
考虑协调开销：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

Worker 要有生产力，`c` 必须相对于每步解码时间很小。在产生 5k+ token 的推理模型上，worker 可以承受数百 token 的协调开销仍然划算。在短聊天任务上，协调主导，Hogwild! 比串行更差。

### 具体例子

推理问题：10k token 的思维链。假设问题有 `p = 0.7` 可并行内容（不同证明策略、不同案例分析）和每 worker `c = 200` token 的协调开销。在 `N = 4` 个 worker 下：

- 串行时间：10000 解码步。
- Hogwild! 时间：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 解码步。
- 加速：10000 / 5550 = 1.8 倍。

这不算大。但在更长的推理问题（50k token）上，协调开销摊销，加速推向 2.5-3 倍。Hogwild! 是推理等效的线程级并行，在允许你自然编写多线程代码的语言中。

### 何时使用 Hogwild!

- 长推理问题（数千 token），任务可以在独立子目标间并行。
- 被训练为逐步思考的推理模型。非推理模型无法良好自我协调。
- 单节点部署，有足够显存放共享缓存加 N 个 worker 进程。缓存是共享的，但每个 worker 有自己的激活内存。

### 何时不用

- 短交互聊天。协调开销主导。
- 不可并行的任务（单一线性证明、单次编译）。N=1 是上限。
- 非推理模型。不会涌现协调。
- 多节点部署。共享缓存需要非常快的跨 worker 同步。节点内没问题；跨节点是延迟灾难。

### 实验状态

截至 2026 年 4 月，Hogwild! 是一个带有开源 PyTorch 实现的研究方法。生产采用尚未发生。三个阻碍：

1. 跨并发进程的共享 KV 缓存管理是非平凡的工程。
2. 涌现协调是任务依赖的；基准测试仍在构建中。
3. 加速相比推测解码已提供的效果是适中的，两者可以组合但组合工程是另一层。

值得了解。值得实验。还不值得押注产品。

```figure
continuous-batching
```

## 开始构建

`code/main.py` 实现了玩具 Hogwild! 模拟器：

- 两个 worker 进程，每个是确定性的"LLM"，以已知概率产生几种 token 类别之一（工作 token、观察 token、协调 token）。
- 一个共享缓存（只是一个 token 列表），两个 worker 都读写。
- 简单协调逻辑：当一个 worker 看到另一个已在某类别产生了足够的工作 token 时，选择不同类别。

模拟器运行固定步数预算并报告：

- 产出的总工作 token。
- 总时钟时间（worker 步数）。
- 相对于单 worker 的有效加速。
- 哪个 worker 写了哪个 token 的追踪。

### 第 1 步：共享缓存

两个 worker 都追加的列表。真实实现中使用简单锁（Python `threading.Lock`）；我们用计数器模拟。

### 第 2 步：worker 循环

每个 worker 在每步：

- 读取当前共享缓存。
- 基于已有内容决定写入什么类别的 token。
- 写入一个 token。

### 第 3 步：协调启发式

如果类别 X 在缓存中已有 K 个 token 且 worker 的目标类别是 X，worker 切换到类别 Y。这是推理模型"注意到这已被覆盖，做别的事"行为的玩具替代。

### 第 4 步：度量加速

用 N=1 worker 和 N=2 worker 运行模拟器，相同总步数预算。统计产出的工作 token。N=2 应产出约 1.5-1.8 倍更多的工作 token，因为协调驱动的任务分工。

### 第 5 步：压力测试协调

降低协调启发式的敏感度。再次运行。观察到没有好的协调，N=2 冗余产出相同 token，加速降至 1 以下。这与论文的观察匹配：技巧只有在 worker 有推理能力来自我协调时才有效。

## 使用它

截至 2026 年 4 月，Hogwild! 在生产中的集成是研究级的。来自 Yandex/HSE/IST 的参考实现基于 PyTorch，针对 DeepSeek-R1 和 QwQ 模型的单节点多进程设置。

务实的采用路径：

1. 分析你的推理任务工作负载。度量探索性 token（多策略、案例分析、搜索）vs 线性的比例。
2. 如果探索主导，运行双 worker Hogwild! 实验。度量时钟时间改善。
3. 如果改善低于 1.3 倍，你在协调主导区域。回退到单 worker。
4. 如果改善超过 1.5 倍，推到 N=4 并再次度量。递减回报通常在 N=4-8 左右出现。

与推测解码组合：每个 Hogwild! worker 可以独立使用推测解码。两个加速大致相乘，将 3 倍推测解码和 1.8 倍 Hogwild! 带到相对于朴素单 worker 解码的有效 5.4 倍。

## 交付它

本课程产出 `outputs/skill-parallel-inference-router.md`。给定推理工作负载画像（token 预算、任务并行画像、模型家族、部署目标），它在投票、思维树、多智能体、Hogwild! 和推测解码策略之间路由。

## 练习

1. 用默认设置运行 `code/main.py`。确认 N=2 Hogwild! 配置在相同时钟时间内产出比 N=1 基线更多的工作 token。

2. 降低协调启发式的强度（设置 `coordination_weight=0.1`）。重新运行。展示加速崩溃。解释为什么：worker 在无法协调时重复劳动。

3. 计算 50k token 推理任务 `p=0.8, c=500` 和 N=4 worker 的预期 Hogwild! 加速。对 1k token 聊天任务 `p=0.3, c=200` 和 N=4 做同样的计算。为什么一个赢一个输？

4. 阅读 Hogwild! 论文的第 4 节（初步评估）。找出作者报告的两种失败模式。描述更好的协调提示如何缓解每种。

5. 在玩具中组合 Hogwild! 和推测解码：每个 worker 内部使用 2 token 推测解码。报告乘法加速。当两个 worker 都想扩展同一共享缓存前缀时，会出现什么记账问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Hogwild! | "并行 worker，共享缓存" | N 个同一 LLM 的实例并发运行，共享一个 KV 缓存；通过自提示的涌现协调 |
| 共享 KV 缓存 | "协调介质" | 所有 worker 读写的单个增长 KV 缓冲区；使 token 在 worker 间即时可见 |
| 涌现协调 | "无需训练" | 具有推理能力的 LLM 可以读取共享缓存并分工，无需任何微调或显式协议 |
| 协调开销（c） | "用于定位的 token" | 每 worker 读取扩展缓存并决定做什么的成本；必须相对于总解码时间保持很小 |
| 可并行比例（p） | "什么可以并行运行" | 任务级并行：总工作中非固有顺序的比例 |
| RoPE 使 Hogwild! 可行 | "旋转位置是平移不变的" | 因为位置是旋转，写入共享缓存不需要重算先前 token |
| 投票集成 | "运行 N，选多数" | 最简单的并行推理拓扑；对分类有用，对长文本推理用处小 |
| 思维树 | "分支和剪枝" | 探索多分支并剪枝的推理策略；显式协调逻辑 |
| 多智能体框架 | "分配子任务" | 每个智能体获得角色；协调器编排；协议开销重 |

## 延伸阅读

- [Rodionov et al. -- Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261) -- Hogwild! 论文，在 QwQ 和 DeepSeek-R1 上的初步评估
- [Recht, Re, Wright, Niu -- Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730) -- 原始 Hogwild!，命名来源
- [Su et al. -- RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864) -- RoPE，使共享缓存推理可行的特性
- [Yao et al. -- Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) -- Hogwild! 与之正交的思维树推理策略
- [Leviathan et al. -- Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) -- 推测解码，Hogwild! 与之组合的序列内并行
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm) -- 论文实验的唯一事实来源
