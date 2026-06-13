# Jamba——混合 SSM-Transformer

> 状态空间模型（SSM）和 Transformer 想要不同的东西。Transformer 通过注意力以二次成本购买质量。SSM 通过递归以线性时间推理和恒定内存购买效率，但质量落后。AI21 的 Jamba（2024 年 3 月）和 Jamba 1.5（2024 年 8 月）将它们放在同一个模型中：每 7 个 Mamba 层配 1 个 Transformer 层，每隔一个块使用 MoE，以及一个能放进单块 80GB GPU 的 256k 上下文窗口。Mamba-3（ICLR 2026）用复数值状态空间和 MIMO 投影收紧了 SSM 一侧。本课程端到端阅读两个架构，并解释为什么混合配方在纯 SSM 和纯 Transformer 长上下文尝试失败的情况下存活了三年。

**类型：** 学习
**语言：** Python（标准库，层混合计算器）
**前置要求：** 第 10 阶段 · 14（开放模型架构），第 10 阶段 · 17（原生稀疏注意力）
**所需时间：** 约60分钟

## 学习目标

- 解释 Jamba 块中的三个原语——Transformer 层、Mamba 层、MoE——以及 1:7:偶数 的交错配方。
- 高层说出 SSM 的递归是什么样的，以及它为什么能实现恒定内存推理。
- 计算 Jamba 模型在 256k 上下文下的 KV 缓存占用，并与纯 Transformer 模型的需求对比。
- 说出 Mamba-3 的三个创新（指数梯形离散化、复数值状态更新、MIMO）以及每个针对的问题。

## 问题所在

注意力在序列长度上是二次的。状态空间模型是线性的。这个差异会复合：在 256k token 时，Transformer 注意力图每头 650 亿条目；SSM 的递归状态大小固定，与序列长度无关。

纯 SSM 模型（Mamba、Mamba-2）在小规模上匹配 Transformer 困惑度，但在状态跟踪任务上落后，在某些类别的上下文内检索上失败。直觉是：SSM 将历史压缩到固定状态，当历史很长时信息会泄漏。注意力精确记住一切但付出二次成本。

显而易见的修复：两者都用。在精确回忆重要的地方放 Transformer 层。其他地方用 SSM 层。调整比率。Jamba 是第一个大规模发布此混合配方的生产级模型（52B 总量、12B 激活、256k 上下文、单块 80GB GPU）。Jamba 1.5 将家族扩展到 398B 总量 / 94B 激活。Mamba-3（ICLR 2026）是当前最佳纯 SSM 基线，混合架构可以围绕它重建。

本课程阅读三篇论文，产出"选对比率"的心智模型。

## 概念说明

### 一页讲清 SSM

状态空间模型通过固定大小状态 `h` 处理序列 `x_1, ..., x_N`：

```
h_t = A h_{t-1} + B x_t
y_t = C h_t
```

每一步状态通过线性动力学 `A` 演化，接收输入 `B x_t`，输出 `C h_t`。`A, B, C` 可以学习。注意关键特性：计算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要更早的 `x`。内存恒定。推理每 token O(1)。

建模质量的关键是 `A` 的结构。S4（Gu 2021）使用高度结构化的矩阵，训练时可以高效地作为长卷积评估。Mamba（Gu、Dao 2023）将固定的 `A, B, C` 替换为数据依赖的（"选择性"部分）。Mamba-2（2024）进一步简化了结构。Mamba-3（2026）在特定位置重新添加复杂性。

关键特性：对于解码器 LLM，SSM 层是注意力层的直接替代品，具有固定大小的逐层状态而非增长的 KV 缓存。

### Jamba 块

Jamba 块按两个数字交错层：

- `l`：注意力与 Mamba 的比率。Jamba 使用 `l = 8`，意味着每 7 个 Mamba 层配 1 个 Transformer 层（7 Mamba + 1 Attention = 每组 8 层）。
- `e`：MoE 频率。Jamba 使用 `e = 2`，意味着每隔一层应用 MoE。

块内的层序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每个 Jamba 块是 8 层。4 个块深（共 32 层），你得到 28 个 Mamba 和 4 个 Attention 层。其中 16 个使用 MoE。

### 为什么是 1:7 比率

AI21 做了消融实验：什么比率的注意力-Mamba 在长上下文评估上给出最佳的每参数困惑度和上下文内召回？

- 太多注意力（1:1）：质量上升但内存和速度退化。
- 太少注意力（1:15）：内存很好但上下文内检索失败。
- 甜蜜点：1:7 或 1:8。

直觉：Transformer 层处理精确回忆和状态跟踪。Mamba 层处理廉价的批量处理。

### 位置编码

Mamba 层本身通过递归是位置感知的。原始基于 Mamba 的混合中的注意力层不使用 RoPE——SSM 层提供位置信息。Jamba 1.5 在注意力层中添加 RoPE 以实现更长上下文的泛化，这是基于经验长上下文评估的事后改进。

### 内存预算

对于 Jamba-1 形状（32 层：28 Mamba + 4 Attention，隐藏 4096，32 个注意力头）：

- KV 缓存（仅注意力层）：在 256k BF16 下 `2 * 4 * 32 * 128 * 256k * 2 = 8.4 GB`。只有 4 个注意力层贡献。
- SSM 状态：`28 * hidden * state_size` 每 token 前缀，但这是每层固定大小，不随序列长度缩放。典型 Mamba 状态每特征 16，隐藏 4096：`28 * 4096 * 16 * 2 = 3.7 MB` 总计。

与 32 层纯 Transformer、相同隐藏、32 头全 MHA 比较：在 256k BF16 下 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV 缓存缩减 8 倍。即使与大多数 2024 模型使用的 GQA(8) 基线（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`）相比，Jamba 的 1:7 混合在 16 GB 仍小 2 倍。

这就是 AI21 所说的"单块 80GB GPU 上 256k 上下文"的含义。全 MHA 纯 Transformer 的 KV 缓存放不下；即使 GQA 基线也没有给权重和激活留空间；Jamba 的可以。

### Mamba-3：2026 年的纯 SSM 基线

Mamba-3（ICLR 2026，arXiv:2603.15569）在纯 SSM 一侧引入三个创新：

1. **指数梯形离散化。** 用更具表达力的递归替代 Mamba-2 中的欧拉法离散化。在核心递归内对状态-输入应用类卷积操作，而非作为 `x_t` 上的外部卷积。

2. **复数值状态更新。** 之前的 Mamba 将状态矩阵从复数（S4）简化为实对角（Mamba）再简化为缩放单位阵（Mamba-2）。Mamba-3 重新添加复数值——相当于状态上的数据依赖旋转嵌入。这恢复了之前实值简化所牺牲的状态跟踪能力。

3. **多输入多输出（MIMO）投影。** 使用矩阵值投影而非逐特征标量投影。提升建模能力和推理时硬件利用率，不增加解码延迟。

在 1.5B 参数下，Mamba-3 比 Gated DeltaNet 平均下游精度提升 0.6 分；MIMO 变体再加 1.2 分，总计 1.8 分提升。在相同状态下，Mamba-3 以 Mamba-2 一半的状态匹配其效果。

Mamba-3 尚未在大规模生产混合中发布——但它是下一个 Jamba 级模型 SSM 一侧的明显候选。

### 何时使用混合

混合在以下情况胜出：

- 上下文长到纯 Transformer KV 缓存变得痛苦（64k+）。
- 任务混合短程结构（SSM 擅长）和长程召回（需要 Transformer）。
- 你想部署在单 GPU 内存预算上，纯 Transformer KV 缓存放不下。

混合在以下情况失败：

- 上下文短（16k 以下）。SSM 开销浪费；纯 Transformer 就好。
- 任务需要全对全注意力（深度推理、多文档交叉引用）。混合中注意力层的稀疏性伤害性能。
- 你在扩展到万亿参数前沿模型。纯 Transformer + MLA + MoE（DeepSeek-V3 风格）目前在能力竞赛中胜出。

### 竞争格局

| 模型 | 家族 | 规模 | 独特卖点 |
|------|------|------|---------|
| Mamba-2 | 纯 SSM | 3B | 线性时间，恒定内存 |
| Jamba | 混合 | 52B/12B | 80GB 上 256k |
| Jamba 1.5 Large | 混合 | 398B/94B | 企业级长上下文 |
| Mamba-3 | 纯 SSM | 1.5B（论文） | 状态跟踪恢复 |
| DeepSeek-V3 | 纯 Transformer + MoE | 671B/37B | 前沿能力 |

2026 年格局：纯 Transformer MoE 主导前沿，但混合拥有 256k 以上上下文的利基。Mamba-3 的状态跟踪胜利可能在下一代推动混合比率更低（更多 SSM，更少注意力）。

```figure
swiglu-ffn
```

## 使用它

`code/main.py` 是混合架构的内存计算器。给定 SSM-Transformer 比率和隐藏大小/层数配置，它计算：

- 目标上下文下的 KV 缓存。
- SSM 状态内存。
- 一系列模型形状在上下文 N 下的总内存。

计算器支持：

- 纯 Transformer 基线（KV 缓存随 N 增长）。
- Jamba 风格 1:7 混合。
- 纯 SSM（完全没有 KV 缓存）。

数字直接来自 Jamba-1 和 Jamba-1.5 论文对已发布形状的数据，以及对假设变体的外推。

真实部署的集成考量：

- 大多数生产推理服务器（vLLM、SGLang）支持 Jamba 和 Mamba。检查具体版本。
- 在 256k 上下文下，Jamba 的内存优势体现在并发请求吞吐上。相同显存下，你能放更多 Jamba 序列而非 Transformer 序列。
- Mamba-3 作为独立模型尚未在生产中发布——1.5B 的研究预览。

## 交付它

本课程产出 `outputs/skill-hybrid-picker.md`。给定工作负载规范（上下文长度画像、任务混合、内存预算），它在纯 Transformer、Jamba 风格混合和纯 SSM 之间推荐，并对内存和质量权衡给出明确推理。

## 练习

1. 运行 `code/main.py` 计算 32 层纯 Transformer（隐藏 4096，32 头）和相同形状的 Jamba-1 混合在 256k 上下文下的 KV 缓存。验证 AI21 论文声称的约 8 倍内存缩减。

2. 修改计算器建模 1:3 混合（4 Mamba : 1 Attention）和 1:15 混合（14 Mamba : 1 Attention）。绘制 KV 缓存与比率的关系。什么比率下 KV 缓存等于 SSM 状态内存？

3. 阅读 Jamba 论文（arXiv:2403.19887）第 3 节。解释为什么 AI21 使用 Mamba-1 而非 Mamba-2，尽管 Mamba-2 更快。提示：混合消融节记录了这一点。

4. 计算 Jamba 1.5 Large（398B 总量，94B 激活）中每隔一层 MoE 的参数开销。将激活比与 DeepSeek-V3（37B/671B）比较，解释为什么 Jamba 的架构推高了激活比。

5. 阅读 Mamba-3 论文（arXiv:2603.15569）第 3 节。用三句话解释为什么复数值状态更新等价于数据依赖旋转嵌入。将答案与第 7 阶段 · 第 04 课的 RoPE 推导联系起来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 状态空间模型（SSM） | "带固定状态的递归" | 带学习递归 `h_t = A h_{t-1} + B x_t` 的层；每 token 恒定内存 |
| 选择性 SSM | "Mamba 的技巧" | 数据依赖的 A、B、C 参数，在线性时间内给予模型类似门控的选择性 |
| 注意力-Mamba 比率 | "多少注意力层" | 在 Jamba 中，`l = 8` 意味着每 7 个 Mamba 层配 1 个注意力层 |
| Jamba 块 | "8 层组" | 一个注意力 + 七个 Mamba + 交替位置的 MoE |
| SSM 状态 | "隐藏缓冲区" | 替代 Mamba 层 KV 缓存的固定大小逐层状态 |
| 256k 上下文 | "Jamba 的旗舰数字" | Jamba-1 能放进单块 80GB GPU 的序列长度；纯 Transformer 在该规模下不行 |
| Mamba-3 | "2026 年纯 SSM" | 当前最佳纯 SSM 架构，带复数状态 + MIMO；混合围绕其重建的基线 |
| MIMO | "多输入多输出" | Mamba-3 创新，使用矩阵值投影而非逐特征标量 |
| 指数梯形离散化 | "Mamba-3 的递归" | 更具表达力的递归，包含 Mamba-2 的欧拉法离散化 |
| 混合架构 | "混合注意力和 SSM" | 任何交错 Transformer 和 SSM 层的模型；Jamba 是生产原型 |

## 延伸阅读

- [Lieber et al. -- Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887) -- 原始 Jamba 论文，比率消融，256k 上下文声明
- [AI21 -- Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570) -- 扩展家族，398B/94B 和 12B/52B 公开发布
- [Gu, Dao -- Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752) -- Jamba 构建的 SSM 论文
- [Dao, Gu -- Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060) -- 简化的结构化状态空间后继
- [Lahoti et al. -- Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569) -- 复数值状态、MIMO、2026 年纯 SSM 前沿
- [Gu et al. -- Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396) -- S4 论文，LLM 的 SSM 谱系起点
