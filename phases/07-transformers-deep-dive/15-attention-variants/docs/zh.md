# 注意力机制变体——滑动窗口、稀疏型、差分型

> 全注意力机制呈现为一个圆形结构。每个标记都能感知到其他所有标记，而内存消耗则是其代价。通过四种变体可以改变该圆形的形状，从而将内存成本降低一半。

**类型：** 构建
**语言：** Python
**先修知识：** 第7阶段·02（自注意力机制）、第7阶段·03（多头注意力机制）、第7阶段·12（KV缓存/闪存注意力机制）
**耗时：** 约60分钟

## 问题所在

全注意力机制在序列长度为 N 时需要 `O(N²)` 的内存和计算资源。对于上下文长度为 128K 的 Llama 3 70B 模型，每层包含 160 亿个注意力项，共 80 层。Flash Attention（第 12 课）虽然消除了 `O(N²)` 的激活值内存需求，但并未改变计算成本——每个标记仍需与所有其他标记进行注意力计算。

有三类变体通过改变注意力矩阵的拓扑结构来优化性能：

1. **滑动窗口注意力（Sliding window attention, SWA）**：每个标记仅与固定大小的邻居窗口交互，而非整个前缀序列。其内存和计算复杂度降至 `O(N · W)`，其中 W 为窗口大小。代表模型包括 Gemma 2/3、Mistral 7B 的前几层以及 Phi-3-Long。

2. **稀疏/分块注意力（Sparse / block attention）**：仅对选定的 `(i, j)` 对计算得分，其余对的权重被强制设为零。典型代表模型有 Longformer、BigBird 以及 OpenAI 的稀疏Transformer架构。

3. **差分注意力（Differential attention）**：通过独立的 Q/K 投影计算两个注意力图，然后将二者相减。该方法能够消除会将权重泄漏到前几个标记的“注意力汇聚效应”。微软在 2024 年提出的 DIFF Transformer 即采用此机制。

这些技术可以相互结合使用。2026 年前沿模型通常会混合运用多种机制：大多数层采用 SWA-1024 结构，每五层设置一个全局全注意力层，还有少量差分注意力头用于优化信息检索效果。Gemma 3 当前采用的 5:1 的 SWA与全局注意力比例已被视为行业标准配置。

## 概念概述

### 滑动窗口注意力机制（SWA）

位置为 `i` 的每个查询仅关注区间 `[i - W, i]` 内的元素（因果单向注意力），或区间 `[i - W/2, i + W/2]` 内的元素（双向注意力）。窗口外的标记在得分矩阵中对应的值为 `-inf`。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

当 `N = 8192` 且 `W = 1024` 时，得分矩阵预计包含 1024 × 8192 个非零元素——相比之前减少了 8 倍。

**KV 缓存会随 SWA 的应用而缩小。** 每层仅需保留 K 和 V 中最近的 `W` 个标记。在类似 Gemma-3 的配置下（窗口大小为 1024，上下文长度为 128K），KV 缓存的大小将减少 128 倍。

**质量方面的代价。** 仅使用 SWA 的 Transformer 模型难以实现长距离信息检索。解决办法是将 SWA 层与全注意力层交错排列。Gemma 3 采用了 5:1 的 SWA:global 比例；而 Mistral 7B 则使用了因果式 SWA 叠加结构，通过重叠的窗口让信息“向前传递”——每层都会将模型的有效感受野扩展 `W` 个标记的长度，经过 `L` 层后，模型即可回溯 `L × W` 个标记的距离进行注意力计算。

### 稀疏/块注意力机制

提前选定一个 `N × N` 的稀疏模式。主要有三种标准形态：

- **局部 + 步长采样（OpenAI 稀疏Transformer）**：关注最后的 `W` 个标记，以及其之前的每隔一个步长的标记。该方式能够在 `O(N · sqrt(N))` 的计算量下同时捕捉局部信息与长距离依赖。
- **Longformer / BigBird**：采用局部窗口，再加上一小部分全局标记（例如 `[CLS]`），这些全局标记会关注所有其他标记，同时也被所有其他标记所关注；此外还包含随机稀疏连接。在保持相同质量的前提下，其上下文处理能力可达普通模型的2倍。
- **原生稀疏注意力（DeepSeek，2025年）**：通过学习识别哪些 `(Q, K)` 块是重要的，在内核层面直接跳过值为零的块。该机制兼容 FlashAttention。

稀疏注意力本质上属于内核工程范畴。其数学原理较为简单（即对得分矩阵进行掩码处理）；真正的优势在于能够避免将全为零的元素加载到 SRAM 中。FlashAttention-3 以及2026年推出的 FlexAttention API，使得在 PyTorch 中自定义稀疏模式成为可能。

### 差分注意力机制（DIFF Transformer，2024年）

常规注意力机制存在“注意力汇聚”问题：softmax函数要求每行的元素之和为1，因此那些不想关注特定内容的标记会将权重集中到第一个（或前几个）标记上。这会占用本应用于真实内容处理的计算资源。

差分注意力通过计算**两个**注意力图并相减来解决这一问题：

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

其中 `λ` 是一个经过学习的标量值（通常在 0.5–0.8 之间）。A1 负责捕捉真实内容权重，而 A2 则用于表示“汇点”。通过相减操作可以消除“汇点”的影响，并将权重重新分配给相关的标记。  

根据 Microsoft 2024 的测试结果：困惑度降低 5–10%，在相同的训练长度下有效上下文长度可延长 1.5–2 倍，且信息检索的精准度显著提升。

### 变体对比

| 变体类型 | 计算复杂度 | KV缓存复杂度 | 质量与全注意力模型对比 | 生产环境适用性 |
|---------|---------|----------|-----------------|----------------|
| 全注意力机制 | O(N²) | 每层 O(N) | 基准水平 | 所有模型的默认层结构 |
| SWA（窗口大小1024） | O(N·W) | 每层 O(W) | 误差为-0.1 ppl，全局层效果较好 | Gemma 2/3、Phi-3-Long模型 |
| 局部注意力+步长稀疏化 | O(N·√N) | 复合复杂度 | 与SWA性能相近 | OpenAI稀疏Transformer、Longformer模型 |
| BigBird（局部+全局+随机注意力） | 近似O(N) | 复合复杂度 | 上下文长度为全注意力模型的2倍时性能相当 | 早期长上下文BERT模型 |
| 原生稀疏机制（DeepSeek-V3.2） | O(N · 活跃节点比例) | O(N) | 误差在0.05 ppl以内 | DeepSeek-V3.2，2025年版本 |
| 微分注意力机制 | O(2·N²) | O(2N) | 误差为-5%至-10% | DIFF Transformer，2026年初的模型版本 |

```figure
gqa-kv-sharing
```

## 构建它

请参阅 `code/main.py`。我们在一个示例序列上实现了一个因果掩码比较器，可同时展示完整注意力、SWA注意力、局部+步长注意力以及差分注意力这四种机制。

### 步骤 1：完整因果掩码（基线）

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

第07课的基线模型。采用下三角矩阵结构，对角线以上的元素权重为零。

### 步骤 2：滑动窗口因果掩码

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

有一个参数——`window`。当 `window >= n` 时，即可恢复完整的因果注意力机制。而当 `window = 1` 时，每个标记仅会关注其自身。

### 步骤 3：局部 + 步进稀疏掩码

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集的局部窗口，再加上每隔一个 `stride` 个标记就回溯到序列开头。随着层数的增加，感受野以对数步长方式扩展。

### 第 4 步：差分注意力机制

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

进行两次注意力计算，然后使用一个可学习的混合系数进行减法运算。在代码中，我们会对比单通道与差分模式下的注意力汇聚热图，观察汇聚中心如何逐渐消失。

### 步骤 5：KV 缓存大小

为每个变体打印 `N = 131072` 时的各层缓存大小。SWA 及稀疏型变体的缓存大小会下降 10 到 100 倍，而差分型变体的缓存大小则会翻倍。请务必谨慎管理内存使用成本。

## 使用它

2026年生产模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 版本中的 FlexAttention 支持传入掩码函数：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

该实现会编译为自定义的 Triton 内核。在常见模式下的运行速度与 FlashAttention-3 相差不超过 10%，且掩码函数为可调用的 Python 函数。

**各方案的适用场景：**

- **纯全注意力机制** —— 上下文长度高达约 16K 的所有层，或对检索质量要求极高的场景。
- **SWA + 全局混合机制** —— 长上下文（>32K）、训练及推理过程中存在内存限制的场景。上述 2026 版本适用于超过 32K 的上下文长度。
- **稀疏块注意力机制** —— 用于自定义内核及特定模式，专为检索、音频等专业工作负载设计。
- **差分注意力机制** —— 出现注意力“污染”问题会严重影响性能的任何工作负载（如长上下文的 RAG 任务、大海捞针式检索）。

## 发布它

请参阅 `outputs/skill-attention-variant-picker.md`。该技能会根据目标上下文长度、检索需求以及训练/推理时的计算资源情况，为新的模型选择合适的注意力拓扑结构。

## 练习题

1. **简单。** 运行 `code/main.py`。验证当 `window=4` 时，SWA 能将每行最后 4 个标记之外的所有内容置零。再验证当 `window=n` 时，其输出与全因果注意力机制的结果完全一致。
2. **中等难度。** 在第 07 课的最终项目基础上，实现窗口大小为 `1024` 的因果 SWA 模型。在 tinyshakespeare 数据集上训练 1,000 步。对比该模型与全注意力机制的验证损失下降幅度，以及峰值内存占用降低的程度。
3. **高难度。** 在最终项目模型中实现类似 Gemma-3 的 5:1 层混合结构（即 5 个 SWA 层与 1 个全局层）。在参数设置相同的情况下，对比该模型与纯 SWA 模型及纯全局注意力模型的损失值、内存占用以及生成质量。
4. **高难度。** 实现带有每个头独立学习参数 `λ` 的差分注意力机制。在一个包含一个目标项和 2,000 个干扰项的合成检索任务上进行训练。在参数设置相同的情况下，测量该模型的检索准确率与单注意力基准模型的表现。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 滑动窗口注意力（Sliding window attention, SWA） | “局部注意力” | 每个查询仅关注其最近的 `W` 个标记；KV 缓存大小缩减为 `O(W)`。 |
| 有效感受野 | “模型能回溯多远” | 在具有窗口宽度 `W` 的 `L` 层 SWA 叠加结构中，可达 `L × W` 个标记。 |
| Longformer / BigBird | “局部 + 全局 + 随机” | 采用稀疏模式，并包含少量始终参与全局关注的标记；属于早期的长上下文处理方法。 |
| 原生稀疏注意力 | “DeepSeek 的核技巧” | 学习块级稀疏性；在保持质量的同时，在内核层面跳过零值块。 |
| 微分注意力 | “两个映射，一个相减” | DIFF Transformer：通过从第一个注意力映射中减去经过学习得到的、大小为 `λ` 的第二个注意力映射，来消除注意力汇聚现象。 |
| 注意力汇聚点 | “权重泄漏到第 0 个标记” | Softmax 归一化要求各行之和为 1；信息量不足的查询会将权重集中到位置 0 上。 |
| FlexAttention | “类 Python 的掩码机制” | 基于 PyTorch 2.5+ 的 API，可将任意掩码函数编译为 FlashAttention 格式的内核。 |
| 层类型混合 | “5:1 SWA 与全局注意力的比例” | 在叠加结构中交替使用稀疏注意力层和全注意力层，以在较低内存消耗下保持性能。 |

## 延伸阅读

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 该论文提出了经典的滑动窗口与全局令牌处理方法。  
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — 采用局部、全局及随机三种处理策略。  
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI提出的基于局部处理与步长调整的序列生成方法。  
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 采用1:1比例的滑动窗口与全局处理相结合的方式。  
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 采用5:1的比例，窗口大小为1024，现已成为行业标准配置。  
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer相关论文。  
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2中采用的基于学习机制的稀疏注意力机制。  
- [PyTorch — FlexAttention博客与文档](https://pytorch.org/blog/flexattention/) — 介绍`Use It`模块中“将掩码视为可调函数”这一模式的API参考资料。
