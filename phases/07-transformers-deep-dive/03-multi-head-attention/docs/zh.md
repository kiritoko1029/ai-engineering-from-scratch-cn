# 多头注意力机制

>
> 每个注意力头一次仅学习一种关系。8个头部则可同时学习8种关系。这些头部是独立的，可根据需求增加数量。

**类型：** 构建
**语言：** Python
**先修课程：** 第7阶段 · 02（从零实现自注意力机制）
**时长：** 约75分钟

## 问题所在

单个自注意力头会计算出一个注意力矩阵。该矩阵仅能捕捉一种关系——通常是那种能够最小化训练信号损失的关系。如果你的数据中同时存在主谓一致、共指现象、长距离语义关联以及句法分块等复杂要素，单个注意力头会将它们混合为单一的软最大分布，从而导致一半的信息丢失。

2017年Vaswani论文提出的解决方案是：并行运行多个注意力函数，每个函数拥有独立的Q、K、V投影层，并将各函数的输出进行拼接。每个注意力头在维度为`d_model / n_heads`的较小子空间中工作。虽然总参数量保持不变，但模型的表达能力却得到了提升。

到2026年，所有Transformer模型均默认采用多注意力头结构。唯一的配置选项在于*头部数量*，以及键和值是否共享投影层（即分组查询注意力、多查询注意力或多头潜在注意力）。

## 概念概述

![多头注意力：拆分、查询/键/值计算与拼接](../assets/multi-head-attention.svg)

**拆分。** 取形状为 `(N, d_model)` 的向量 `X`，将其分别投影到 Q、K、V 上，得到三个同样形状为 `(N, d_model)` 的向量。接着将它们重新塑形为 `(N, n_heads, d_head)` 的结构，其中 `d_head = d_model / n_heads`。最后将该矩阵转置为 `(n_heads, N, d_head)`。

**并行执行查询/键/值计算。** 在每个注意力头内部运行缩放点积注意力机制。每个头都会生成形状为 `(N, d_head)` 的输出向量。这些头在嵌入空间的不同子空间上独立工作，在注意力计算过程中彼此不会交互。

**拼接并投影。** 将所有头的输出重新堆叠回 `(N, d_model)` 的形状，然后乘以一个形状为 `(d_model, d_model)` 的可学习输出矩阵 `W_o`。正是通过 `W_o`，各个头能够实现信息混合。

**其有效性的原因。** 每个头都可以专注于特定任务，而无需与其他头竞争表示资源。2019–2024 年间的多项研究显示了不同头部所承担的特定功能：用于处理位置信息的头部、用于关注前一个标记的头部、复制信息用的头部、用于识别命名实体的头部，以及用于实现上下文学习的诱导头部。

**2026 年前的各类变体：**

| 变体 | Q 头数量 | K/V 头数量 | 应用模型 |
|---------|----------|-----------|---------|
| 多头注意力（MHA） | N | N | GPT-2、BERT、T5 |
| 多查询注意力（MQA） | N | 1 | PaLM、Falcon |
| 分组查询注意力（GQA） | N | G（例如 N/8） | Llama 2 70B、Llama 3+、Qwen 2+、Mistral |
| 多头潜在注意力（MLA） | N | 压缩为低秩形式 | DeepSeek-V2、V3 |

GQA 是目前的默认选择，因为它能在几乎保持相同性能的前提下，将 KV 缓存内存占用降低 `N/G` 倍。MLA 则更进一步，先将 K/V 向量压缩到潜在空间中，再在计算时重新投影——虽然会带来一定的浮点运算开销，但能节省更多内存。

```figure
multihead-split
```

## 构建它

### 步骤 1：从现有的单头注意力机制中分离出多头注意力结构

使用第 02 课中的 `SelfAttention` 模块，并将其封装在拆分/合并对中。具体的 NumPy 实现可见于 `code/main.py`，其逻辑如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次重塑操作和一次转置操作。无需循环。这正是 PyTorch 在 `nn.MultiheadAttention` 中的实现方式。

### 步骤 2：为每个注意力头运行缩放点积注意力机制

每个头部都会获得属于自己的 Q、K、V 片段。注意力机制便转化为批量矩阵乘法：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真实硬件上，`Qh @ Kh.transpose(...)` 实际上对应一次 `bmm` 操作。GPU 将其视为一个形状为 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的批量矩阵乘法，增加 head 数量不会产生额外开销。

### 步骤 3：分组查询注意力变体

只有键和值的投影会发生变化。Q 会被划分为 `n_heads` 组；K 和 V 则会被划分为数量为 `n_kv_heads < n_heads` 的组，并通过重复的方式使其数量与 Q 相匹配：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

在推理阶段，这有助于节省内存，因为KV缓存中仅存储`n_kv_heads`个副本，而非`n_heads`个。Llama 3 70B模型采用64个查询头和8个KV头——这使得缓存大小减少了8倍。

### 步骤 4：检测每个头部节点学到了什么

在包含 4 个头的短句子上运行 MHA。对于每个头，输出 `(N, N)` 形状的注意力矩阵。即使初始化是随机的，你也会发现不同的头会捕捉到不同的结构——这部分源于信号特征，部分则源于子空间中的旋转对称性。

## 使用它

在 PyTorch 中，单行版本为：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 版本的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**有多少个注意力头？** 2026年量产模型的经验法则：

| 模型规模 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| 小型（约1.25亿参数） | 768 | 12 | 64 |
| 基础型（约3.5亿参数） | 1024 | 16 | 64 |
| 大型（约10亿参数） | 2048 | 16 | 128 |
| 高端型（约700亿参数） | 8192 | 64 | 128 |

`d_head` 的值几乎总是为64或128。它表示单个注意力头能够“感知”的数据量单位。若该数值低于32，各个注意力头之间会因缩放因子 `sqrt(d_head)` 而产生冲突；若高于256，则无法发挥“众多小型专用模块”带来的优势。

## 发布它

请参阅 `outputs/skill-mha-configurator.md`。该技能会根据参数预算、序列长度以及部署目标，为新的 Transformer 推荐所需的人员数量、KV 领域人数以及预测策略。

## 练习题

1. **简单。** 从 `code/main.py` 中获取 MHA 模型，将 `n_heads` 从 1 更改为 16，同时保持 `d_model=64` 不变。在合成数据复制任务上绘制单层小型模型的损失曲线。增加注意力头数会有帮助、导致性能停滞还是反而降低性能？
2. **中等难度。** 实现 MQA 模型（所有查询头共享一个 KV 头）。测量与完整 MHA 相比参数数量减少了多少。计算在 N=2048 的情况下，推理时的 KV 缓存大小会缩小到何种程度。
3. **高难度。** 实现多注意力头潜在注意力的简化版本：将 K 和 V 压缩为秩为 `r` 的潜在向量，并将该潜在向量存储在 KV 缓存中，在执行注意力计算时再解压。当 `r` 取何值时，缓存内存占用能降至完整 MHA 的 1/8 以下，同时模型在验证集上的性能误差仍控制在 1 比特以内？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Head | “单个注意力电路” | 维度为 `d_head = d_model / n_heads` 的一个 Q/K/V 投影层，拥有独立的注意力矩阵。 |
| d_head | “Head 维度” | 每个 attention head 的隐藏状态宽度；在实际应用中几乎总是 64 或 128。 |
| Split / combine | “重塑技巧” | 在注意力操作前后进行 `(N, d_model) ↔ (n_heads, N, d_head)` 的重塑与转置操作。 |
| W_o | “输出投影层” | 在将各个 head 连接之后应用的 `(d_model, d_model)` 维度矩阵，用于混合各 head 的输出。 |
| MQA | “单个 KV head” | 多查询注意力：使用单一的共享 K/V 投影层。KV 缓存大小最小，但会带来一定的质量损失。 |
| GQA | “Llama 2 及之后的默认方案” | 分组查询注意力，其 `n_kv_heads < n_heads`，并通过重复操作使 KV 头的数量与 Q 的数量相匹配。 |
| MLA | “DeepSeek 的独特技术” | 多头潜在注意力：将 K 和 V 压缩为低秩潜在向量，在执行注意力计算时再解压。 |
| Induction head | “上下文学习背后的电路” | 用于检测先前出现的内容并复制其后内容的一对 attention head。 |

## 延伸阅读

- [Vaswani 等人（2017）。《Attention Is All You Need》§3.2.2](https://arxiv.org/abs/1706.03762) —— 原始的多头注意力机制规范。
- [Shazeer（2019）。《Fast Transformer Decoding: One Write-Head is All You Need》](https://arxiv.org/abs/1911.02150) —— MQA 论文。
- [Ainslie 等人（2023）。《GQA：基于多头注意力检查点训练通用多查询 Transformer 模型》](https://arxiv.org/abs/2305.13245) —— 讲述如何在训练后将 MHA 转换为 GQA。
- [DeepSeek-AI（2024）。《DeepSeek-V2 技术报告》](https://arxiv.org/abs/2405.04434) —— MLA 模型及其在缓存内存使用效率上优于 MHA/GQA 的原因。
- [Olsson 等人（2022）。《上下文学习与引导头》](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) —— 从机制层面解析各引导头的实际功能。
