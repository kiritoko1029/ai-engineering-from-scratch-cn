# 注意力机制——这一重大突破

>
> 解码器不再只盯着压缩后的摘要，而是开始查看完整的源代码。此之后的所有内容都属于注意力机制与工程实现的范畴。

**类型：** 构建
**语言：** Python
**先修要求：** 第 5 阶段 · 09（序列到序列模型）
**耗时：** 约 45 分钟

## 问题所在

第09课以一次有控制的失败告终。在一个简单的复制任务上训练的GRU编解码器，其在序列长度为5时的准确率高达89%，而当序列长度达到80时则几乎退化到随机水平。造成这一现象的原因是结构上的限制，而非训练错误：编码器获取的所有信息都必须存储在固定大小的隐藏状态中，而解码器无法看到其他任何信息。

2014年，Bahdanau、Cho和Bengio提出了一个仅需三行代码的解决方案。他们不再只向解码器提供最终的编码器状态，而是保留所有的编码器状态。在每个解码步骤中，根据权重计算这些编码器状态的加权平均值，这些权重决定了“当前解码器需要关注编码器位置`i`的程度”。该加权平均值即为上下文，并且会在每一个解码步骤中发生变化。

这就是其核心思想。Transformer模型对其进行了扩展：自注意力机制将其应用于单个序列，多头注意力机制则使其能够并行处理。不过，2014年的版本就已经打破了这一瓶颈；一旦掌握了这一方法，向Transformer的过渡便只是工程实现上的调整，而非概念上的变革。

## 概念概述

![Bahdanau注意力机制：解码器查询所有编码器状态](../assets/attention.svg)

在每个解码器步骤 `t` 中：

1. 使用之前的解码器隐藏状态 `s_{t-1}` 作为**查询向量**。
2. 将其与每一个编码器隐藏状态 `h_1, ..., h_T` 进行匹配度计算，每个编码器位置对应一个标量值。
3. 对这些匹配度值应用Softmax函数，得到注意力权重 `α_{t,1}, ..., α_{t,T}`，这些权重的总和为1。
4. 上下文向量 `c_t = Σ α_{t,i} * h_i`，即编码器状态值的加权平均值。
5. 解码器结合 `c_t` 以及之前的输出词元，生成下一个词元。

这种加权平均机制是关键所在。当解码器需要将“Je”翻译为“I”时，它会提高对应“Je”的编码器状态的权重，而降低其他状态的权重；当需要翻译“not”时，则会提高“pas”对应的权重。上下文向量会在每一步中改变其形态。

## 形状（让所有人头疼的问题）

每次实现注意力机制时，问题往往都出在这里。请慢慢阅读。

| 项目 | 形状 | 备注 |
|-------|-------|-------|
| 编码器隐藏状态 `H` | `(T_enc, d_h)` | 若为 BiLSTM，则 `d_h = 2 * d_hidden` |
| 解码器隐藏状态 `s_{t-1}` | `(d_s,)` | 单个向量 |
| 注意力得分 `e_{t,i}` | 标量 | 每个编码器位置一个 |
| 注意力权重 `α_{t,i}` | 标量 | 对所有 `i` 应用 softmax 后得到 |
| 上下文向量 `c_t` | `(d_h,)` | 形状与编码器状态相同 |

**Bahdanau（加法）得分。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的形状为 `(d_s,)`，`h_i` 的形状为 `(d_h,)`。
- `W_a` 的形状为 `(d_attn, d_s)`。`U_a` 的形状为 `(d_attn, d_h)`。
- 两者在 tanh 内部的和的形状为 `(d_attn,)`。
- `v_α` 的形状为 `(d_attn,)`。与 `v_α` 做内积后会得到一个标量。**这就是 `v_α` 的作用。** 它并非魔法，而是将注意力维度向量转换为标量得分的投影操作。

**Luong（乘法）得分。** 有三种变体：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`，这是一个严格约束。如果使用双向编码器，则跳过此方式。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 的形状为 `(d_s, d_h)`。消除了维度相等的约束。
- `concat`：本质上与 Bahdanau 形式相同。由于前两种方式计算成本更低，因此很少使用。

**关于 Bahdanau 和 Luong 需要特别注意的一点。** Bahdanau 使用的是 `s_{t-1}`（生成当前词之前的解码器状态）。而 Luong 使用的是 `s_t`（生成之后的状态）。如果混淆两者，会导致梯度出现细微错误，极难调试。请选择其中一篇论文并遵循其约定。

```figure
attention-heatmap
```

## 构建它

### 步骤 1：加性（Bahdanau）注意力机制

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

请将您的形状与上表进行比对。`encoder_states` 的形状为 `(T_enc, d_h)`。`projected_enc` 的形状为 `(T_enc, d_attn)`。`projected_dec` 的形状为 `(d_attn,)`，并进行广播操作。`combined` 的形状为 `(T_enc, d_attn)`。`scores` 的形状为 `(T_enc,)`。`weights` 的形状为 `(T_enc,)`。`context` 的形状为 `(d_h,)`。现在将其提交即可。

### 步骤 2：Luong 点与通用设置

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

每行三句。这就是Luong的论文能够被采用的理由：在大多数任务上具有相同的准确率，但代码量却少得多。

### 步骤 3：一个带详细计算的数值示例

给定三个编码器状态（大致对应“cat”、“sat”、“mat”）以及一个与第一个状态最匹配的解码器状态时，注意力分布会集中在位置 0。如果解码器状态转变为与最后一个状态对齐，则注意力会转移到位置 2。上下文向量则会随之变化。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一行获胜。接着将解码器状态向第三个编码器状态靠近，并观察权重的变化。就是这样。注意力机制即显式对齐。

### 第 4 步：为何这是通往 Transformer 的桥梁

- **查询** = 解码器状态 `s_{t-1}`
- **键** = 编码器状态（用于对比评分的对象）
- **值** = 编码器状态（用于加权求和的对象）

在传统注意力机制中，键与值是同一概念。自注意力则将二者分开：可以通过不同的学习得到的投影，让序列对自身进行查询操作。多头注意力则利用不同的学习得到的投影并行执行该操作。Transformer模型通过多次堆叠整个处理阶段来替代RNN结构。

其数学公式相同，数据形状也一致。从Bahdanau注意力机制到缩放点积注意力的教学过渡，主要体现在符号表示上的变化。

## 使用它

PyTorch 和 TensorFlow 均直接内置了注意力机制。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

这就是一个 Transformer 注意力层。查询批次包含 5 个位置，键/值批次包含 10 个位置，二者各自的维度均为 128，并拥有 8 个注意力头。`output` 是经过上下文增强后的新查询向量。`weights` 则是那个 5×10 的对齐矩阵，你可以将其可视化。

### 当传统注意力机制依然重要时

- 教学方法。基于单头、单层 RNN 的版本能够清晰展现每一个概念。
- 变换器模型无法处理的设备端序列任务。
- 2014 年至 2017 年间的任何论文。若不了解 Bahdanau 的约定，将无法正确解读这些内容。
- 翻译中的细粒度对齐分析。即使在变换器模型中，原始注意力权重也是一种可解释性工具，而要理解它们需要先知晓其含义。

### 将注意力权重视为解释的陷阱

注意力权重看似具有可解释性。这些权重在各个位置上的值之和为一，可以将其绘制成图表；数值较高意味着“关注了该位置”。审稿人通常很青睐这类指标。

但实际上它们的可解释性并不如表面那样简单。Jain 和 Wallace（2019）的研究表明，在某些任务中，注意力分布可以通过重排或替换为任意其他形式，而不会改变模型的预测结果。在没有进行消融实验或反事实验证的情况下，切勿将注意力权重作为模型具备推理能力的证据。

## 发布它

保存为 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 练习题

1. **简单。** 实现 `softmax` 叠加机制，使得编码器中的填充标记获得零注意力权重。在序列长度不固定的批次数据上进行测试。  
2. **中等难度。** 在 Luong 的 `general` 形式中加入多头注意力机制。将 `d_h` 分为 `n_heads` 组，对每组分别执行注意力计算后再进行拼接。验证单头注意力情况是否与之前的实现结果一致。  
3. **高难度。** 基于第 09 课中的简易复制任务，使用 Bahdanau 注意力机制训练 GRU 编码器-解码器模型。绘制准确率随序列长度变化的曲线，并与无注意力机制的基线模型进行对比。应当会发现随着序列长度的增加，两者之间的差距会逐渐扩大，从而证明注意力机制有效缓解了性能瓶颈。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 注意力机制 | 观察事物 | 值序列的加权平均值，权重由查询项与键项的相似度计算得出。 |
| 查询项、键项、值项 | QKV | 三种投影：Q用于提问，K用于匹配，V用于返回结果。 |
| 加法注意力机制 | Bahdanau | 前馈得分公式：`v^T tanh(W q + U k)`。 |
| 乘法注意力机制 | Luong dot / 通用型 | 得分公式为 `q^T k` 或 `q^T W k`。计算成本更低，在大多数任务中精度相当。 |
| 对齐矩阵 | 可视化图表 | 以 `(T_dec, T_enc)` 网格形式呈现的注意力权重，通过查看该矩阵可了解模型关注了哪些内容。 |

## 延伸阅读

- [Bahdanau, Cho, Bengio (2014). 通过联合学习对齐与翻译实现神经机器翻译](https://arxiv.org/abs/1409.0473) —— 相关论文。
- [Luong, Pham, Manning (2015). 基于注意力的神经机器翻译的有效方法](https://arxiv.org/abs/1508.04025) —— 三种评分变体及其对比分析。
- [Jain and Wallace (2019). 注意力并非解释](https://arxiv.org/abs/1902.10186) —— 关于可解释性的警示。
- [深入深度学习 —— Bahdanau注意力机制](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) —— 基于PyTorch的可运行示例演示。
