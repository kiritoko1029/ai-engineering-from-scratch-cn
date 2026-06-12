# 位置编码——正弦函数型、RoPE、ALiBi

> 注意力机制具有排列不变性。句子“猫坐在垫子上”与“mat the on sat cat the”在缺乏位置信息的情况下会产生相同的输出。有三种算法可以解决这一问题——它们对“位置”的含义有不同的理解。

**类型：** 构建
**语言：** Python
**先修知识：** 第7阶段 · 02（自注意力机制），第7阶段 · 03（多头注意力机制）
**耗时：** 约45分钟

## 问题所在

缩放点积注意力机制具有顺序无关性。注意力矩阵 `softmax(Q K^T / √d) V` 是根据元素间的相似度计算得出的。对 `X` 的行进行打乱后，输出结果的行也会以相同方式被打乱。注意力机制内部不会考虑位置信息。

这在词袋模型中并非缺陷，但对于语言、代码、音频、视频等顺序具有语义意义的领域而言，则是致命问题。

解决方法是通过某种方式为嵌入向量注入位置信息。目前主要有三种解决方案：

1. **绝对正弦编码**（Vaswani 2017）。在嵌入向量中加入位置的 `sin/cos` 值。该方法简单，无需训练参数，但在训练数据范围之外的表现较差。
2. **RoPE — 旋转位置嵌入**（Su 2021）。根据位置按比例旋转 Q 和 K 向量，在点积运算中直接编码*相对*位置信息。该方案在 2026 年已成为主流。
3. **ALiBi — 带线性偏置的注意力机制**（Press 2022）。完全跳过嵌入向量环节，而是根据距离为每个注意力头添加线性惩罚项。该方法在处理长序列时的表现极为出色。

截至 2026 年，几乎所有前沿的开源模型都采用了 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi 等。仅有少数长上下文模型仍在使用 ALiBi 或其现代变体。绝对正弦编码则已属于历史产物。

## 概念概述

![正弦绝对值旋转、RoPE旋转与ALiBi距离偏差对比图](../assets/positional-encoding.svg)

### 绝对正弦波

预先计算一个形状为 `(max_len, d_model)` 的固定矩阵 `PE`：

```
PE[pos, 2i]   = sin(pos / 10000^(2i / d_model))
PE[pos, 2i+1] = cos(pos / 10000^(2i / d_model))
```

在注意力机制之前，首先执行 `X' = X + PE[:N]`。每个维度都对应不同频率的正弦波。模型通过相位模式来学习位置信息。当超出 `max_len` 时会出现问题：由于模型仅见过位置 0–2047 的数据，因此无法知晓位置 2048 的情况。

### RoPE

旋转 Q 向量和 K 向量（而非嵌入向量）。对于尺寸为 `(2i, 2i+1)` 的一对维度：

```
[q'_2i    ]   [ cos(pos·θ_i)  -sin(pos·θ_i) ] [q_2i   ]
[q'_2i+1  ] = [ sin(pos·θ_i)   cos(pos·θ_i) ] [q_2i+1 ]

θ_i = base^(-2i / d_head),  base = 10000 by default
```

对位置为 `pos_k` 的键应用相同的旋转操作。点积 `q'_m · k'_n` 将仅取决于 `(m - n)` 这一差值。也就是说：**注意力分数仅依赖于相对距离**，尽管旋转操作是基于绝对位置来执行的。真是巧妙的技巧。

扩展 RoPE：可以通过对 `base` 值进行缩放（如 NTK-aware、YaRN、LongRoPE 等方法）来实现对更长上下文的支持，而无需重新训练模型。Llama 3 就是通过这种方式将其上下文长度从 8K 扩展到了 128K。

### ALiBi

跳过嵌入技巧，直接对注意力分数进行偏置处理：

```
attn_score[i, j] = (q_i · k_j) / √d  -  m_h · |i - j|
```

其中 `m_h` 是针对特定头部的斜率（例如 `1 / 2^(8·h/H)`）。距离更近的标记会获得增强，而距离较远的标记则会受到惩罚。该方法在训练过程中不会产生额外成本。论文表明，在原始训练长度上，这种长度外推方法的表现优于正弦函数模型，并与 RoPE 方法相当。

### 2026年该选择什么？

| 变体类型 | 外推能力 | 训练成本 | 应用场景 |
|---------|---------------|---------------|---------|
| 绝对正弦函数 | 差 | 免费 | 原始 Transformer、早期 BERT |
| 学习得到的绝对值 | 无 | 极低 | GPT-2、GPT-3 |
| RoPE | 具备缩放功能，表现良好 | 免费 | Llama 2/3/4、Qwen 2/3、Mistral、DeepSeek-V3、Kimi |
| RoPE + YaRN | 表现极佳 | 需进行微调 | Qwen2-1M、Llama 3.1 128K |
| ALiBi | 表现极佳 | 免费 | BLOOM、MPT、Baichuan |

RoPE 获胜的原因在于它无需改变注意力机制的架构即可融入其中，能够编码相对位置信息，且其 `base` 超参数为长上下文微调提供了便捷的调节手段。

```figure
rope-explorer
```

## 构建它

### 步骤 1：正弦编码

参见 `code/main.py`。这是一段仅包含 4 行的计算代码：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一个注意力层之前，将此内容添加到嵌入矩阵中。

### 步骤 2：将 RoPE 应用于 Q 和 K

RoPE 在 Q 和 K 上直接进行原地操作。对于每一对维度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

重要提示：需将同一函数应用于位置为 `m` 的 Q 和位置为 `n` 的 K。它们的点积会在每一对坐标上乘以 `cos((m-n)·θ_i)` 这一因子。注意力机制因此能够免费获取相对位置信息。

### 步骤 3：ALiBi 斜率与偏置

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

将 `bias[h]` 加到第 `h` 个头的 `(seq_len, seq_len)` 注意力得分矩阵中，随后进行 softmax 操作。

### 步骤 4：验证 RoPE 的相对距离属性

随机选取两个向量 `a` 和 `b`。首先将其绕点 `(pos_a, pos_b)` 旋转，然后再绕点 `(pos_a + k, pos_b + k)` 旋转。两次旋转后的点积必须在浮点误差范围内保持一致。这一特性正是 RoPE 的核心所在——它对绝对偏移量具有不变性，仅有相对间距才起作用。

## 使用它

PyTorch 2.5 及更高版本在 `torch.nn.functional` 模块中提供了 RoPE 相关工具函数。大多数生产环境中的代码会使用 `flash_attn` 或 `xformers`，这些库会在注意力核内部实现 RoPE 的计算。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026年的长上下文处理技巧：**

- **NTK感知插值法。** 在从4K扩展到16K及以上分辨率时，将`base`重新缩放为 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 一种更智能的插值方法，能够在长上下文中保持注意力熵值不变。Llama 3.1 128K版本便采用了该技术。
- **LongRoPE。** 微软在2024年提出的方法，通过进化搜索为每个维度选择合适的缩放因子。Phi-3-Long模型也使用了这一技术。
- **位置插值结合微调。** 仅需根据扩展比例缩小位置信息，并对1–5B个token进行微调，效果出奇地好。

## 发布它

请参阅 `outputs/skill-positional-encoding-picker.md`。该技能会根据目标上下文长度、外推需求以及训练预算，为新模型选择合适的编码策略。

## 练习题

1. **简单。** 对 `max_len=512, d=128` 的正弦型 `PE` 矩阵进行热图绘制，验证“随着维度索引增大，条纹会变宽”的规律。
2. **中等难度。** 实现考虑 NTK 特性的 RoPE 缩放机制。首先在长度为 256 的序列上训练一个小型语言模型，然后在有与无缩放的情况下对长度为 1024 的序列进行测试，并测量困惑度。
3. **高难度。** 在同一个注意力模块中同时实现 ALiBi 和 RoPE 算法。在长度为 512 的序列的复制任务上训练一个 4 层 Transformer 模型，并在测试时将其扩展到长度为 2048 的序列，比较性能下降程度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 位置编码 | “告诉注意力机制顺序信息” | 添加到嵌入向量或注意力权重中的、用于编码位置的任意信号。 |
| 正弦函数型 | “原始的那种” | 在嵌入向量中添加几何频率形式的 `sin/cos` 函数；不会产生外推误差。 |
| RoPE | “旋转嵌入” | 根据位置信息对 Q 和 K 向量进行角度旋转；通过点积来编码相对距离。 |
| ALiBi | “线性偏置技巧” | 在注意力得分中添加 `-m·\|i-j\|` 项；无需使用嵌入向量，具备良好的外推能力。 |
| base | “RoPE 的调节参数” | RoPE 中用于控制频率的缩放因子；增大该值可在推理时扩展上下文长度。 |
| NTK-aware | “一种 RoPE 缩放技巧” | 对 `base` 值进行重新缩放，以避免在上下文扩展时高频维度被压缩。 |
| YaRN | “最复杂的那一种” | 通过针对不同维度的插值与外推处理来保持注意力熵值不变。 |
| 外推能力 | “能在训练长度之外正常工作” | 该位置编码方案是否能够在超出训练数据中 `max_len` 长度的情况下仍输出正确结果？ |

## 延伸阅读

- [Vaswani 等人（2017）。《Attention Is All You Need》§3.5](https://arxiv.org/abs/1706.03762) —— 原始的正弦函数型 RoPE。
- [Su 等人（2021）。《RoFormer：基于旋转位置嵌入的增强版 Transformer》](https://arxiv.org/abs/2104.09864) —— RoPE 相关论文。
- [Press、Smith、Lewis（2021）。《Train Short, Test Long：利用线性偏置的注意力机制实现输入长度扩展》](https://arxiv.org/abs/2108.12409) —— ALiBi 方法。
- [Peng 等人（2023）。《YaRN：大型语言模型高效上下文窗口扩展方法》](https://arxiv.org/abs/2309.00071) —— 当前最先进的 RoPE 缩放技术。
- [Chen 等人（2023）。《通过位置插值扩展大型语言模型的上下文窗口》](https://arxiv.org/abs/2306.15595) —— Meta 的 Llama 2 长上下文相关论文。
- [Ding 等人（2024）。《LongRoPE：将大语言模型的上下文窗口扩展至 200 万个标记以上》](https://arxiv.org/abs/2402.13753) —— Phi-3-Long 所采用的微软方法，亦在“使用指南”章节中被引用。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) —— 各种 RoPE 缩放方案（默认型、线性型、动态型、YaRN 型、LongRoPE 型、Llama-3 型）的工业级实现代码。
