# 完整的 Transformer 模型——编码器与解码器

> 注意力机制是核心所在。其余所有内容——残差连接、归一化处理、前馈网络以及交叉注意力——都只是用于构建更深层结构的支撑框架。

**类型：** 构建
**语言：** Python
**先修知识：** 第 7 阶段 · 02（自注意力机制）、第 7 阶段 · 03（多头注意力机制）、第 7 阶段 · 04（位置编码）
**耗时：** 约 75 分钟

## 问题所在

单个注意力层仅是一个特征提取器，而非完整的模型。每层仅一次矩阵乘法无法满足处理语言任务的需求，因此需要增加层数——但若架构设计不当，增加层数反而会导致性能下降。

2017年Vaswani发表的论文总结了六项关键设计决策，正是这些决策使得单个注意力层得以演变为可堆叠的模块。此后的所有Transformer模型——仅编码器结构（BERT）、仅解码器结构（GPT）以及编解码器结构（T5）——都沿用了相同的架构框架。尽管到2026年时这些模块已经过进一步优化（加入了RMSNorm、SwiGLU、预归一化及RoPE等技术），但其核心架构依然保持不变。

本课将介绍这一核心架构框架，后续课程将分别针对编码器、解码器以及编解码器结构进行深入讲解——分别为第06课、第07课和第08课。

## 概念概述

![编码器与解码器模块内部结构示意图，连线版](../assets/full-transformer.svg)

### 六个组件

1. **嵌入向量 + 位置信号。** 将标记转换为向量，通过 RoPE（现代方法）或正弦函数（传统方法）注入位置信息。  
2. **自注意力机制。** 每个位置都会关注其他所有位置，在解码器中会应用掩码处理。  
3. **前馈网络（FFN）。** 按位置分组的双层多层感知机：`W_2 · activation(W_1 · x)`，默认扩展倍率为 4 倍。  
4. **残差连接。** `x + sublayer(x)`。若无此结构，梯度在超过约 6 层后会消失。  
5. **层归一化。** 使用 `LayerNorm` 或 `RMSNorm`（现代方法），用于稳定残差流。  
6. **交叉注意力机制（仅解码器使用）。** 查询来自解码器，键和值则来自编码器的输出。  

观察一个向量如何流经一个处理模块：自注意力机制实现跨位置的信息混合，残差连接将其传递下去，前馈网络对其进行变换，而层归一化则确保整个流的稳定性。

```figure
transformer-block
```

### 编码器模块（BERT、T5 编码器所使用）

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

编码器为双向结构，不进行掩码处理。所有位置均可查看其他所有位置的信息。

### 解码器模块（由 GPT、T5 解码器使用）

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

解码器每个块包含三个子层。其中位于中间的交叉注意力层是信息从编码器流向解码器的唯一通道。在纯解码器架构（如 GPT）中，会省略交叉注意力层，仅保留掩码自注意力和前馈网络。

### 规范前处理与规范后处理

原始论文：`x + sublayer(LN(x))` 对比 `LN(x + sublayer(x))`。约在 2019 年左右，后归一化逐渐失宠——若不进行仔细的预热处理，则难以训练深度模型。前归一化（在子层操作*之前*进行 `LN` 操作）已成为 2026 年的默认方案：Llama、Qwen、GPT-3+ 以及 Mistral 等模型均采用该方式。

### 2026年升级版块

Vaswani 在 2017 年提出了 LayerNorm 和 ReLU。现代模型架构已用其他组件取代了它们。以下是实际生产环境中的配置对比：

| 组件 | 2017 年 | 2026 年 |
|-----------|------|------|
| 归一化层 | LayerNorm | RMSNorm |
| FFN 激活函数 | ReLU | SwiGLU |
| FFN 扩展倍数 | 4× | 2.6×（SwiGLU 使用三个矩阵，总参数量保持一致） |
| 位置编码 | 正弦绝对值 | RoPE |
| 注意力机制 | 全连接多头注意力 | GQA（或 MLA） |
| 偏置项 | 有 | 无 |

RMSNorm 省去了 LayerNorm 的均值居中操作（减少了一次减法运算），从而节省了计算资源，且实验表明其稳定性至少与 LayerNorm 相当。在 Llama、PaLM 和 Qwen 的相关论文中，SwiGLU（`Swish(W1 x) ⊙ W3 x`）的性能始终比 ReLU/GELU 构建的 FFN 高出约 0.5 个 PPL 分数。

### 参数数量

对于 `d_model = d` 且 FFN 展开系数为 `r` 的单个块：

- MHA：`4 · d²`（Q、K、V、O 四个投影）
- FFN（SwiGLU）：`3 · d · (r · d)` ≈ `3d²`
- 范数项：可忽略不计

在 `d = 4096, r = 2.6, 层数 = 32`（大致对应 Llama 3 8B 模型）的条件下，总参数量为：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B 参数/层 × 32 ≈ 7B`（此数值未包含嵌入层和输出头参数）。与已公布的参数数量相符。

## 构建它

### 步骤 1：基础构建模块

使用第 03 课中的小型 `Matrix` 类（为保持独立性已复制到此文件中）：

- `layer_norm(x, eps=1e-5)` —— 先减去均值，再除以标准差。
- `rms_norm(x, eps=1e-6)` —— 直接除以均方根值，不进行均值减法。
- `gelu(x)` 以及 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 和 `decoder_block(x, enc_out, params)`。

完整的代码实现请参见 `code/main.py`。

### 步骤 2：连接双层编码器与双层解码器

将它们堆叠起来。将编码器的输出传递给每个解码器的交叉注意力层。在输出投影之前添加一个最终的线性层。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步骤 3：在示例数据上执行前向传播

将长度为6个标记的源序列和长度为5个标记的目标序列输入模型。需验证输出形状为`(5, vocab)`。本环节无需进行训练——本课重点在于模型架构，而非损失函数。

### 步骤 4：替换为 RMSNorm + SwiGLU

将 LayerNorm 和 ReLU-FFN 替换为 RMSNorm 和 SwiGLU。需确认各张量的形状依然匹配。这是 2026 年的现代化改进，仅涉及一个函数的替换。

## 使用它

PyTorch/TF 的参考实现为：`nn.TransformerEncoderLayer` 和 `nn.TransformerDecoderLayer`。但大多数 2026 年的实战代码会自行实现相关模块，原因如下：

- Flash Attention 在注意力计算内部被直接调用，而非通过 `nn.MultiheadAttention` 实现。
- GQA / MLA 不包含在标准库的参考实现中。
- RoPE、RMSNorm、SwiGLU 并非 PyTorch 的默认配置。

Hugging Face 的 `transformers` 库提供了结构清晰的参考模块，值得查阅：`modeling_llama.py` 是 2026 年标准的仅解码器模块，约含 500 行代码，建议仔细研读一遍。

**编码器、解码器与编解码器——何时选择：**

| 需求 | 推荐类型 | 示例 |
|------|----------|---------|
| 文本分类、嵌入生成、文本问答 | 仅编码器 | BERT、DeBERTa、ModernBERT |
| 文本生成、对话系统、代码生成、推理 | 仅解码器 | GPT、Llama、Claude、Qwen |
| 结构化输入转换为结构化输出（翻译、摘要） | 编解码器 | T5、BART、Whisper |

仅解码器类型的语言模型因扩展性最佳且能同时处理理解与生成任务而备受青睐。当输入具有明确的“源序列”特征时（如翻译、语音识别、结构化任务），编解码器仍然是最佳选择。

## 发布它

请参阅 `outputs/skill-transformer-block-reviewer.md`。该技能会依据 2026 年的默认配置来审查新的 Transformer 块实现，并标记缺失的组件（预归一化层、RoPE、RMSNorm、GQA 以及 FFN 扩展比）。

## 练习题

1. **简单。** 统计 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 的 encoder_block 中的参数数量。通过实现该模块并使用 `sum(p.numel() for p in block.parameters())` 进行验证。
2. **中等难度。** 将归一化方式从后归一化改为前归一化。分别初始化这两种归一化方式，然后对随机输入进行 12 层堆叠处理后，测量激活值的范数。后归一化的激活值应出现剧烈波动；而前归一化的激活值则应保持在一个限定范围内。
3. **高难度。** 在一个简单的复制任务（将输入 `x` 反转后复制）上实现一个包含 4 层的编解码器结构。训练 100 步，并报告损失值。随后替换为 RMSNorm、SwiGLU 和 RoPE —— 损失值是否会下降？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Block | “一个Transformer层” | 由归一化 + 注意力机制 + 归一化 + FFN堆叠而成，并通过残差连接包裹。 |
| Residual | “跳过连接” | 输出为 `x + f(x)`；使梯度能够贯穿深层结构。 |
| Pre-norm | “先归一化，后处理” | 现代做法：`x + sublayer(LN(x))`。无需预热步骤即可训练更深层模型。 |
| RMSNorm | “去掉了均值项的LayerNorm” | 以RMS值作为除数；操作次数减少1次，但稳定性保持不变。 |
| SwiGLU | “大家都改用的FFN结构” | `Swish(W1 x) ⊙ W3 x → W2`。在语言模型任务中性能优于ReLU/GELU。 |
| Cross-attention | “解码器如何感知编码器” | 使用来自解码器的Q值以及来自编码器输出的K/V值构成的多头注意力机制。 |
| FFN expansion | “中间MLP的宽度” | 隐藏层大小与d_model大小的比值，通常为4（LayerNorm）或2.6（SwiGLU）。 |
| Bias-free | “去掉+b项” | 现代架构在线性层中省略偏置项；可略微提升模型性能并减小模型体积。 |

## 延伸阅读

- [Vaswani 等人 (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始块结构规范。
- [Xiong 等人 (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 深度学习中为何前置归一化优于后置归一化。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 相关论文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 标准的 2026 版仅解码器块结构。
