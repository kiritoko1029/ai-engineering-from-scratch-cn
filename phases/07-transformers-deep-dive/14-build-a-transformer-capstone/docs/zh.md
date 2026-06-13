# 从零构建 Transformer 模型——课程总结项目

>
> 十三节课。一个模型。没有捷径。

**类型：** 构建
**语言：** Python
**先修要求：** 第7阶段 · 第01至13课。不可跳过。
**时长：** 约120分钟

## 问题所在

你已经阅读了所有相关论文，也实现了注意力机制、多头分割、位置编码、编码器与解码器模块、BERT和GPT损失函数、MoE架构以及KV缓存。现在，需要将这些组件整合起来，用于解决一个实际任务。

课程的最终项目是：在字符级语言建模任务上端到端训练一个小型纯解码器结构的Transformer模型。该模型能够阅读莎士比亚的作品，并生成新的莎士比亚风格文本。它的规模小到足以在笔记本电脑上不到10分钟的时间内完成训练，且其性能足够好——只要更换更大的数据集并延长训练时间，就能得到一个真正的语言模型。

这就是本课程中的“nanoGPT”。它并非原创成果——Karpathy在2023年发布的nanoGPT教程是每位学生至少需要实现一次的参考代码。我们只是沿用了其结构，并根据本课程所学的内容对其进行了调整。

## 概念概述

![从零构建 Transformer 的结构图](../assets/capstone.svg)

带注释的架构图：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我们交付的内容

- `GPTConfig` — 所有超参数的统一配置入口。
- `MultiHeadAttention` — 具有因果关系的批量注意力机制，可选 Flash 风格路径（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN` — 现代化的 FFN 结构。
- `Block` — 前归一化处理后的残差注意力层 + FFN。
- `GPT` — 嵌入层、堆叠的块结构、语言模型头以及生成函数 `generate()`。
- 训练循环采用 AdamW 优化器、余弦学习率调度以及梯度裁剪技术。
- 基于莎士比亚文本的字符级分词器。

### 我们未发布的版本

- RoPE — 在第 04 课中已从概念层面进行讲解。此处为简化实现，我们使用了学习得到的位置嵌入。练习要求你将其替换为 RoPE。
- 生成过程中的 KV 缓存 — 每一步生成都会重新计算整个前缀的注意力权重。这种方式速度较慢但实现更简单。练习要求你添加 KV 缓存。
- Flash Attention — 若输入条件满足，PyTorch 2.0+ 会自动启用该机制；我们在此使用 `F.scaled_dot_product_attention` 函数。
- MoE — 每个块仅包含一个 FFN。你已在第 11 课中了解过 MoE。

### 目标指标

在 Mac M2 笔记本电脑上，使用 4 层、4 个头、d_model=128 的 GPT 模型，在 `tinyshakespeare.txt` 数据集上训练了 2,000 步：

- 训练损失从随机初始化的约 4.2 降至约 1.5，耗时约 6 分钟。
- 生成的样本文本具有莎士比亚风格的特征：出现古旧词汇、换行符以及“ROMEO:”之类的专有名词。
- 验证损失（使用留出的最后 10% 的文本计算）与训练损失走势极为接近，说明在该模型规模及资源限制下不存在过拟合现象。

## 构建它

本课使用 PyTorch。请安装 `torch`（CPU 版即可）。参见 `code/main.py`。该脚本负责以下功能：

- 若缺失则下载 `tinyshakespeare.txt`，否则读取本地副本。
- 基于字节的字符分词器。
- 90/10 的训练集与验证集划分。
- 在支持该特性的硬件上使用 bf16 自动类型转换进行训练循环。
- 训练完成后进行采样。

### 步骤 1：数据

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65个唯一字符。词汇量极小，可容纳4字节的vocab_size。无需BPE算法，也无需复杂的分词器。

### 步骤 2：模型

请参阅 `code/main.py`。该代码块源自第 05 课的典型示例——预归一化、RMSNorm、SwiGLU 以及因果 MHA。参数量为 4/4/128 时约为 80 万个。

### 步骤 3：训练循环

获取长度为 256 的随机批次令牌窗口。前向传播。使用逐元素交叉熵损失函数进行计算。反向传播。执行 AdamW 参数更新步骤。记录日志。重复上述流程。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步骤 4：样本

给定一个提示词，重复执行以下步骤：进行前向传播、从 top-p 对数概率中采样、将新生成的文本追加到当前序列中，然后继续循环。当序列长度达到 500 个标记时停止。

### 第 5 步：读取输出结果

执行 2,000 步后：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

并非莎士比亚原作，却有着类似莎剧的风格。在笔记本电脑上运行时，拥有约800万参数及6分钟的运算时间，显然具有显著优势。

## 使用它

该最终项目是一个参考架构。要将其转化为实际可用模型，需进行以下三项扩展：

1. **更换分词器。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。词汇表大小将从 65 个提升至约 50,000 个，因此需要相应提升模型容量以保持性能。
2. **在更大的语料库上训练。** 可使用 HuggingFace 提供的 `OpenWebText` 或 `fineweb-edu` 数据集。对于参数量为 125M 的 GPT 模型，在单块 A100 硬件上处理 10B 个标记大约需要 24 小时。
3. **加入 RoPE、KV 缓存及 Flash Attention 技术。** 下面的练习将逐步指导你完成这些配置。

最终得到的模型仍为参数量为 125M 的 GPT，能够生成流畅的英语文本。虽然算不上最前沿的模型，但 Karpathy、EleutherAI 以及艾伦研究所于 2026 年用于训练研究级模型的代码路径其实与此相同，仅规模更大而已。

## 发布它

请参阅 `outputs/skill-transformer-review.md`。该技能检查从零实现的 Transformer 模型在全部 13 个先前课程中的正确性。

## 练习题

1. **简单。** 运行 `code/main.py`，检查训练完成的模型在最终步骤的验证损失是否低于 2.0。将 `max_steps` 的值从 2,000 更改为 5,000 —— 验证损失是否会持续下降？
2. **中等难度。** 用 RoPE 替换学习得到的位置嵌入。在 `MultiHeadAttention` 模块中对 Q 和 K 向量应用旋转操作。进行训练并验证验证损失是否至少保持不变或进一步降低。
3. **中等难度。** 在采样循环中实现 KV 缓存功能。分别使用缓存与不使用缓存的方式生成 500 个令牌。在笔记本电脑上，实际运行时间应能提升 5–20 倍。
4. **高难度。** 在模型中增加第二个头，用于预测下一个及之后的令牌（即 DeepSeek-V3 中的 MTP —— 多令牌预测功能）。对这两个头进行联合训练。这样的改进是否有效？
5. **高难度。** 将每个块中的单个 FFN 替换为包含 4 个专家组的 MoE 结构，并采用路由器加上 top-2 路由策略。观察在相同活跃参数数量下验证损失的变化情况。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| nanoGPT | “Karpathy 的教程代码库” | 最简的仅解码器的 Transformer 训练代码，约 300 行；是该领域的标准参考。 |
| tinyshakespeare | “标准的玩具语料集” | 约 1.1 MB 的文本量；2015 年以来的所有字符级语言模型教程均使用它。 |
| Tied embeddings | “共享输入/输出矩阵” | 语言模型头权重等于词元嵌入矩阵的转置；可节省参数数量并提升模型质量。 |
| bf16 autocast | “训练精度技巧” | 以 bf16 格式执行前向/反向传播，同时将优化器状态保存为 fp32 格式；自 2021 年起成为标准做法。 |
| Gradient clipping | “抑制梯度峰值” | 将全局梯度范数限制在 1.0 以内；可防止训练过程出现爆炸性增长。 |
| Cosine LR schedule | “2020 年及以后的默认策略” | 学习率先进行线性上升（预热阶段），随后以余弦曲线形式递减至峰值的 10%。 |
| MFU | “模型 FLOP 利用率” | 实际执行的 FLOPs 数与理论峰值之比；2026 年时，40% 的密集型结构及 30% 的 MoE 结构已属于较高水平。 |
| Val loss | “保留的验证集损失” | 对模型从未见过的数据计算交叉熵损失；用于检测过拟合现象。 |

## 延伸阅读

- [带注释的 Transformer（哈佛大学自然语言处理项目）](https://nlp.seas.harvard.edu/annotated-transformer/) —— 经典的带注释实现版本。
