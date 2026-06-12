# 用于文本处理的 CNN 与 RNN

> 卷积操作用于学习 n-gram 模式，而递归操作则负责记忆。这两种方法均已被注意力机制所取代。但在硬件资源受限的情况下，它们依然具有重要意义。

**类型：** 构建
**语言：** Python
**先修课程：** 第 3 阶段 · 11（PyTorch 入门）、第 5 阶段 · 03（词嵌入）、第 4 阶段 · 02（从零实现卷积）
**时长：** 约 75 分钟

## 问题所在

TF-IDF与Word2Vec生成的向量是扁平的，且不考虑词序。基于这些方法的分类器无法区分“狗咬人”和“人咬狗”。在某些情况下，词序本身就蕴含着重要信息。

在Transformer出现之前，有两类架构填补了这一空白。

**文本卷积网络（TextCNN）**：对词嵌入序列应用一维卷积操作。宽度为3的滤波器可作为一种可学习的三词组检测器：它覆盖三个词并输出一个得分值。通过堆叠不同宽度的滤波器（2、3、4、5），可以检测多尺度模式。最后通过最大池化得到固定大小的表示形式。这类网络具有扁平结构、并行处理能力且运算速度快。

**循环神经网络（RNN、LSTM、GRU）**：逐个处理序列中的元素，同时维护一个隐藏状态以传递信息。这类网络具备顺序处理能力、记忆功能，并且对输入长度的适应性较强。从2014年到2017年，它们在序列建模领域占据主导地位，之后注意力机制应运而生。

本课将介绍这两类架构，并阐明促使注意力机制诞生的缺陷所在。

## 概念概述

**TextCNN**（Kim，2014）。首先对词元进行嵌入处理。宽度为`k`的一维卷积操作会用滤波器遍历连续的`k`元嵌入序列，从而生成特征图。随后通过对该特征图进行全局最大池化，选出最强的激活值。将多个不同滤波器宽度下的最大池化输出结果拼接起来，再输入到分类器中。

其工作原理如下：滤波器实际上是一个可学习的n-gram模型。由于最大池化操作与位置无关，因此无论在评论的开头还是中间，“负面”信息都会触发相同的特征值。使用三种不同宽度的滤波器，每种宽度包含100个滤波器，即可获得300个可学习的n-gram检测器。训练过程是并行的，不存在序列依赖关系。

**RNN**。在每个时间步`t`中，隐藏状态`h_t = f(W * x_t + U * h_{t-1} + b)`。参数`W`、`U`和`b`会在不同时间步之间共享。到时间步`T`时的隐藏状态即为整个序列前缀的摘要。在进行分类时，需要对`h_1 ... h_T`这些隐藏状态进行池化处理（可采用最大值、平均值或最后一个值）。

普通的RNN存在梯度消失的问题。**LSTM**通过引入门控机制来决定哪些信息需要遗忘、哪些需要保留以及哪些需要输出，从而在长序列中稳定梯度。而**GRU**则将LSTM的复杂结构简化为两个门控，以更少的参数实现类似的功能。

**双向RNN**会同时向前和向后运行两个RNN，并将它们的隐藏状态拼接在一起。这样，每个词元的表示就能同时获取左右两侧的上下文信息，对于文本标注任务而言至关重要。

```figure
rnn-unroll
```

## 构建它

### 步骤 1：PyTorch 中的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 会将形状为 `[batch, seq_len, embed_dim]` 的数据重新排列为 `[batch, embed_dim, seq_len]`，因为 `nn.Conv1d` 将中间轴视为通道维度。无论输入长度如何，池化后的输出都具有固定尺寸。

### 步骤 2：LSTM 分类器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

对整个序列进行最大池化，而非仅使用最后状态。在分类任务中，最大池化通常优于直接使用最后一个隐藏状态，因为长序列末尾的信息往往会对最终状态产生主导影响。

### 步骤 3：梯度消失演示（直观理解）

没有门控机制的普通RNN无法学习长距离依赖关系。以一个简单的任务为例：预测序列中是否出现标记`A`。如果`A`位于第1位，而序列长度为100个标记，则损失函数的梯度必须通过99次循环权重的乘法传递回去。若该权重小于1，梯度将会消失；若大于1，则梯度会出现爆炸现象。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

长短期记忆网络（LSTM）通过引入**单元状态**来解决这一问题，该状态在网络中仅通过加法运算进行传递（遗忘门虽会对其进行乘法缩放，但梯度依然沿着这条“通道”流动）。门控循环单元（GRU）则采用类似机制，但参数量更少。这两种结构都能确保在超过100步的序列训练过程中保持稳定性。

### 步骤 4：为何这仍然不够

即便使用了 LSTM，仍有三个问题存在。

1. **序列处理瓶颈。** 对长度为 1000 的序列进行 RNN 训练时，需要执行 1000 次串行的前向/反向传播步骤，无法在时间维度上进行并行化处理。
2. **编解码器结构中的固定大小上下文向量。** 解码器只能看到经过压缩后的编码器最终隐藏状态，该状态包含了整个输入的信息。过长的输入会导致细节丢失。第 09 节将直接探讨这一问题。
3. **远距离依赖关系的处理精度上限。** 尽管 LSTM 的性能优于普通 RNN，但在需要传递超过 200 步的特定信息时仍存在困难。

注意力机制解决了以上三个问题。而 Transformer 则完全摒弃了循环结构。第 10 节将是理解这一转变的关键所在。

## 使用它

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 已具备可直接投入生产的成熟度，其训练代码也属于标准范式。

Hugging Face 提供了预训练嵌入向量，可直接将其作为输入层使用：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

符合约束条件时的选用清单。

- **边缘端/设备端推理。** 使用 GloVe 嵌入的 TextCNN 模型体积仅为 Transformer 的 1/10 至 1/100。若部署目标为手机，此方案最为合适。
- **流式/在线分类任务。** RNN 一次仅能处理一个词元，而 Transformer 需要完整的序列信息。对于实时输入的文本，LSTM 仍具有优势。
- **用于基准测试的极小模型。** 便于快速迭代新任务。可在 CPU 上用 5 分钟时间训练出 TextCNN 模型。
- **数据量有限的序列标注任务。** BiLSTM-CRF（第 06 课）仍是处理 1k 至 10k 条带标签句子的成熟 NER 架构。

其余所有场景均应选用 Transformer。

## 发布它

保存为 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 练习题

1. **简单。** 在一个包含3类的小型测试数据集上训练TextCNN（数据可自行生成）。验证滤波器宽度为2、3、4时的平均F1分数是否优于仅使用宽度为3的情况。
2. **中等难度。** 为LSTM分类器实现最大池化、均值池化和最后状态池化三种池化方法。在小型数据集上进行对比，记录哪种池化方法的性能最优，并分析原因。
3. **高难度。** 构建一个BiLSTM-CRF命名实体识别系统（结合第06课的内容与本课内容）。使用CoNLL-2003数据集进行训练。将该模型与第06课中的纯CRF基线模型以及经过微调的BERT模型进行对比，报告训练时间、内存占用及F1分数。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| TextCNN | 文本专用 CNN | 在词嵌入上堆叠一维卷积层，并使用全局最大池化操作。Kim（2014）。 |
| RNN | 循环神经网络 | 在每个时间步更新隐藏状态：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 门控循环神经网络 | 包含输入门、遗忘门和输出门以及细胞状态，能够稳定地训练长序列数据。 |
| GRU | 简化版 LSTM | 只有两个门而非三个，准确率相近但参数更少。 |
| Bidirectional | 双向处理 | 将正向与反向 RNN 连接起来，使每个词都能感知其上下文的两侧信息。 |
| Vanishing gradient | 梯度消失 | 在普通 RNN 中，权重若持续小于 1 的数值相乘，会导致早期步骤的梯度几乎为零。 |

## 延伸阅读

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — 即 TextCNN 论文。共八页，通俗易懂。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 即 LSTM 论文。阐述清晰，出人意料。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 那些让所有人都能理解 LSTM 的图表。
