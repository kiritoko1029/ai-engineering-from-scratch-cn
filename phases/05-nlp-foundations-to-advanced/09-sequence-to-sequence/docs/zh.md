# 序列到序列模型

> 两个充当翻译器的 RNN。它们所遇到的瓶颈正是注意力机制产生的原因。

**类型：** 构建
**语言：** Python
**先修要求：** 第 5 阶段 · 08（用于文本的 CNN 和 RNN）、第 3 阶段 · 11（PyTorch 入门）
**时长：** 约 75 分钟

## 问题所在

分类任务将变长序列映射为单一标签，而翻译任务则将一个变长序列映射为另一个变长序列。输入与输出分别属于不同的词汇表，甚至可能是不同的语言，且长度并不一定相等。

Seq2Seq架构（Sutskever、Vinyals、Le，2014年）通过一种极为简单的方案解决了这一问题：该架构包含两个RNN。其中一个RNN负责读取源句子并生成固定大小的上下文向量；另一个RNN则读取该向量，并逐个生成目标句子的标记。其实现方式与第08课中编写的代码类似，只是组合方式有所不同。

研究这一内容有两个重要原因。首先，上下文向量瓶颈是自然语言处理领域中最具教学价值的缺陷之一，它为注意力机制及变换器等技术的出现提供了动力。其次，相关的训练方法（教师强制、定时采样以及推理时的束搜索）至今仍适用于包括大语言模型在内的所有现代生成式系统。

## 概念概述

**编码器。** 一种用于读取源句子的 RNN。其最终的隐藏状态即为**上下文向量**——是对整个输入内容进行固定长度摘要后的结果。理论上，除了源文本信息外，不应丢失任何其他内容。

**解码器。** 另一个基于上下文向量初始化的 RNN。在每一步中，它都会将之前生成的标记作为输入，并生成目标词汇表中的概率分布。通过采样或 argmax 算法选择下一个标记，然后将其反馈回解码器中。重复此过程，直到生成 `<EOS>` 标记或达到最大长度为止。

**训练方式：** 在解码器的每一步都会计算交叉熵损失，并对整个序列求和。同时通过对两个网络应用标准的反向传播算法来进行训练。

**教师强制策略。** 在训练期间，解码器在步骤 `t` 的输入应为位置 `t-1` 处的*真实*标记，而非解码器自身之前的预测结果。这一做法能够稳定训练过程；若不采用该策略，早期的错误会逐级累积，导致模型永远无法学会正确规律。而在推理阶段，则必须使用模型自身的预测结果，因此始终存在训练分布与推理分布之间的差异。这种差异被称为**曝光偏差**。

**瓶颈问题。** 编码器所学到的关于源文本的所有信息都必须被压缩到那个单一的上下文向量中。长句子会导致细节丢失，稀有词汇也会变得模糊不清。对于词序不同的情况（如“chat noir”与“black cat”），模型必须依靠记忆而非计算来处理。

注意力机制（第 10 课）通过允许解码器查看*所有*编码器的隐藏状态，而不仅仅是最后一个状态，从而解决了上述问题。这就是该机制的核心理念。

```figure
lstm-gates
```

## 构建它

### 步骤 1：编码器

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形状为 `[batch, seq_len, hidden_dim]`——每个输入位置对应一个隐藏状态。`hidden` 的形状为 `[1, batch, hidden_dim]`——代表最终步骤的状态。第 08 课提到“通过对输出进行池化操作来进行分类”。在此方法中，我们直接使用最后一个隐藏状态作为上下文向量，而忽略各个步骤的输出。

### 步骤 2：解码器

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

解码器会逐步骤被调用。输入：一批单个标记以及当前的隐藏状态。输出：下一个标记的词汇表对数概率以及更新后的隐藏状态。

### 步骤 3：带教师强制的训练循环

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有两个值得特别说明的参数。`ignore_index=0` 表示忽略填充标记带来的损失。`teacher_forcing_ratio` 则是指在每一步中选择真实标记而非模型预测结果的概率。该参数初始值为 1.0（完全强制使用真实标记），并在训练过程中逐渐降低至约 0.5，以此缩小曝光偏差。

### 第 4 步：推理循环（贪婪算法）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

贪心解码在每一步都会选择概率最高的标记。这种方法可能会出现偏差：一旦选定某个标记，就无法再更改。而**束搜索**则会保留得分最高的`k`个部分序列，并在最后选出整体得分最高的那个序列。通常，束宽设置为3到5较为合适。

### 步骤 5：瓶颈现象演示

在一个简单的复制任务上训练模型：源序列为 `[a, b, c, d, e]`，目标序列也为 `[a, b, c, d, e]`。逐步增加序列长度，并观察模型的准确率变化。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

单个GRU隐藏状态无法无损地存储长度为40个标记的输入序列。虽然每个编码器步骤都会处理这些信息，但解码器仅能获取到最新的隐藏状态。注意力机制可直接解决这一问题。

## 使用它

PyTorch 提供基于 `nn.Transformer` 和 `nn.LSTM` 的序列到序列模型模板。Hugging Face 的 `transformers` 库则提供了在数十亿个标记上训练完成的完整编码器-解码器模型（如 BART、T5、mBART、NLLB）。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

现代的编码器-解码器架构已摒弃了RNN，转而采用Transformer。其整体结构（编码器、解码器以及逐个生成标记）与2014年的seq2seq论文相同，但各模块内部的机制则有所不同。

### 何时仍需使用基于RNN的seq2seq模型

在新建项目中几乎从不采用此方法。特定例外情况包括：

- 流式翻译，即以有限内存逐个处理输入标记。
- 设备端文本生成，此时Transformer模型的内存消耗过高。
- 教学目的。理解编码器-解码器瓶颈是快速掌握Transformer为何更优的最佳途径。

### 曝光偏差及其缓解方法

- **定时采样。** 在训练过程中逐步降低教师强制比例，使模型学会从自身的错误中恢复。
- **最小风险训练。** 以句子级的 BLEU 分数而非词元级的交叉熵进行训练。更贴近实际需求。
- **强化学习微调。** 使用特定指标对序列生成器给予奖励。广泛应用于现代大语言模型的 RLHF 过程中。

以上三种方法均适用于基于 Transformer 的生成模型。

## 发布它

保存为 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 练习题

1. **简单。** 实现一个简单的复制任务。使用输入输出对训练 GRU seq2seq 模型，其中目标序列与源序列相同。在长度为 5、10、20 的情况下测量准确率，并重现性能瓶颈现象。
2. **中等难度。** 添加束搜索解码功能，束宽设置为 3。在一个小型平行语料库上使用 BLEU 分数衡量束搜索与贪婪算法的性能差异。记录束搜索在哪些情况下表现更好（通常体现在最后几个词上），以及在哪些情况下两者没有区别。
3. **高难度。** 在一个包含 10,000 对释义数据的集子上对 `facebook/bart-base` 模型进行微调。将微调后模型在束宽为 4 的情况下的输出与基础模型的输出在保留的测试数据上进行比较。报告 BLEU 分数，并挑选出 10 个具有代表性的示例。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 编码器 | 输入 RNN | 负责读取源文本，逐步生成隐藏状态以及最终的上下文向量。 |
| 解码器 | 输出 RNN | 以上下文向量为初始状态，依次生成目标标记。 |
| 上下文向量 | 摘要信息 | 即编码器的最终隐藏状态，具有固定大小，这是瓶颈注意力机制所要解决的问题。 |
| 教师强制 | 使用真实标记 | 在训练过程中输入真实的上一轮标记，以此稳定学习过程。 |
| 暴露偏差 | 训练/测试差距 | 基于真实标记训练的模型从未练习过如何纠正自身的错误。 |
| 波束搜索 | 更优的解码方式 | 在每一步保留前 k 个最佳的部分序列，而非一味采用贪婪策略。 |

## 延伸阅读

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 最初的 seq2seq 相关论文，共四页。
- [Cho 等人 (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 提出了 GRU 以及编码器-解码器架构。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 关于注意力机制的论文，建议在本课之后立即阅读。
- [PyTorch NLP 从零开始教程](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可直接运行的 seq2seq + 注意力机制示例代码。
