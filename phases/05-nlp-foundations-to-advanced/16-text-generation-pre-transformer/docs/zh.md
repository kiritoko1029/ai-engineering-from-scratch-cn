# Transformer 之前的文本生成技术——N-gram语言模型

> 若某个单词的“惊讶度”较高，则说明模型性能不佳。困惑度将这种“惊讶感”转化为数值，而平滑处理则确保该数值为有限值。

**类型：** 构建
**语言：** Python
**先修课程：** 第 5 阶段 · 01（文本处理）、第 2 阶段 · 14（朴素贝叶斯）
**耗时：** 约 45 分钟

## 问题所在

在 Transformer、RNN 以及词嵌入出现之前，语言模型是通过统计某个单词在前 `n-1` 个单词之后出现的频率来预测下一个单词的。例如，“the cat”后接“sat”的次数为 47 次，“the cat”后接“jumped”的次数为 12 次，而“the cat”后接“refrigerator”的次数则为 0 次。通过对这些频率进行归一化处理，即可得到概率分布。

这就是 n-gram 语言模型。从 1980 年到 2015 年间，所有的语音识别系统、拼写检查工具以及基于短语的机器翻译系统都采用了这种模型。即便在如今需要低成本设备端语言建模的场景中，它依然被广泛使用。

有趣的问题在于如何处理那些未曾出现过的 n-gram 序列。基于原始计数的模型会将其从未见过的序列的概率设为零，这会导致严重问题，因为句子通常很长，几乎每个长句都至少包含一个未曾出现的序列序列。经过五十年的平滑技术研究，这一问题得到了解决。Kneser-Ney 平滑算法便是其成果，而现代深度学习也继承了这一基于实证的传统。

## 概念概述

![N-gram模型：计数、平滑处理、生成](../assets/ngram.svg)

**N-gram概率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。需确定`n`的值（三元语法通常为3，四元语法则为4）。可通过计数方式计算得出：

```text
P(w | context) = count(context, w) / count(context)
```

**零计数问题。** 任何在训练数据中未出现过的 n-gram 的概率都会被设为零。2007 年针对 Brown 语料库的一项研究显示，即便是 4-gram 模型，也有 30% 的保留测试用 4-gram 在训练集中并不存在。若不采用平滑技术，则无法对任何真实文本进行评估。

**按复杂度排序的平滑方法：**

1. **拉普拉斯平滑（加一法）。** 给每个计数值加上 1。实现简单，但对罕见事件的处理效果极差。
2. **古丁平滑。** 根据频率的频率分布，将概率权重从高频事件重新分配到未出现的事件上。
3. **插值法。** 使用可调节的权重，结合 n-gram、(n-1)-gram 等不同阶数的估计结果。
4. **回退法。** 若某 n-gram 的计数为零，则回退至 (n-1)-gram 进行预测。Katz 回退法对这一过程进行了标准化处理。
5. **绝对折扣法。** 从所有计数值中减去固定的折扣量 `D`，然后将其重新分配给未出现的事件。
6. **Kneser-Ney 平滑。** 在绝对折扣法的基础上，巧妙地选择了低阶模型：使用*延续概率*（即一个词出现的上下文数量）而非原始频率作为依据。

Kneser-Ney 方法的洞察十分深刻。“San Francisco”是一个常见的二元词组，而单词“Francisco”大多出现在“San”之后。朴素的绝对折扣法会因为“Francisco”的出现次数较多而赋予其较高的单词概率。Kneser-Ney 方法则注意到“Francisco”仅出现在一个上下文中，因此相应地降低了其延续概率。这样一来，以“Francisco”结尾的新的二元词组就会获得恰当的较低概率。

**评估指标：困惑度。** 指在保留的测试集上，每个单词的平均负对数似然值的指数形式。数值越低表示模型性能越好。困惑度为 100 意味着模型的表现相当于随机从 100 个词中选择一样。

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

```figure
ngram-backoff
```

## 构建它

### 步骤 1：三字词频统计

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

输入为一个分词后的句子列表。输出为n元语法计数和上下文计数。<s>与</s>表示句子边界。

### 步骤 2：拉普拉斯平滑

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

将所有计数值加 1。虽然能够平滑数据，但会为未观测到的事件过度分配质量，进而影响那些出现频率较低的已知事件。

### 步骤 3：Kneser-Ney（二元词，插值法）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三个运动部件。`continuation_prob`用于表示“该单词出现在多少种不同的上下文中？”（即 Kneser-Ney 创新机制）。`lambda_prev`是折扣作用后释放的权重，用于对回退值进行加权。最终概率为经过折扣处理的主项与加权后的延续项之和。

### 步骤 4：通过采样生成文本

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

按概率比例采样。每次使用不同的种子时都会产生不同的输出结果。若需类似束搜索的输出，可在每一步选择最大值（贪婪策略），并加入一个小的随机性控制参数（温度）。

### 步骤 5：困惑度

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

数值越低越好。以 Brown 语料库为例，经过良好调优的 4-gram KN 模型的困惑度约为 140。而在同一测试集上，Transformer 类语言模型的困惑度则在 15 到 30 之间。两者之间的差距约为 10 倍。正是这一巨大差距推动了该领域的技术发展。

## 使用它

- **传统 NLP 教学内容。** 最直观地介绍平滑处理、最大似然估计以及困惑度等概念。
- **KenLM。** 用于生产环境的 n-gram 库，在对延迟要求较高的语音识别和机器翻译系统中被用作重评分工具。
- **设备端自动补全功能。** 键盘中的三词模型技术，至今仍在使用。
- **基准测试方法。** 在宣称你的神经网络语言模型性能优异之前，务必先计算其 n-gram 语言模型的困惑度。如果你的变换器模型未能以较大优势超越 KenLM，则说明存在问题。

## 发布它

保存为 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 练习题

1. **简单级。** 使用包含1,000句莎士比亚作品的语料库训练三词单元语言模型（trigram LM），生成20句话。这些句子在局部语境下看似合理，但从整体来看则缺乏连贯性。这是该技术的标准演示示例。
2. **中等级。** 在预留的莎士比亚作品子集上为你的KN模型实现困惑度计算功能，并与拉普拉斯模型进行对比。你应该能够观察到KN模型的困惑度降低30-50%。
3. **高级别。** 构建一个三词单元拼写校正器：给定一个拼写错误的单词及其上下文，根据语言模型计算各种可能的修正方案，并按上下文概率对它们进行排序。在公开的Birkbeck拼写语料库上进行性能评估。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| N-gram | 单词序列 | 长度为 `n` 的连续标记序列。 |
| 平滑处理 | 避免概率为零 | 重新分配概率值，使未观测到的事件也具有非零概率。 |
| 混乱度 | 语言模型质量指标 | 基于保留数据计算的 `exp(-平均对数概率)`。数值越低表示质量越好。 |
| 回退策略 | 转而使用更短上下文 | 若三词序列的数量为零，则改用二元序列。Katz回退算法对此进行了形式化定义。 |
| Kneser-Ney算法 | 适用于N-gram的最佳平滑方法 | 结合绝对折扣因子以及低阶模型的延续概率。 |
| 延续概率 | Kneser-Ney算法特有的概念 | `P(w)` 的计算权重取决于标记 `w` 出现的上下文数量，而非单纯的出现次数。 |

## 延伸阅读

- [Jurafsky 和 Martin —— 《语音与语言处理》，第3章（2026年草案）](https://web.stanford.edu/~jurafsky/slp3/3.pdf) —— 关于n-gram语言模型及平滑技术的经典论述。
- [Chen 和 Goodman (1998)。《语言建模中平滑技术的实证研究》](https://dash.harvard.edu/handle/1/25104739) —— 确立Kneser-Ney算法为最佳n-gram平滑器的论文。
- [Kneser 和 Ney (1995)。《M-gram语言建模中改进的回退策略》](https://ieeexplore.ieee.org/document/479394) —— Kneser-Ney算法的原始论文。
- [KenLM](https://kheafield.com/code/kenlm/) —— 高效的生产级n-gram语言模型，截至2026年仍被用于对延迟敏感的应用场景。
