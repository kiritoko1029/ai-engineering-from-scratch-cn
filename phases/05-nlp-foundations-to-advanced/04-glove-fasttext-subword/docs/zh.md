# GloVe、FastText与子词嵌入模型

> Word2Vec 为每个单词训练一个嵌入向量。GloVe 对共现矩阵进行分解处理。FastText 则对单词片段进行嵌入。BPE 技术则为后续的 Transformer 模型奠定了基础。

**类型：** 构建
**语言：** Python
**先修要求：** 第 5 阶段 · 03（从零实现 Word2Vec）
**耗时：** 约 45 分钟

## 问题所在

Word2Vec 留下了两个未解之谜。

首先，有一类研究路线直接对共现矩阵进行分解（如 LSA、HAL），而非采用在线跳字模型更新方式。Word2Vec 的迭代方法是否本质上更优，还是两者之间的差异仅源于处理词频的方式不同？**GloVe** 给出了答案：通过精心选择的损失函数进行矩阵分解，其效果可与 Word2Vec 相媲美甚至更佳，且训练成本更低。

其次，这两种方法都无法为从未见过的单词提供表示。无论是 `Zoomer-approved`、`dogecoin` 这类新创造的专有名词，还是上周刚出现的词汇，以及罕见词根的各种变形，它们都难以被处理。**FastText** 通过嵌入字符 n-gram 解决了这一问题：一个单词是其各个组成部分的总和，包括语素，因此即便是词汇表外的单词也能获得合理的向量表示。

第三，随着Transformer模型的出现，问题又发生了变化。基于词的词汇表规模通常在百万条左右；而真实语言的复杂性远超此限。**字节对编码（BPE）** 及其衍生技术通过学习涵盖所有内容的常见子词单元来解决这一问题。如今，所有现代大型语言模型的分词器均为子词分词器。

本课将依次介绍这三种方法，并说明在不同场景下应选择哪种。

## 概念概述

**GloVe（全局向量模型）**。构建词与词之间的共现矩阵 `X`，其中 `X[i][j]` 表示词 `j` 在词 `i` 的上下文中出现的频率。训练向量使得满足 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。对损失函数进行加权处理，以避免高频词对整体结果产生主导影响。完成。

**FastText**。一个词由其字符n-gram序列与词本身之和构成。例如，`where` 可表示为 `<wh, whe, her, ere, re>, <where>`。该词的向量即为这些组成向量的总和。训练方式类似于Word2Vec。其优势在于：未知词（如 `whereupon`）可以由已知的n-gram序列组合而成。

**BPE（字节对编码）**。从单个字节（或字符）的词汇表开始。统计语料库中所有相邻的字节对，将出现频率最高的字节对合并为一个新标记。重复此过程 `k` 次。最终得到的词汇表包含 `k + 256` 个标记：高频序列（如 `ing`、`tion`、`the`）被视作单个标记，而生僻词则会被拆解为常见的组成部分。这样，每个句子都能被分解为相应的标记。

## 构建它

### GloVe：对共现矩阵进行分解

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有两个值得特别提及的组成部分。权重函数 `f(x) = (x/x_max)^alpha` 会降低出现频率极高的词对（如 `(the, and)`）的权重，防止其主导损失函数的结果。最终的嵌入向量是中心表 `W` 与上下文表 `W_tilde` 的之和。将两者相加是一种已被广泛应用的技巧，其效果通常优于仅使用其中一种表。

### FastText：具备分词意识的语义嵌入模型

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每个单词都由其对应的n元组（通常为3到6个字符）来表示。该单词的嵌入向量即为这些n元组嵌入向量的总和。在skip-gram训练中，只需将此公式代入原本Word2Vec使用单个向量所在的位置即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

对于一个未见过的单词，只要其某些n-gram是已知的，依然可以生成对应的向量。`whereupon`与`where`共享子词`<wh`、`her`、`ere`以及`<where>`，因此这两个词的向量会出现在彼此相近的位置。

### BPE：学习到的子词词汇表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一轮迭代会合并出现频率最高的相邻字符对。经过足够多的迭代后，高频子串（如 `low`、`est`、`tion`）会变成独立的标记，而低频词汇则会被清晰地拆分。

真正的 GPT / BERT / T5 分词器会学习 3 万到 10 万个合并规则。其结果是：任何文本都能被分词为长度有限的已知 ID 序列，从而彻底避免出现未知词汇（OOV）的情况。

## 使用它

在实践中，你很少会自行训练这些模型。通常都是加载预训练的检查点文件。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 Transformer 时代用于 BPE 风格的子词分词：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

前缀 `Ġ` 用于标记词边界（这是 GPT-2 的约定）。所有现代分词器均为 BPE 变体、WordPiece（BERT）或 SentencePiece（T5、LLaMA）的一种。

### 何时选择哪种

| 情况 | 推荐选择 |
|-----------|----------|
| 预训练的通用词向量，无需处理未知词汇 | GloVe 300d |
| 预训练的通用词向量，需能够处理拼写错误、新造词及形态复杂的语言 | FastText |
| 任何输入 Transformer 模型的数据（无论是训练还是推理阶段） | 使用模型自带的分词器。切勿更换。 |
| 从零开始训练自定义语言模型 | 首先在自有语料库上训练 BPE 或 SentencePiece 分词器 |
| 基于线性模型的生产环境文本分类任务 | 仍应使用 TF-IDF。参见第 02 节。 |

## 发布它

另存为 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 练习题

1. **简单。** 运行 `char_ngrams("playing")` 和 `char_ngrams("played")`，并计算这两个 n-gram 集合的杰卡德重叠度。你应该会看到大量相同的子串（如 `pla`、`lay`、`play`），这正是 FastText 能够很好地处理词形变体的原因。
2. **中等难度。** 扩展 `learn_bpe` 函数以记录词汇量的增长情况。将每个语料库字符对应的标记数作为合并次数的函数进行绘图。你应该会观察到初期词汇量快速压缩，最终趋于每个标记约 2-3 个字符的水平。
3. **高难度。** 使用 1k 次合并的 BPE 对莎士比亚的全部作品进行训练。比较常见单词与罕见专有名词的分词结果，并测量处理前后的平均每个单词的标记数。总结一下让你感到意外之处。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 共现矩阵 | 单词频率表 | `X[i][j]` = 单词 `i` 周围窗口中单词 `j` 出现的频率。 |
| 子词 | 单词的片段 | n-gram 字符序列（FastText）或学习得到的标记（BPE/WordPiece/SentencePiece）。 |
| BPE | 字节对编码 | 通过迭代合并出现频率最高的相邻字节对，直到词汇表大小达到目标值。 |
| OOV | 词汇表外词 | 模型从未见过的单词。Word2Vec/GloVe 无法处理此类词，而 FastText 和 BPE 可以处理。 |
| 字节级 BPE | 对原始字节的 BPE 编码 | GPT-2 采用的方案。词汇表从 256 个字节开始构建，因此不存在词汇表外词。 |

## 延伸阅读

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 论文，共七页，仍是关于该损失函数最完善的推导。
- [Bojanowski 等人 (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 将 BPE 引入现代自然语言处理领域的论文。
- [Hugging Face 分词器概述](https://huggingface.co/docs/transformers/tokenizer_summary) — 实际应用中 BPE、WordPiece 与 SentencePiece 的具体差异。
