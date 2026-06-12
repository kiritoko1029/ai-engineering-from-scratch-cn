# 词嵌入——从零实现 Word2Vec

> 人如其友。若用这一理念训练浅层神经网络，几何特性便会自然显现。

**类型：** 构建
**语言：** Python
**先修要求：** 第5阶段 · 02（词袋模型 + TF-IDF），第3阶段 · 03（从零实现反向传播）
**耗时：** 约75分钟

## 问题所在

TF-IDF 能识别“dog”与“puppy”是不同的单词，但它无法意识到二者含义几乎相同。基于“dog”训练的分类器无法迁移到对“puppy”的评论分析上。虽然可以通过列出同义词来弥补这一问题，但这种方法在处理生僻词汇、领域术语以及未预料到的各种语言时都会失效。

我们需要一种表示方法，使得“dog”与“puppy”在向量空间中距离相近，“king - man + woman”的结果接近“queen”，从而使基于“dog”训练的模型能够自动将部分特征传递给“puppy”。

Word2Vec 为我们提供了这样的向量空间。它是一种双层神经网络，通过万亿级词元的训练数据得出，于2013年发表。其架构简单到近乎令人惊讶，却为自然语言处理领域带来了十年的变革。

## 概念概述

**分布假设**（Firth，1957）：“通过一个词出现的周围词汇，即可推断该词的含义。”如果两个词出现在相似的上下文中，它们很可能表示相似的意义。

Word2Vec 有两种实现方式，均基于这一理念。

- **Skip-gram。** 给定中心词，预测其周围的词汇。以窗口大小为 2 为例，`cat -> (the, sat, on)`。
- **CBOW（连续词袋模型）。** 给定周围的词汇，预测中心词。（`the, sat, on`）-> `cat`。

Skip-gram 的训练速度较慢，但对罕见词的处理能力更强。它后来成为了默认选择。

该网络包含一个没有非线性函数的隐藏层。输入为词汇表对应的独热向量，输出为词汇表的 softmax 概率分布。训练完成后，可舍弃输出层，隐藏层的权重即为词嵌入。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

关键技巧：对10万个词汇应用softmax计算成本极高。Word2Vec通过**负采样**将其转化为二分类任务，即预测“该上下文词是否出现在该中心词附近”，答案仅为“是”或“否”。对于每对训练样本，只需抽取少量不共现的负样本词，而无需对整个词汇表进行softmax计算。

```figure
word-vector-arithmetic
```

## 构建它

### 步骤 1：从语料库中提取训练对

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

窗口中的每一对（中心点，上下文）都构成一个正样本训练数据。

### 步骤 2：嵌入表格

两个矩阵。`W` 是中心词嵌入表（即需要保留的表格）。`W'` 是上下文词表（通常会被丢弃，有时会与 `W` 取平均值）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

小型随机初始化。词汇量1万、维度为100是较为现实的设定；用于教学时，50个词汇搭配16个维度便足以展示其结构特征。

### 步骤 3：负采样目标函数

对于每一对正样本 `(center, context)`，从词汇表中随机抽取 `k` 个单词作为负样本。训练模型，使得正样本的点积 `W[center] · W'[context]` 值较高，而负样本的点积值较低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

核心公式：正样本对对应的逻辑损失函数值（期望sigmoid函数值接近1）加上负样本对对应的逻辑损失函数值（期望sigmoid函数值接近0）。梯度会同时流向这两个表。完整的推导过程见原始论文；若希望牢记，建议用纸笔过一遍推导过程。

### 步骤 4：在小型语料库上进行训练

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型语料库上经过足够多的训练轮次后，具有相似上下文的词会拥有相近的嵌入向量中心值。在小型语料库中，这一效应较为微弱；而在包含数十亿个标记的语料库中，则会表现得极为显著。

### 第 5 步：类比技巧

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

基于预训练的300维Google News向量：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。这并非因为模型理解了王室的概念，而是因为向量 `(king - man)` 能够表示类似“皇室”的特征，将其与 `woman` 相加后便位于代表皇室女性的区域附近。

## 使用它

从零开始实现 Word2Vec 是一种教学手段。实际生产环境中的自然语言处理任务通常使用 `gensim` 库。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

在实际工作中，几乎从不自行训练 Word2Vec 模型，而是直接下载预训练好的向量。

- **GloVe** — 斯坦福大学提出的基于共现矩阵分解的方法。提供 50d、100d、200d、300d 等不同维度的模型检查点，具有较好的通用性。第 04 课将专门讲解 GloVe。
- **fastText** — Facebook 对 Word2Vec 的扩展版本，能够嵌入字符 n-gram。通过组合子词来处理词汇表之外的单词。内容见第 04 课。
- **基于 Google News 预训练的 Word2Vec 模型** — 维度为 300d，词汇量达 300 万，于 2013 年发布，至今仍被频繁下载使用。

### 当 Word2Vec 在 2026 年依然占据优势时

- 轻量级的领域专用检索。可在笔记本电脑上用一小时时间使用医学摘要进行训练，生成通用模型无法捕捉的专用向量。
- 类比式特征工程。`gender_vector = mean(man - woman pairs)`。通过从其他词汇中减去该值来构建性别中性轴。该方法至今仍被用于公平性研究。
- 可解释性强。100维的空间尺寸足够小，可通过PCA或t-SNE进行可视化，并清晰观察到聚类的形成。
- 任何场景下的推理都必须在无GPU的设备上本地完成。Word2Vec的查找操作仅需检索单行数据即可。

### Word2Vec 的局限性

多义性难题。“bank”一词仅对应一个向量，而“river bank”与“financial bank”共享该向量，“table”（指电子表格或家具）也共享同一向量。下游的分类器无法根据该向量区分不同含义。

上下文嵌入模型（如ELMo、BERT以及之后的所有Transformer模型）通过根据单词周围的上下文为每个出现次数生成不同的向量来解决这一问题。这正是Word2Vec向BERT发展的关键：从静态向量转向上下文相关向量。第7阶段将讲解Transformer模型部分内容。

另一个缺陷是词汇表外词问题。如果训练数据中不存在“Zoomer-approved”这样的词汇，Word2Vec将无法识别它，且没有备用方案。fastText则通过子词组合技术解决了这一问题（见第04课）。

## 发布它

保存为 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 练习题

1. **简单级。** 在一个极小的语料库上运行训练循环（20句关于猫和狗的句子）。经过200个周期后，验证`nearest(vocab, W, W[vocab["cat"]])`在Top 3结果中是否返回`dog`。如果不是，则增加周期数或扩大词汇表规模。
2. **中等级。** 增加对高频词的子采样处理。频率高于`10^-5`的词会以与其频率成正比的概率从训练对中被移除。测量该操作对稀有词相似度的影响。
3. **高级别。** 在20个新闻组语料库上训练模型。计算两个偏见轴：`he - she`和`doctor - nurse`。将职业相关词汇投影到这两个轴上，并报告哪些职业的偏见差距最大。这类方法正是公平性研究人员所使用的检测手段。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 单词嵌入 | 向量形式的单词 | 一种从上下文中学习得到的密集、低维（通常为100-300维）的表示形式。 |
| Skip-gram | Word2Vec的技巧 | 从中心词预测上下文词。其速度慢于CBOW，但对罕见词的处理效果更好。 |
| 负采样 | 训练中的捷径 | 用针对`k`个随机词的二元分类来替代对整个词汇表的softmax计算。 |
| 静态嵌入 | 每个单词一个向量 | 不论上下文如何都使用相同的向量。无法处理多义词问题。 |
| 上下文嵌入 | 具有上下文敏感性的向量 | 根据周围词语的不同，为每个词的出现生成不同的向量。即Transformer模型所生成的向量形式。 |
| OOV | 词汇表外 | 训练数据中未出现过的单词。Word2Vec无法为这类单词生成向量。 |

## 延伸阅读

- [Mikolov 等人 (2013). 单词与短语的分布式表示及其组合性](https://arxiv.org/abs/1310.4546) —— 该论文提出了负采样方法，内容简短且易于理解。
- [Rong, X. (2014). word2vec 参数学习详解](https://arxiv.org/abs/1411.2738) —— 若原论文中的数学推导较为复杂，本文提供了最清晰的梯度推导过程。
- [gensim Word2Vec 教程](https://radimrehurek.com/gensim/models/word2vec.html) —— 实际可用的生产环境训练配置。
