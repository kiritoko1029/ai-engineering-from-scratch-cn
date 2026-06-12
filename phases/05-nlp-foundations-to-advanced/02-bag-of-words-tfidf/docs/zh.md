# 词袋模型、TF-IDF 与文本表示方法

> 先统计，后思考。在 2026 年，对于定义明确的任务，TF-IDF 依然优于嵌入模型。

**类型：** 构建
**语言：** Python
**先修课程：** 第 5 阶段 · 01（文本处理）、第 2 阶段 · 02（从零实现线性回归）
**耗时：** 约 75 分钟

## 问题所在

该模型需要数值，而你手中只有字符串。

每一个自然语言处理流程都必须回答同一个问题：如何将长度可变的词元流转换为分类器能够处理的固定尺寸向量。最初出现的解决方案是最简单且可行的——统计单词数量并构建一个向量。

在众多实际应用中，这种向量所承担的自然语言处理任务比任何嵌入模型都要多。从垃圾邮件过滤器、主题分类器、日志异常检测，到搜索排序（BM25出现之前），再到早期的情绪分析方法以及最初十年的学术自然语言处理基准测试，无不涉及其应用。即便在2026年，处理简单分类任务时，从业者仍会首先选择这种方法。它速度快、可解释性强，在那些以单词是否存在为关键因素的任务上，其表现往往与参数量为4亿的嵌入模型几乎没有差别。

本课将从零开始构建词袋模型，随后介绍TF-IDF算法。接着展示如何用三行代码在scikit-learn中实现相同功能。最后会指出促使人们转向使用嵌入模型的缺陷所在。

## 概念概述

**词袋模型（BoW）**会忽略词语的顺序。对于每篇文档，统计词汇表中每个单词出现的次数。向量的长度即为词汇表的大小，位置 `i` 对应的是单词 `i` 的出现次数。

**TF-IDF**是对词袋模型的加权改进。那些在所有文档中都出现的单词信息量较低，因此会被降低权重；而那些在整个语料库中较为罕见但在某篇文档中出现频率较高的单词则具有较高价值，因此会被提高权重。

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

其中，`TF` 表示文档中的词频，`df` 表示文档频率（包含该词的文档数量），`N` 表示总文档数。使用 `log` 可以限制那些出现频率极高的词汇的权重。

核心特性：这两种方法都能生成具有可解释性维度的稀疏向量。通过查看训练好的分类器的权重，可以了解是哪些词促使文档偏向某一类别。而 768 维度的 BERT 嵌入则无法做到这一点。

```figure
bow-tfidf
```

## 构建它

### 步骤 1：构建词汇表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

输入：分词后的文档列表（任何单词级分词器均可；本课中的 `code/main.py` 使用了简化的小写版本）。输出： `{word: index}` 字典。稳定的插入顺序意味着索引为 0 的单词是第一个文档中出现的第一个单词。具体惯例因实现而异；scikit-learn 会按字母顺序排序。

### 步骤 2：词袋模型

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

行代表文档，列代表词汇索引。条目 `[i][j]` 的含义是“词 `j` 在文档 `i` 中出现的次数”。文档 1 中包含两次 `cat`，因为该词确实出现；而文档 0 中没有出现一次 `ran`，因为该词并未出现。

### 步骤 3：词频与文档频率

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

值得提及的两种平滑技巧。使用 `(n+1)/(d+1)` 可以避免出现 `log(x/0)` 的错误。末尾的 `+1` 确保每个文档中的词仍具有 IDF 值 1（而非 0），这与 scikit-learn 的默认设置一致。其他实现则直接使用 `log(N/df)`。这两种方法都有效，但平滑处理后的版本更为友好。

### 步骤 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三份文档，五个词汇词条（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出现在所有三份文档中，因此其 IDF 值较低。`dog` 仅出现在一份文档中，因此其 IDF 值较高。这些向量属于稀疏向量（大多数元素数值较小），且具有区分度的词汇会显得尤为突出。

### 步骤 5：对行进行 L2 归一化

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

若不进行归一化，较长的文档会产生更大的向量，从而主导相似度得分。L2归一化可将所有文档映射到单位超球面上。此时行与行之间的余弦相似度就等同于点积运算。

## 使用它

scikit-learn 提供的是生产环境版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 可在一次调用中完成分词、构建词汇表以及词袋模型构建。`TfidfVectorizer` 则会在此基础上添加 IDF 加权与 L2 归一化处理。二者均返回稀疏矩阵。对于 10 万份文档而言，密集版本无法容纳在内存中；应始终保持稀疏格式，直到分类器要求转换为密集格式为止。

影响所有参数的关键设置：

| 参数 | 效果 |
|-----|------|
| `ngram_range=(1, 2)` | 包含二元词组。通常有助于提升分类性能。 |
| `min_df=2` | 移除在少于 2 份文档中出现的单词，可减少噪声数据对词汇表的影响。 |
| `max_df=0.95` | 移除在超过 95% 的文档中出现的单词，无需硬编码停用词列表即可实现类似功能。 |
| `stop_words="english"` | scikit-learn 内置的停用词列表。具体应用场景需灵活调整——情感分析时不应移除否定词。 |
| `sublinear_tf=True` | 使用 `1 + log(tf)` 代替原始的 `tf` 值。当某个术语在单份文档中出现多次时，此设置更为有效。 |

### 当 TF-IDF 依然占据优势时（截至 2026 年）

- 垃圾信息检测、主题标注、日志异常标记。关键在于词汇是否存在，而非其语义细微差别。
- 小数据场景（仅有数百个标注样本）。TF-IDF结合逻辑回归无需预训练成本。
- 在任何对延迟敏感的场景中，TF-IDF配合线性模型可在微秒级给出结果；而通过Transformer对文档进行嵌入则需要10至100毫秒。
- 需要能够解释预测结果的系统。可通过检查分类器的系数来确定原因，最具影响力的正面词汇即为答案所在。

### 当 TF-IDF 失效时

语义盲区故障。请看以下两段文本：

- “这部电影简直糟透了。”
- “这部电影非常出色。”

一段是负面评价，另一段是正面评价。它们的 TF-IDF 重叠词集恰好为 `{the, movie, was}`。基于词袋模型的分类器必须记住“not”与“good”相邻时会导致标签反转这一规则。虽然通过足够多的数据它也能学会这一点，但远不如理解句法的模型那样自然。

另一种故障：推理时的词汇表外词问题。在 IMDb 评论上训练的词袋模型如果从未见过“Zoomer-approved”这个词，就完全不知道该如何处理它。分词嵌入模型（第04课）可以解决这个问题，而 TF-IDF 则无法做到。

### 混合模型：TF-IDF 加权嵌入

2026年针对中等数据量分类的实用默认方案：将TF-IDF权重作为注意力机制用于词嵌入上。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

嵌入模型为模型提供了语义表征能力，而 TF-IDF 则用于强调罕见词汇。分类器则在这些融合后的向量上进行训练。在约 5 万个标注样本的规模下，该组合方法在情感分析、主题识别以及意图分类任务中的表现均优于单独使用任一方法。

## 发布它

另存为 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 练习题

1. **简单。** 在经过 L2 标准化的 TF-IDF 结果上实现 `cosine_similarity(doc_vec_a, doc_vec_b)` 函数。验证相同文档对之间的得分应为 1.0，而词汇集完全不重叠的文档对得分应为 0.0。
2. **中等难度。** 为 `bag_of_words` 方法添加 `n-gram` 支持。参数 `n` 用于指定生成 `n`-gram 的长度。测试当 `n=2` 且输入为 `["the", "cat", "sat"]` 时，是否能正确输出 `["the cat", "cat sat"]` 这两个二元词组的出现次数。
3. **高难度。** 使用 GloVe 100d 向量构建上述的 TF-IDF 加权嵌入混合模型（该向量只需下载一次并缓存）。在 20 Newsgroups 数据集上，分别对比该混合模型、纯 TF-IDF 模型以及纯均值池化嵌入模型的分类准确率，并说明在不同任务中哪种模型表现更优。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| BoW | 单词频率向量 | 记录某份文档中词汇的出现次数，忽略单词的顺序。 |
| TF | 词频 | 某个单词在文档中的出现次数，可选地会根据文档长度进行归一化处理。 |
| DF | 文档频率 | 至少包含该单词一次的文档数量。 |
| IDF | 反向文档频率 | 经过平滑处理的 `log(N / df)` 值，用于降低那些在所有文档中都出现的单词的权重。 |
| 稀疏向量 | 大部分为零 | 词汇表通常包含1万到10万个单词；在任意给定的文档中，大部分单词都不会出现。 |
| 余弦相似度 | 向量夹角 | 经过L2归一化后的两个向量的点积。值为1表示完全相同，值为0表示正交。 |

## 延伸阅读

- [scikit-learn — 从文本中提取特征](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 标准的 API 参考文档，以及针对每个参数的说明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 这篇论文使得 TF-IDF 成为长达十年的默认算法。
- ["为何 TF-IDF 依然优于嵌入模型" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 探讨在何种情况下旧方法仍具优势及其原因的 2026 年视角文章。
