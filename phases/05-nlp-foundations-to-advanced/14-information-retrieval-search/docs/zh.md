# 信息检索与搜索

> BM25精度高但易出错。Dense算法覆盖范围广，但容易遗漏关键词。Hybrid算法将是2026年的默认选择。其余方法均需通过调优来优化性能。

**类型：** 构建
**语言：** Python
**先修课程：** 第5阶段·02（BoW + TF-IDF），第5阶段·04（GloVe, FastText, Subword）
**耗时：** 约75分钟

## 问题所在

用户输入“如果有人谎称骗取钱财会怎样”，期望找到实际适用于该情况的法规：“《印度刑法典》第420条”。关键词搜索完全无法定位到它（因为两者缺乏共享的词汇表）；若嵌入模型未在法律文本上训练过，语义搜索也会失效。真正的搜索系统必须同时处理这两种情况。

信息检索是每个RAG系统、每个搜索栏以及每个文档网站模糊查询功能背后的核心流程。2026年可用于生产环境的架构并非单一方法，而是一系列互补方法的组合，每种方法都能弥补前一种方法的缺陷。

本课程将逐步讲解这些组成部分，并说明每一种方法能够解决哪些问题。

## 概念概述

![混合检索：BM25 + 密集向量检索 + RRF + 交叉编码器重排](../assets/retrieval.svg)

共四层，可根据需求选择使用。

1. **稀疏检索（BM25）。** 执行速度快，对精确匹配结果准确度较高，但在处理语义层面表现较差。通过倒排索引进行查询，在数百万份文档的情况下每次查询耗时不到10毫秒，能够准确获取法规引用、产品代码、错误信息及命名实体。
2. **密集向量检索。** 将查询和文档编码为向量，然后进行最近邻搜索，以此捕捉同义表达与语义相似性，但无法识别仅相差一个字符的精确关键词匹配。使用 FAISS 或向量数据库时，每次查询耗时约为50至200毫秒。
3. **融合层。** 将稀疏检索和密集向量检索得到的排序列表合并在一起。互反排名融合（RRF）是较为简单的默认选择，因为它忽略原始分数（这些分数处于不同的量级），仅依据排名位置进行融合；当明确知道某一类特征在特定领域中起主导作用时，也可采用加权融合的方式。
4. **交叉编码器重排。** 从融合层得到的前30个结果中选取，再使用交叉编码器对每对查询-文档组合进行评分，最终保留排名最高的5个结果。虽然交叉编码器处理每对数据的速度慢于双编码器，但其准确度要高得多。由于只需在top-30的结果上运行交叉编码器，因此可以平衡性能与成本。

在2026年的各项基准测试中，采用三路检索方式（BM25 + 密集向量检索 + 如 SPLADE 这类的学习型稀疏索引）的性能优于两路检索，但需要相应的基础设施来构建学习型稀疏索引。对于大多数团队而言，两路检索结合交叉编码器重排是最佳选择。

## 构建它

### 步骤 1：从零实现 BM25 算法

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

有两个值得了解的参数。`k1=1.5`用于控制词频饱和度；数值越大，对词汇重复出现的权重赋予越高。`b=0.75`用于控制长度归一化；值为 0 表示忽略文档长度，值为 1 表示完全进行归一化。这些默认值源自原始论文中 Robertson 的建议，通常无需调整。

### 步骤 2：使用双编码器进行密集检索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

对嵌入向量进行 L2 标准化，使得点积值等于余弦值。`all-MiniLM-L6-v2` 的维度为 384，计算速度快，且性能足以满足大多数英语检索需求。如需处理多语言场景，请使用 `paraphrase-multilingual-MiniLM-L12-v2`。若追求最高精度，则可选择 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步骤 3：互逆排名融合

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 这一常量源自最初的 RRF 论文。较大的 `k` 值能够减弱排名差异带来的影响；较小的 `k` 值则会使顶级排名的权重占据主导地位。60 是该论文中规定的默认值，通常无需进行调整。

### 步骤 4：混合搜索 + 重排序

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

该算法由三个阶段组成。BM25用于查找词汇层面的匹配项，Dense则用于查找语义层面的匹配项。RRF可在无需进行分数校准的情况下将这两种排序结果合并。Cross-Encoder则会利用查询词与文档对共同对前30个结果重新评分，从而捕捉到双编码器所遗漏的细粒度相关性。最终保留排名前5的结果。

### 第 5 步：评估

| 指标 | 含义 |
|------|------|
| Recall@k | 在存在正确文档的查询中，该文档出现在前 k 个结果中的频率。 |
| MRR（平均倒数排名） | 第一个相关文档的排名值的 1/排名值之和的平均值。 |
| nDCG@k | 考虑了相关性程度的差异，而不仅仅是二元的相关/不相关判断。 |

针对 RAG 系统而言，检索器的 **Recall@k** 是最重要的指标。如果正确的段落未出现在检索结果中，用户便无法获得答案。

调试技巧：对于失败的查询，对比稀疏排名与密集排名。若一种排名方式能找到正确文档而另一种找不到，则可能是词汇表不匹配（解决方法：补充缺失的词汇）或存在语义歧义（解决方法：优化嵌入模型或使用重排器）。

## 使用它

2026年技术栈：

| 规模 | 技术栈 |
|-------|-------|
| 1k-100k篇文档 | 内存型BM25 + `all-MiniLM-L6-v2`嵌入模型 + RRF排序算法。无需单独的数据库。 |
| 100k-10M篇文档 | 密集向量存储使用FAISS或pgvector，BM25查询则使用Elasticsearch/OpenSearch。各组件并行运行。 |
| 10M+篇文档 | 支持混合检索方式的Qdrant/Weaviate/Vespa/Milvus，对前30条结果进行跨编码器重排序。 |
| 最高精度方案 | BM25、密集向量检索与SPLADE三种方式结合，再通过ColBERT实现后期交互式重排序。 |

无论选择哪种方案，都需预留评估预算。在测试端到端的RAG系统准确率之前，应先评估检索召回率。若检索器未能找到相关内容，后续的阅读理解模块也无法弥补这一缺陷。

### 2026年生产环境RAG项目积累的宝贵经验教训

- **80%的RAG系统故障源于数据摄取与分块环节，而非模型本身。**团队往往花费数周时间更换大语言模型并调整提示词，而检索机制却会在每三次查询中就悄悄返回错误的上下文。应优先解决分块问题。
- **分块策略比分块大小更为重要。**固定大小的拆分方式会破坏表格、代码及嵌套标题的结构。默认采用基于句子的分块方式；对于技术文档和产品手册，基于语义或大语言模型的分块方式能带来更好的效果。
- **父文档模式。**通过检索小型“子”分块来提升精确度。当同一父章节中出现多个子内容时，替换为父文档块以保留上下文。这种方式无需重新训练即可持续提高答案质量。
- **k_rerank=3通常为最佳值。**超过该数值的额外分块只会增加令牌成本和生成延迟，而无法提升答案质量。如果对您而言k=8仍优于k=3，说明重排器性能不佳。
- **HyDE / 查询扩展技术。**根据查询内容生成假设性答案并对其进行嵌入，随后进行检索。该技术可弥补简短问题与长文档之间的表述差异，且无需训练即可提升精确度。
- **上下文容量应控制在8K令牌以内。**若在该限制下仍能持续获得准确结果，说明重排器的阈值设置过宽。
- **所有内容均需版本控制。**包括提示词、分块规则、嵌入模型以及重排器。任何细微的变化都可能悄无声息地降低答案质量。通过针对准确性、上下文精确度及未回答问题率的CI检测机制，可在用户察觉之前阻止性能下降。
- **在2026年的基准测试中，三路检索方式（BM25 + 密集向量检索 + 如SPLADE之类的学习型稀疏向量检索）的表现优于两路检索**，尤其是在涉及专有名词与语义混合的查询场景下。当基础设施支持SPLADE索引时即可采用该方案。
根据2026年的行业数据，合理的检索设计可将幻觉现象减少70-90%。RAG系统性能的提升主要来自更优的检索机制，而非模型微调。

## 发布它

保存为 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 练习题

1. **简单。** 在包含500篇文档的语料库上实现上述的`hybrid_search`函数，并测试20个查询。分别计算仅使用BM25、仅使用密集向量以及混合方法在召回率达到5%时的表现，进行对比。
2. **中等难度。** 增加MRR（平均每次点击收益）的计算功能。对于每个已知正确文档的测试查询，分别找出该正确文档在BM25、密集向量和混合方法排序结果中的排名，并报告各自的MRR值。
3. **高难度。** 使用Sentence Transformers库中的MultipleNegativesRankingLoss函数，在您的特定领域数据上对密集向量编码器进行微调。从500对查询-文档对中构建训练集，比较微调前后的召回率表现。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| BM25 | 关键词搜索 | Okapi BM25。通过词频、IDF和文档长度对文档进行评分。 |
| 密集检索 | 向量搜索 | 将查询和文档编码为向量，进而查找最近的邻居。 |
| 双向编码器 | 嵌入模型 | 分别独立地对查询和文档进行编码。查询时速度较快。 |
| 交叉编码器 | 重排模型 | 将查询和文档一起编码。速度较慢但精度较高。 |
| RRF | 排名融合 | 通过计算 `1/(k + rank)` 的值并将两个排名相加来合并结果。 |
| Recall@k | 检索指标 | 在前 k 个结果中包含相关文档的查询比例。 |

## 延伸阅读

- [Robertson 和 Zaragoza (2009). 《概率相关性框架：BM25 及其扩展》](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) —— 关于 BM25 最权威的论述。
- [Karpukhin 等人 (2020). 《面向开放领域问答的密集段落检索》](https://arxiv.org/abs/2004.04906) —— DPR，即标准的双编码器模型。
- [Formal 等人 (2021). 《SPLADE：稀疏词汇与扩展模型》](https://arxiv.org/abs/2107.05720) —— 这种基于学习的稀疏检索器缩小了与密集检索器的差距。
- [Cormack、Clarke、Büttcher (2009). 《互反排名融合优于康德赛特方法及单独的排名学习方法》](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) —— RRF 方法的相关论文。
- [Khattab 和 Zaharia (2020). 《ColBERT：高效且有效的段落检索技术》](https://arxiv.org/abs/2004.12832) —— 基于后期交互的检索方法。
