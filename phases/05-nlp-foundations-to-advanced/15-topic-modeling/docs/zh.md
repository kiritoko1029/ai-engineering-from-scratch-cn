# 主题建模——LDA与BERTopic

> LDA：文档是主题的混合体，而主题则是词汇的分布。BERTopic：文档在嵌入空间中聚类，这些簇即为主题。目标相同，分解方式不同。

**类型：** 学习
**语言：** Python
**先修知识：** 第5阶段 · 02（词袋模型 + TF-IDF），第5阶段 · 03（Word2Vec）
**时长：** 约45分钟

## 问题所在

你有 10,000 个客户支持工单、50,000 篇新闻文章，或是 200,000 条推文。你需要在不阅读内容的情况下了解这些集合的主题是什么。你没有已标记的类别，甚至不知道存在多少个类别。

主题建模可以在无监督情况下解决这一问题。只需提供语料库，它就会返回一组连贯的主题，并为每篇文档生成这些主题的分布概率。

目前主要有两大类算法。LDA（2003 年提出）将每篇文档视为潜在主题的混合体，同时将每个主题视为词汇的分布。其推理过程基于贝叶斯理论。在需要混合成员身份的主题分配以及可解释的词级概率分布的场景中，它仍被广泛应用于实际生产环境。

BERTopic（2020 年提出）则利用 BERT 对文档进行编码，通过 UMAP 降低维度，再用 HDBSCAN 进行聚类，并借助基于类别的 TF-IDF 方法提取主题词汇。它在处理短文本、社交媒体内容以及语义相似性比词重叠更重要的场景时表现更为出色。不过，它要求每篇文档仅对应一个主题，这对于长篇内容来说存在一定的局限性。

本课程将帮助你理解这两种算法的原理，并指导你在面对特定语料库时应选择哪种算法。

## 概念概述

![LDA混合模型与BERTopic聚类对比](../assets/topic-modeling.svg)

**LDA生成模型原理。** 每个主题都是词汇的分布集合，每篇文档则是这些主题的混合体。要生成文档中的某个词，需先从该文档的主题混合中抽取一个主题，再从该主题的词汇分布中抽取一个词。推理过程则相反：给定观测到的词，推断出每篇文档的主题分布以及每个主题的词汇分布。通常使用折叠吉布斯采样或变分贝叶斯方法来完成这些数学计算。

LDA的主要输出包括：

- `doc_topic`：形状为`(n_docs, n_topics)`的矩阵，每行的元素之和为1（表示文档的主题混合比例）。
- `topic_word`：形状为`(n_topics, vocab_size)`的矩阵，每行的元素之和为1（表示主题的词汇分布）。

**BERTopic处理流程。**

1. 使用句子编码器（例如`all-MiniLM-L6-v2`）对每篇文档进行编码，生成384维向量。
2. 通过UMAP算法将维度降维至约5维。BERT嵌入的维度过高，不适合直接用于聚类。
3. 利用HDBSCAN算法进行聚类。该算法基于密度划分，能够生成大小不固定的簇，并为异常数据标记“outlier”标签。
4. 对每个簇，计算其内部文档的类别型TF-IDF值，从而提取出最重要的词汇。

最终输出为每篇文档对应的一个主题（若为异常数据则标记为-1）。可选地，还可以通过HDBSCAN的概率向量表示文档属于各簇的软隶属度。

## 构建它

### 步骤 1：使用 scikit-learn 进行 LDA 分析

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：已移除停用词，同时应用了 min_df 和 max_df 过滤器以排除过于罕见或普遍的词汇；此处使用的是 CountVectorizer（而非 TfidfVectorizer），因为 LDA 算法需要原始计数值。

### 步骤 2：BERTopic（生产环境）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

`Topic != -1` 这一过滤条件用于剔除 BERTopic 中的异常数据桶（即 HDBSCAN 无法对其聚类的文档）。`min_topic_size` 用于控制 HDBSCAN 的最小聚类规模；BERTopic 库的默认值为 10。本示例为适应课程教学需求，将其显式设置为 15。对于包含超过 10,000 篇文档的语料库，该值应提升至 50 或 100。

### 步骤 3：评估

这两种方法都会输出主题词，问题在于这些词之间是否具有连贯性。

- **主题连贯性（c_v）**：计算滑动窗口上下文中各主题词对之间的归一化点互信息 NPMI，将得到的分数聚合为主题向量，再通过余弦相似度来比较这些向量。数值越高表示连贯性越好。可使用 `gensim.models.CoherenceModel` 并设置参数 `coherence="c_v"`。
- **主题多样性**：所有主题的顶级词中唯一词所占的比例。该值越高表示主题之间的重叠越少，多样性越好。
- **定性检查**：查看每个主题的顶级词。这些词是否指代了真实的事物？人工判断仍是最终的验证手段。

## 何时选择哪种

| 情况 | 推荐模型 |
|-----------|----------|
| 短文本（推文、评论、标题） | BERTopic |
| 含有多种主题的长文档 | LDA |
| 无 GPU 或计算资源有限 | LDA 或 NMF |
| 需要文档级的多主题分布 | LDA |
| 需结合大语言模型进行主题标注 | BERTopic（直接支持） |
| 资源受限的边缘设备部署 | LDA |
| 最高语义连贯性要求 | BERTopic |

在实际应用中，最需考虑的因素是文档长度。BERT 嵌入模型会对文本进行截断处理，而 LDA 的计数方法则不受长度限制。对于超出嵌入模型上下文范围的文档，可采用分块聚合或直接使用 LDA 方法。

## 使用它

2026年主流技术栈：

- **BERTopic。** 适用于短文本及语义至关重要的场景的默认选择。
- **`gensim.models.LdaModel`。** 用于生产环境的经典LDA算法，成熟且经过大量实战检验。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 便于实验使用的简易LDA实现。
- **NMF。** 非负矩阵分解技术。作为LDA的快速替代方案，在处理短文本时能达到相近的质量。
- **Top2Vec。** 设计理念与BERTopic类似，社区规模较小，但在某些基准测试中表现优异。
- **FASTopic。** 较新推出的算法，在处理极大规模语料库时比BERTopic更快。
- **基于LLM的标注方法。** 先执行任意聚类操作，再通过提示词让模型为每个簇命名。

## 发布它

保存为 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 练习题

1. **简单。** 在 20 Newsgroups 数据集上拟合包含 5 个主题的 LDA 模型。输出每个主题的前 10 个高频词，并手动为每个主题标注标签。该算法能否准确识别出真实的类别？
2. **中等。** 在同一份 20 Newsgroups 子集数据上拟合 BERTopic 模型。将所发现的主题数量、高频词以及定性连贯性分别与 LDA 的结果进行对比。哪种方法能更清晰地呈现真实的类别？
3. **困难。** 对你的语料库分别计算 LDA 和 BERTopic 的 c_v 连贯性值。分别使用 5、10、20、50 个主题运行这两种模型，并绘制连贯性随主题数量变化的图表。分析哪种方法在不同的主题数量下表现更为稳定。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 主题 | 语料库所涉及的内容 | LDA模型中的词概率分布，或BERTopic模型中相似文档的聚类结果。 |
| 混合成员身份 | 文档属于多个主题 | LDA为每篇文档分配所有主题的概率分布。 |
| UMAP | 维度降维技术 | 一种保留局部结构的流形学习方法；被用于BERTopic中。 |
| HDBSCAN | 密度聚类算法 | 能够发现不同大小的聚类；会将异常值标记为“噪声”标签（-1）。 |
| c_v一致性系数 | 主题质量评估指标 | 在滑动窗口内，顶级主题词之间的平均点对互信息值。 |

## 延伸阅读

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 相关论文。  
- [Grootendorst (2022). BERTopic: 基于类化 TF-IDF 方法的神经网络主题建模](https://arxiv.org/abs/2203.05794) — BERTopic 相关论文。  
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 提出 c_v 及相关指标的论文。  
- [BERTopic 文档](https://maartengr.github.io/BERTopic/) — 实际应用参考资料，包含大量优秀示例。
