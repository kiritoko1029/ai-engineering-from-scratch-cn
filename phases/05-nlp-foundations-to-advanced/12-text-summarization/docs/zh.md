# 文本摘要生成

> 提取式系统能告诉你文档中写了什么内容，而抽象式系统则能揭示作者的真实意图。不同的任务对应着不同的陷阱。

**类型：** 构建
**语言：** Python
**先修知识：** 第 5 阶段 · 02（词袋模型 + TF-IDF）、第 5 阶段 · 11（机器翻译）
**时长：** 约 75 分钟

## 问题所在

一篇2000字的新闻文章出现在您的信息流中。您需要用120字概括它。您可以选取文章中最重要的三句话（抽取式摘要），也可以用自己的话重写内容（生成式摘要）。这两种方法都被称为摘要生成，但本质上是完全不同的问题。

抽取式摘要属于排序问题。需要对每句话进行评分，然后选出排名前`k`的句子。由于其内容是直接摘录的，因此生成的文本始终符合语法规范。但其风险在于可能遗漏分散在文章各处的关键信息。

生成式摘要则属于生成问题。Transformer模型会根据输入信息生成新的文本。虽然生成的文本流畅且简洁，但也可能会编造出原文中不存在的事实。其风险在于可能出现毫无根据的虚构内容。

## 概念概述

![抽取式TextRank与抽象式Transformer对比](../assets/summarization.svg)

**抽取式方法。** 将文章视为一个图结构，其中节点代表句子，边代表句子之间的相似度。在该图上运行PageRank算法（或其类似算法），根据句子与其他所有句子的连接程度为它们打分。得分最高的句子即为摘要。其经典实现为**TextRank**（Mihalcea和Tarau，2004年）。

**抽象式方法。** 在文档与摘要的对子上对Transformer编码器-解码器模型（如BART、T5、Pegasus）进行微调。在推理阶段，模型会读取文档，并通过交叉注意力逐词生成摘要。尤其是Pegasus模型采用了间隙句预训练目标函数，这使得它在几乎无需大量微调的情况下就能表现出色地完成摘要生成任务。

评估通常使用**ROUGE**指标（面向召回率的摘要评估辅助工具）。ROUGE-1和ROUGE-2用于衡量单词级和双词级的重叠程度，而ROUGE-L则用于衡量最长公共子序列的长度。数值越高越好，40的ROUGE-L得分可视为“良好”，50则为“优秀”。每篇论文都会报告这三种指标的得分。可使用`rouge-score`包进行计算。

## 构建它

### 步骤 1：TextRank（抽取式）

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有两点值得特别说明。相似度函数采用对数归一化的词重叠度计算方式，这正是原始 TextRank 版本所使用的算法。TF-IDF 向量的余弦值同样可用。阻尼因子为 0.85，迭代次数则采用 PageRank 的默认设置。

### 步骤 2：使用 BART 进行抽象生成

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN是在CNN/DailyMail语料库上微调得到的模型。它可直接生成新闻风格的摘要。对于其他领域（科学论文、对话、法律文本），请使用相应的Pegasus检查点，或在对目标数据上进行微调后使用。

### 步骤 3：ROUGE 评估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

始终使用词干提取功能。若不使用，"running" 和 "run" 会被视为不同的单词，从而导致 ROUGE 分数被低估。

### 超越ROUGE：2026年摘要评估标准

二十年来，ROUGE一直是最主流的摘要评估指标，但在2026年已显不足。一项针对自然语言生成相关论文的大规模元分析显示：

- **BERTScore**（基于上下文嵌入的相似度度量）在2023年前后逐渐占据优势，如今在大多数摘要评估论文中都会与ROUGE一同被提及。
- **BARTScore**将评估视为生成过程：通过预训练的BART模型根据源文本判断摘要出现的概率来给出分数。
- **MoverScore**（基于上下文嵌入的地球移动距离度量）因能比ROUGE更准确地捕捉语义重叠，在2025年的摘要评估基准测试中位居榜首。
- **FactCC**以及基于问答能力的忠实度评估在2021至2023年间较为常见，如今常被**G-Eval**所取代——后者是一种基于GPT-4的提示链方法，可通过思维链推理来评估摘要的连贯性、一致性、流畅度及相关性。
- 当评分标准设计合理时，**G-Eval**及类似的LLM评判方法在约80%的情况下能与人类判断结果保持一致。

生产环境建议：为便于与传统指标对比，使用ROUGE-L；用于衡量语义重叠程度则选用BERTScore；而评估摘要的连贯性与事实性则采用G-Eval。评分时需以50至100份由人工标注的摘要作为参照基准。

### 第 4 步：事实性问题

抽象式摘要容易产生幻觉。抽取式摘要的幻觉风险要低得多，因为其输出是直接从源文本中原样提取的；不过，如果源文本句子缺乏上下文、已经过时或引用顺序错误，仍可能造成误导。这也是生产系统在处理需符合规范的内容时依然更倾向于使用抽取式方法的最主要原因。

常见的幻觉类型包括：

- **实体替换**：源文本为“John Smith”，摘要却写成“John Brown”。
- **数值偏差**：源文本为“25,000”，摘要却变为“25百万”。
- **极性反转**：源文本为“拒绝了该提议”，摘要却写成“接受了该提议”。
- **事实编造**：源文本中未提及CEO，摘要却称CEO已批准。

有效的评估方法包括：

- **FactCC**：一种基于源句与摘要句之间蕴含关系的二分类模型，用于判断内容是否真实。
- **基于问答的事实性检测**：向问答模型提出答案均存在于源文本中的问题，若摘要给出的答案与之不同，则标记为异常。
- **实体级F1分数**：对比源文本与摘要中的命名实体，仅出现在摘要中的实体值得怀疑。

对于任何对事实准确性要求较高的面向用户的内容（如新闻、医疗、法律、金融领域），抽取式方法都是更安全的默认选择。而抽象式方法则需要在处理过程中进行持续的事实性校验。

## 使用它

2026年的技术栈：

| 应用场景 | 推荐模型 |
|---------|-----------|
| 新闻内容，3-5句摘要，英文文本 | `facebook/bart-large-cnn` |
| 科学论文处理 | `google/pegasus-pubmed` 或经过调优的 T5 模型 |
| 多文档、长篇内容处理 | 任何支持32k以上上下文且可通过提示词引导的 LLM |
| 对话摘要生成 | `philschmid/bart-large-cnn-samsum` |
| 提取式处理，天然具有较低的幻觉风险 | TextRank 或 `sumy` 的 LSA / LexRank 方法 |

在计算资源不受限制的情况下，2026年拥有长上下文能力的 LLM 通常能优于专用模型。两者的权衡在于成本与结果一致性；专用模型能够提供更为稳定的输出。

## 发布它

保存为 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 练习题

1. **简单。** 对5篇新闻文章运行TextRank算法。将生成的前三句与参考摘要进行对比，并计算ROUGE-L分数。对于CNN或DailyMail风格的文章，该分数通常应在30到45之间。
2. **中等难度。** 实现实体级事实性检测：使用spaCy从原文和摘要中提取命名实体，进而计算源实体在摘要中的召回率以及摘要实体与源实体的精确度。高精确度但低召回率意味着摘要内容准确但过于简略；而低精确度则表明存在虚构的实体。
3. **高难度。** 使用BART-large-CNN模型与LLM模型（如Claude或GPT-4）分别处理50篇CNN或DailyMail风格的文章。需报告ROUGE-L分数、实体级事实性指标（即实体F1值），以及生成每条摘要的成本。同时记录两种模型在各项指标上的优劣势。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 提取式 | 拿出句子 | 原样返回源文本中的句子，绝不会编造内容。 |
| 抽象式 | 重新撰写 | 根据源文本生成新文本，可能会出现虚构内容。 |
| ROUGE | 摘要度量指标 | 系统输出与参考文本之间的N-gram/最长公共子序列重叠程度。 |
| TextRank | 基于图的提取方法 | 在句子相似性图上应用PageRank算法。 |
| 事实性 | 是否正确 | 摘要中的陈述是否得到源文本的支持。 |
| 虚构内容 | 编造的内容 | 摘要中源文本未提及的虚构信息。 |

## 延伸阅读

- [Mihalcea 和 Tarau（2004）。TextRank：为文本带来秩序](https://aclanthology.org/W04-3252/) —— 该领域的奠基性论文。
- [Lewis 等人（2019）。BART：序列到序列预训练的去噪方法](https://arxiv.org/abs/1910.13461) —— BART 相关论文。
- [Zhang 等人（2019）。PEGASUS：基于提取的间隙句进行预训练](https://arxiv.org/abs/1912.08777) —— Pegasus 及其间隙句目标函数相关内容。
- [Lin（2004）。ROUGE：用于摘要自动评估的工具包](https://aclanthology.org/W04-1013/) —— ROUGE 相关论文。
- [Maynez 等人（2020）。关于抽象总结中的忠实性与事实性](https://arxiv.org/abs/2005.00661) —— 关于事实性评估的代表性论文。
