# 问答系统

> 三种系统共同塑造了现代的质量检测技术。抽取式模型用于定位文本片段，检索增强型模型将这些片段与文档关联起来，而生成式模型则负责输出答案。所有现代的 AI 助手都是这三种技术的结合体。

**类型：** 构建
**语言：** Python
**先修知识：** 第 5 阶段 · 11（机器翻译）、第 5 阶段 · 10（注意力机制）
**耗时：** 约 75 分钟

## 问题所在

用户输入“第一代 iPhone 是何时发布的？”时，期望得到的答案是“2007年6月29日”。而不是“苹果公司的历史悠久且丰富多彩”这类内容，也不是孤立出现的“2007年”而没有上下文。需要的是一个直接、准确且基于事实的回答。

在过去的十年中，有三种架构主导着问答系统的发展。

- **抽取式问答**：给定一个问题以及已知包含答案的文本，找出文本中答案所在区间的起始和结束索引。SQuAD 是该领域的标准基准测试。
- **开放领域问答**：不直接提供文本，需要先检索出相关内容，然后再从中提取或生成答案。这是当今所有 RAG 流水线的核心基础。
- **生成式/闭卷式问答**：大型语言模型直接从其参数化记忆中回答问题，无需进行信息检索。推理速度最快，但在事实准确性方面最不可靠。

2026年的发展趋势是混合模式：先检索出最佳的几段文本，再提示生成型模型基于这些文本来回答问题。这就是 RAG 技术，第14课将深入讲解其中的检索部分，而本课则侧重于问答部分的实现。

## 概念概述

![QA架构：抽取式、检索增强式、生成式](../assets/qa.svg)

**抽取式。** 使用Transformer模型（BERT系列）对问题与文本进行编码。训练两个输出头，分别预测答案的起始和结束token索引。损失函数为有效位置上的交叉熵损失。最终输出为文本中的一个片段。由于架构设计，该方式不会产生幻觉，也无法处理文本无法回答的问题。

**检索增强式（RAG）。** 包含两个阶段。首先，检索器从语料库中找出排名前`k`的文本片段。其次，阅读器（抽取式或生成式）利用这些片段生成答案。由于检索器与阅读器分离，二者可以独立训练和评估。现代的RAG系统通常会在两者之间增加一个重排器。

**生成式。** 仅使用解码器的LLM模型（如GPT、Claude、Llama）根据学到的权重直接生成答案。无需检索步骤。在处理常识类问题时表现优异，但对于罕见或最新的事实则效果较差。其幻觉率与预训练数据中事实的出现频率呈反比关系。

## 构建它

### 步骤 1：使用预训练模型进行提取式质量检测

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 模型是在包含无法回答的问题的 SQuAD 2.0 数据集上训练的。默认情况下，即使模型的空值得分最高，`question-answering` 流水线也会返回得分最高的文本片段——它不会自动返回空答案。若需实现明确的“无答案”处理逻辑，可在调用流水线时传入 `handle_impossible_answer=True` 参数：此时只有当空值得分高于所有文本片段的得分时，流水线才会返回空答案。无论何种情况，都应检查 `score` 字段。

### 步骤 2：检索增强型流水线（概要）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

两阶段处理流程。密集检索器（Sentence-BERT）通过语义相似度查找相关段落，而抽取式阅读器（RoBERTa-SQuAD）则从这些优选段落中提取答案片段。该方案适用于小型语料库；对于包含百万份文档的语料库，则应使用 FAISS 或向量数据库。

### 步骤 3：基于 RAG 的生成式模型

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

提示词的模式至关重要。明确要求模型以上下文为依据，并在上下文不足时返回“我不知道”，相比简单的提示方式，这能将幻觉率降低40%-60%。更为复杂的模式还能加入引用、置信度分数以及结构化的信息提取功能。

### 第 4 步：反映现实世界的评估

SQuAD采用**精确匹配（EM）**和**分词级F1分数**两种评估方式。精确匹配是在对文本进行规范化处理（转为小写、去除标点符号及冠词）后的严格比对——预测结果必须与参考答案完全一致，否则得分为0；而分词级F1分数则是通过计算预测结果与参考答案之间的分词重叠度来给出部分分数。对于语义相近但存在细微差异的表述，例如“June 29, 2007”与“June 29th, 2007”，通常会在精确匹配中得分为0（因为序数词破坏了规范化后的文本结构），但由于分词存在重叠，仍能在分词级F1分数中获得较高得分。

在实际生产环境中的问答系统中，还需关注以下指标：

- **答案准确性**：可由大语言模型或人工进行评估，因为现有指标无法完全体现语义等价性。
- **引用准确性**：所引用的段落是否确实能够支持该答案？只需通过比对生成的引用文本与检索到的段落即可自动完成检测。
- **拒绝响应的准确性**：当答案不在检索到的段落中时，系统是否能正确给出“我不知道”的回应？可通过测量错误自信率来评估。
- **检索召回率**：在评估阅读器之前，需先确认检索组件能否将正确的段落纳入前`k`个结果中。一旦缺失关键段落，阅读器便无法完成正确解答。

### RAGAS：2026年生产环境评估框架

`RAGAS` 是专为 RAG 系统设计的工具，也是 2026 年的默认上线版本。它无需依赖标准参考答案即可对四个维度进行评分：

- **真实性**：答案中的每一项陈述是否均来源于检索到的上下文？通过基于 NLI 的蕴含关系来衡量，是检测幻觉的主要指标。
- **答案相关性**：答案是否针对问题本身？通过从答案中生成假设性问题并与真实问题进行比对来进行评估。
- **上下文精确度**：在所有检索到的片段中，实际相关的比例是多少？精确度低意味着提示词中存在噪声。
- **上下文召回率**：检索到的内容是否包含了所有必要的信息？召回率低会导致用户无法获得正确结果。

由于无需参考答案，因此可以在真实的生产环境中进行评估，而无需准备精心整理的标准答案。对于那些无法使用精确匹配指标解答的开放式问题，可再叠加“LLM-as-judge”层来进行进一步判断。

执行命令 `pip install ragas` 即可。只需接入您的检索器和解码器，即可为每个查询获取四个数值指标，并能及时检测到性能下降的情况。

## 使用它

2026年的技术栈。

| 使用场景 | 推荐方案 |
|---------|-----------|
| 给定段落，查找答案片段 | `deepset/roberta-base-squad2` |
| 基于固定语料库，不允许闭卷查询 | RAG：密集检索器 + LLM阅读器 |
| 基于文档存储的实时查询 | 混合型RAG（BM25 + 密集检索）+ 重排器（第14课内容） |
| 对话式问答（后续问题） | 具备对话历史记录的LLM + 每次轮次对应的RAG |
| 高度依赖事实、受监管的行业领域 | 基于权威语料库的抽取式查询；绝不能仅使用生成式方法 |

由于结合LLM的RAG能处理更多场景，2026年抽取式问答已不再流行。但在需要精确引用原文的场景中，它依然会被采用：法律研究、合规审查、审计工具等。

## 发布它

保存为 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 练习题

1. **简单级。** 在10篇维基百科文章上应用上述的SQuAD抽取式处理流程，手动编写10个问题，并统计答案正确的频率。如果文章和问题的质量良好，正确率应在7-9之间。
2. **中等级。** 添加拒绝分类器。当最高检索得分低于某个阈值（例如余弦相似度为0.3）时，返回“我不知道”而非调用阅读器功能。需在保留的测试集上调整该阈值。
3. **高级别。** 从您选择的10,000篇文档语料库中构建RAG处理流程。实现基于RRF融合的混合检索方式（结合BM25与密集向量检索，参见第14课）。分别测量有无混合检索步骤时的答案准确率，并记录哪种类型的问题受益最大。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 提取式问答 | 定位答案片段 | 预测给定段落中答案的起始与结束索引。 |
| 开放领域问答 | 基于语料库的问答 | 没有给定的段落；需要先检索再作答。 |
| RAG | 先检索后生成 | 基于检索的生成技术。包含检索器与阅读器两个环节。 |
| SQuAD | 标准基准数据集 | 斯坦福问题回答数据集。采用EM与F1指标进行评估。 |
| 虚构答案 | 编造的回答 | 阅读器的输出内容无法在检索到的上下文中得到支持。 |
| 拒绝校准 | 知道何时保持沉默 | 系统在无法作答时能正确输出“我不知道”。 |

## 延伸阅读

- [Rajpurkar 等人（2016）。SQuAD：用于文本机器理解的10万+个问题](https://arxiv.org/abs/1606.05250) —— 该领域的基准论文。
- [Karpukhin 等人（2020）。面向开放领域问答的密集段落检索](https://arxiv.org/abs/2004.04906) —— DPR，即用于问答任务的经典密集检索器。
- [Lewis 等人（2020）。用于知识密集型自然语言处理任务的检索增强生成](https://arxiv.org/abs/2005.11401) —— 首次提出 RAG 概念的论文。
- [Gao 等人（2023）。面向大型语言模型的检索增强生成：综述](https://arxiv.org/abs/2312.10997) —— 关于 RAG 的全面综述文章。
