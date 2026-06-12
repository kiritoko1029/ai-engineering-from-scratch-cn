# 共指消解

> “她给他打了电话。他没有接听。医生正在吃午饭。”此处出现了三次对两个人的提及，但均未给出具体姓名。需要通过共指消解来确定各自的身份。

**类型：** 学习
**语言：** Python
**先修课程：** 第5阶段 · 06（命名实体识别），第5阶段 · 07（词性标注与句法分析）
**时长：** 约60分钟

## 问题所在

从一篇300字的文章中提取所有提及Apple Inc.的内容。如果文章直接写“Apple”，则较为简单；但当使用“the company”、“they”、“Cupertino's technology giant”或“Jobs's firm”等表述时，处理起来就很困难。若不将这些不同表述映射为同一实体，自然语言识别流程将会遗漏60%到80%的提及内容。

共指消解功能会将所有指向同一现实实体的表达关联为一个整体簇。它是表层自然语言处理技术（如命名实体识别、句法分析）与下游语义处理技术（如信息抽取、问答系统、摘要生成、知识图谱构建）之间的桥梁。

在2026年，这一技术的重要性体现在以下几个方面：

- 摘要生成：是“CEO宣布了……”还是“Tim Cook宣布了……”——合理的摘要应当明确指出首席执行官的身份。
- 问答系统：“她给谁打了电话？”这类问题需要先解决“she”所指代的实体。
- 信息抽取：如果知识图谱中将“PER1创立了Apple”与“Jobs创立了Apple”视为两条独立的记录，那是错误的。
- 多文档信息抽取：针对同一事件的各篇文章中的相关提及进行合并，正是跨文档共指消解的应用。

## 概念概述

![共指聚类：提及项 → 实体](../assets/coref.svg)

**任务描述。** 输入：一份文档。输出：对所有提及项（文本片段）的聚类结果，每个簇对应一个实体。

**提及项类型。**

- **专有名词实体。** “Tim Cook”
- **名词性提及项。** “the CEO”、“the company”
- **代词性提及项。** “he”、“she”、“they”、“it”
- **同位语提及项。** “Tim Cook, Apple's CEO,”

**常用架构。**

1. **基于规则的算法（Hobbs, 1978）。** 利用语法规则通过句法树进行代词解析。可作为良好的基准方法，但在处理代词时往往难以被超越。
2. **提及项对分类器。** 对每一对提及项（m_i, m_j）判断它们是否属于同一实体，再通过传递闭包进行聚类。这是2016年之前的标准方法。
3. **提及项排序法。** 为每个提及项列出所有可能的先行词候选项（包括“无先行词”），并选择得分最高的那个作为其先行词。
4. **基于文本片段的端到端算法（Lee et al., 2017）。** 使用Transformer编码器，枚举长度受限的所有可能文本片段，预测每个提及项的得分以及该片段作为先行词的概率，随后进行贪婪聚类。这是目前的默认主流方法。
5. **生成式模型（2024年及以后）。** 向大语言模型发送提示：“列出此文本中的所有代词及其对应的先行词。”在简单场景下表现良好，但在处理长文档或罕见指代对象时效果较差。

**评估指标。** 由于没有单一指标能够全面反映聚类质量，因此采用五种标准指标（MUC、B³、CEAF、BLANC、LEA）。前三个指标的平均值作为CoNLL F1分数报告。2026年针对CoNLL-2012数据集的顶尖模型F1分数约为83。

**常见的难题场景。**

- 指代在前面页面已介绍过的实体的定指描述。
- 跨段指代（如“the wheels”指代之前提到的一辆汽车）。
- 中文和日文等语言中不存在回指现象。
- 前指现象（代词出现在其指代对象之前），例如：“When **she** walked in, Mary smiled.”

## 构建它

### 步骤 1：预训练的神经网络指代消解模型（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said they would ship soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在较长的文档中，内容如下：
- 聚类 1：[Apple, The company, they]
- 聚类 2：[新产品]

### 步骤 2：基于规则的代词解析器（教学内容）

有关仅使用标准库的实现方式，请参见 `code/main.py`：

1. 提取提及内容：命名实体（大写文本片段）、代词（通过字典查询）以及定指描述（“the X”）。
2. 对于每个代词，查看其之前的 K 个提及项，并根据以下标准为它们评分：
   - 性别/数的一致性（启发式规则）
   - 新近程度（越近的得分越高）
   - 句法角色（优先选择主语）
3. 将得分最高的先行词与代词关联起来。

该实现虽无法与神经网络模型相媲美，但能够展示搜索空间以及端到端模型必须做出的决策。

### 步骤 3：使用大语言模型处理共指关系

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

需要重点关注的两种故障模式。首先，大语言模型会出现过度合并的情况（将“他”和“她”误认为是两个不同的人）。其次，大语言模型会在长文档中悄悄忽略某些提及内容。务必通过跨度偏移量检查来加以验证。

### 第 4 步：评估

标准的conll-2012评分脚本用于计算MUC、B³、CEAF-φ4等指标，并输出其平均值。在进行内部评估时，首先基于标注后的测试集计算跨度级别的精确度与召回率，随后再加入提及链接F1分数。

## 常见陷阱

- **单例爆炸问题。**某些系统会将每个提及都视为独立的聚类。B³对此较为宽容，而MUC则会严厉惩罚此类行为。务必同时检查这三项指标。
- **长上下文中的代词处理。**在长度超过2,000个标记的文档中，性能会下降约15个F1分数。需谨慎进行分块处理。
- **性别假设问题。**硬编码的性别规则在涉及非二元性别的人、组织或动物时会出现错误。应使用学习型模型或中性评分方式。
- **长文档下的大语言模型漂移问题。**单次API调用无法可靠地对50多段文字中的提及进行聚类。需采用滑动窗口结合合并的方法。

## 使用它

2026年的技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 英语，单文档 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神经网络核心指代模型 |
| 多语言 | 在 OntoNotes 或 Multilingual CoNLL 数据集上训练的 SpanBERT / XLM-R 模型 |
| 跨文档事件核心指代 | 专用端到端模型（2025–26年最先进技术） |
| 快速实现的LLM基线方案 | 配备结构化输出核心指代提示词的 GPT-4o / Claude |
| 生产环境对话系统 | 基于规则的备用方案 + 主要的神经网络方案 + 对关键字段进行人工审核 |

2026年采用的集成模式：首先运行命名实体识别，再执行核心指代分析，最后将核心指代聚类结果合并到命名实体中。后续任务会针对每个聚类处理一个实体，而非每个提及都视为独立实体。

## 发布它

保存为 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 练习题

1. **简单级。** 在 `code/main.py` 中运行基于规则的解析器，处理5段人工编写的段落，并根据真实标注结果衡量提及链接的准确率。
2. **中等级。** 使用预训练的神经网络核心指代模型处理一篇新闻文章，将生成的指代群与自己手动标注的结果进行对比，分析模型的错误所在。
3. **高级别。** 构建一个经过核心指代增强的命名实体识别流程：首先进行命名实体识别，再通过核心指代群合并结果。在100篇文章上，衡量该流程相比仅使用命名实体识别的方法在实体覆盖率方面的提升效果。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 提及 | 引用 | 指向某个实体（名称、代词、名词短语）的文本片段。 |
| 先行词 | “它”所指代的内容 | 后续提及内容与之共指的先前提及内容。 |
| 群集 | 实体的所有提及 | 所有指向同一现实世界实体的提及集合。 |
| 回指 | 向后引用 | 后续提及内容指向先前的内容（如“他”→“John”）。 |
| 前指 | 向前引用 | 先行提及内容指向后续的内容（如“当他到达时，John...”）。 |
| 桥接 | 隐含引用 | “我买了一辆车。它的轮子坏了。”（即那辆车的轮子。） |
| CoNLL F1 | 排名榜上的数值 | MUC、B³、CEAF-φ4 F1 分数的平均值。 |

## 延伸阅读

- [Jurafsky & Martin, SLP3 第 26 章 — 共指消解与实体链接](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 标准教材章节。
- [Lee 等人 (2017). 端到端神经网络共指消解](https://arxiv.org/abs/1707.07045) — 基于词段的端到端方法。
- [Joshi 等人 (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 提升共指消解性能的预训练模型。
- [Pradhan 等人 (2012). CoNLL-2012 共同任务](https://aclanthology.org/W12-4501/) — 该领域的基准测试。
- [Hobbs (1978). 代词指代消解](https://www.sciencedirect.com/science/article/pii/0024384178900064) — 基于规则的经典方法。
