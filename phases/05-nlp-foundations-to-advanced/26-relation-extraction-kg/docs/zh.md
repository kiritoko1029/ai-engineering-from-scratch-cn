# 关系抽取与知识图谱构建

> 名词实体识别（NER）用于发现实体，实体链接则负责为这些实体建立关联。关系抽取则用于找出实体之间的连接边。知识图谱由节点、边以及它们的来源信息共同构成。

**类型：** 构建
**语言：** Python
**先修课程：** 第5阶段 · 06（名词实体识别），第5阶段 · 25（实体链接）
**耗时：** 约60分钟

## 问题所在

一名分析师读到：“蒂姆·库克于2011年成为苹果公司的首席执行官。”相关事实如下：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

关系抽取（Relation Extraction，RE）可将自由文本转换为结构化的三元组 `(subject, relation, object)`。通过对大量语料进行聚合处理，即可构建知识图谱；进一步通过聚合与查询操作，则可为检索增强生成（RAG）、数据分析或合规审计提供推理基础。

2026年的问题在于：大型语言模型在提取关系时过于积极，甚至过度活跃，从而会凭空编造出原文并未提及的三元组。在没有来源追溯机制的情况下，很难区分真实的三元组与看似合理的虚构内容。针对这一问题的解决方案便是采用类似AEVS的“锚定与验证”流程。

## 概念概述

![文本 → 三元组 → 知识图谱](../assets/relation-extraction.svg)

**三元组形式。** `(主体实体, 关系类型, 对象实体)`。关系来源于封闭本体（如 Wikidata 属性、FIBO、UMLS）或开放集合（类似 OpenIE，无限制）。

**三种提取方法。**

1. **基于规则/模式的方法。** 使用 Hearst 模式：“X 如同 Y” → `(Y, isA, X)`，并结合手工编写的正则表达式。该方法较为脆弱，但精确且可解释。
2. **监督分类器方法。** 给定句子中的两个实体提及，从固定集合中预测二者之间的关系。在 TACRED、ACE、KBP 数据集上进行训练。属于 2015–2022 年的标准方法。
3. **生成式大语言模型方法。** 通过提示让模型直接输出三元组。开箱即用，但需要确保来源可信，否则容易产生看似合理但实际上错误的内容。

**AEVS（锚点提取-验证-补充，2026年）。** 当前的幻觉抑制框架：

- **锚点定位。** 精确识别每个实体片段及关系短语片段的起始与结束位置。
- **三元组生成。** 为这些锚点片段生成对应的三元组。
- **验证环节。** 将每个三元组元素与源文本进行比对，剔除无依据的内容。
- **补充步骤。** 通过全面检查确保没有遗漏任何被标记的片段。

该框架能显著减少幻觉现象。虽然需要更多计算资源，但具备可审计性。

**封闭本体与开放本体的权衡。**

- **封闭本体。** 具有固定的属性列表（例如 Wikidata 拥有 11,000 多个属性）。具有可预测性、易于查询，但难以新增属性。
- **开放 IE 方法。** 任何表述都可以作为关系使用。召回率较高，但精确度较低，且查询较为复杂。

实际生产环境中的知识图谱通常采用混合方式：先用开放 IE 方法进行数据发现，随后将关系规范化到封闭本体中，最后合并到主知识图中。

## 构建它

### 步骤 1：基于模式的提取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的小型提取器代码请参见 `code/main.py`。由于易于调试，Hearst 模式仍被用于特定领域的处理流程中。

### 步骤 2：监督式关系分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSequenceClassification.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
triples = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是一种序列到序列的关系抽取器：输入为文本，输出为三元组，且这些三元组的属性 ID 均来自 Wikidata。该模型基于远距离监督数据进行了微调，属于标准的开源基准模型。

### 步骤 3：基于锚点的 LLM 提示词提取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

将每个返回的 span 与源文本进行比对。若存在 `text[start:end] != triple_entity` 的情况，则予以拒绝。这就是 AEVS “验证”步骤的最简形式。

### 步骤 4：将数据规范化至封闭本体中

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

规范化工作通常占整个工程工作的 60% 到 80%，请为此预留预算。

### 步骤 5：构建一个小图并进行查询

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

这是每个基于知识图谱的RAG系统的核心组成部分。可通过RDF三元组存储（如Blazegraph、Virtuoso）、属性图数据库（如Neo4j）或向量增强型图存储来扩展其规模。

## 常见陷阱

- **实体链接前的指代消解。** “他创立了 Apple” —— 实体链接系统需要先明确“他”指的是谁。需先进行指代消解（第24课）。
- **实体规范化。** “Apple Inc”与“Apple”必须映射为同一个节点。需先执行实体链接操作（第25课）。
- **虚假三元组。** 大语言模型可能会生成文本中并不存在的三元组。需进行跨度验证。
- **关系规范化偏差。** 开放式的关系表述往往不一致，如“出生于”、“来自”、“是……的本地人”等。必须将其归约为标准ID，否则图表将无法被查询。
- **时间错误。** “蒂姆·库克是 Apple 的CEO”——这一说法现在成立，但在2005年则不成立。许多关系都具有时间限制。应使用限定符（如Wikidata中的`P580`表示开始时间，`P582`表示结束时间）。
- **领域不匹配。** REBEL模型是在维基百科上训练的。法律、医学和科学领域的文本通常需要经过领域微调的实体链接模型。

## 使用它

2026年技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 快速部署、通用领域 | REBEL或LlamaPred，结合Wikidata规范标准化处理 |
| 领域特定场景（生物医学、法律等） | SciREX风格的领域微调 + 自定义本体 |
| 基于LLM提示的、需审核的输出 | AEVS流程：锚定 → 提取 → 验证 → 补充 |
| 大量新闻信息抽取 | 基于模式识别与监督学习的混合方法 |
| 从零构建知识图谱 | Open IE + 手动规范标准化步骤 |
| 时间序列知识图谱 | 使用限定词（开始/结束时间、具体时间点）进行提取 |

集成流程：命名实体识别 → 共指消解 → 实体链接 → 关系抽取 → 本体映射 → 图结构加载。每个阶段均为潜在的质量控制环节。

## 发布它

保存为 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 练习题

1. **简单。** 在 `code/main.py` 中运行模式提取器，对5句新闻文章进行处理，并手动检查精确度。
2. **中等难度。** 使用REBEL（或小型大语言模型）处理相同的句子，对比生成的三元组。哪种提取器的精确度更高？召回率呢？
3. **困难。** 构建AEVS流程：先使用大语言模型进行提取，再根据源文本验证提取的片段。在50句维基百科风格的句子上，分别测量验证步骤前后的幻觉率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 三元组 | 主体-关系-客体 | 表示知识图谱的基本单元，即 `(s, r, o)` 元组。 |
| 开放式关系短语 | 可提取任意内容 | 使用开放词汇表构建的关系短语；召回率高但精确度低。 |
| 封闭本体 | 固定模式 | 关系类型被严格限定（如 Wikidata、UMLS、FIBO）。 |
| 规范化 | 对所有内容进行标准化处理 | 将表层名称/关系映射为标准编号。 |
| AEVS | 基于锚点的提取方法 | 包含锚定、提取、验证和补充四个步骤的流程（2026年提出）。 |
| 起源信息 | 真实来源链接 | 每个三元组都包含指向其来源文档的文档编号及字符范围。 |
| 远程监督 | 低成本标注数据 | 通过将文本与现有知识图谱对齐来生成训练数据。 |

## 延伸阅读

- [Mintz 等人 (2009). 无需标注数据的远距离监督关系抽取方法](https://www.aclweb.org/anthology/P09-1113.pdf) —— 关于远距离监督的论文。
- [Huguet Cabot, Navigli (2021). REBEL：基于端到端语言生成的关系抽取模型](https://aclanthology.org/2021.findings-emnlp.204.pdf) —— 序列到序列关系的主流模型。
- [Wadden 等人 (2019). 基于上下文化跨度表示的实体、关系与事件抽取方法（DyGIE++）](https://arxiv.org/abs/1909.03546) —— 实体识别与关系抽取的联合方法。
- [AEVS —— 锚点提取、验证与补充框架](https://www.mdpi.com/2073-431X/15/3/178) —— 2026年提出的幻觉抑制设计方案。
- [Wikidata SPARQL 教程](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) —— 标准图查询指南。
