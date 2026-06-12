# 实体链接与消歧

> NER 已识别出“Paris”。实体链接模块需要确定：是指法国的巴黎？帕丽斯·希尔顿？德克萨斯州的巴黎？还是特洛伊王子帕里斯？若不进行实体链接，您的知识图谱将始终存在歧义。

**类型：** 构建
**语言：** Python
**前置要求：** 第 5 阶段 · 06（NER），第 5 阶段 · 24（共指消解）
**耗时：** 约 60 分钟

## 问题所在

有一句话：“Jordan 打败了媒体。”我们将 “Jordan” 标记为 PERSON 类实体，这是正确的。但具体是哪位 Jordan 呢？

- 迈克尔·乔丹（篮球运动员）？
- 迈克尔·B·乔丹（演员）？
- 迈克尔·I·乔丹（伯克利大学机器学习教授——在机器学习论文中确实存在这种混淆）？
- 乔丹国（国家名）？
- 乔丹（希伯来语中的名字）？

实体链接（EL）任务旨在将每个提及的实体映射到知识库中的唯一条目：Wikidata、Wikipedia、DBpedia 或您自定义的领域知识库。该任务包含两个子任务：

1. **候选项生成**：给定 “Jordan” 这一词汇，哪些知识库条目是合理的候选？
2. **消歧**：结合上下文信息，确定哪个候选项才是正确的实体。

这两个步骤均可通过机器学习方法实现，并且有相应的基准测试。整个处理流程在过去十年中一直保持稳定——唯一变化的是消歧模块的性能水平。

## 概念概述

![实体链接流程：提及词 → 候选实体 → 消歧后的实体](../assets/entity-linking.svg)

**候选实体生成。** 给定提及词的原始形式（如“Jordan”），在别名索引中查找对应的候选实体。维基百科的别名词典涵盖了大多数命名实体：“JFK”对应John F. Kennedy、Jacqueline Kennedy、JFK机场以及电影《JFK》。典型的索引每条提及词可返回10到30个候选实体。

**消歧方法：三种途径。**

1. **先验概率 + 上下文（Milne & Witten, 2008）。** 计算公式为 `P(entity | mention) × context-similarity(entity, text)`。该方法效果良好、速度较快，且无需训练。
2. **基于嵌入的方法（ESS / REL / Blink）。** 首先对提及词及上下文进行编码，同时编码每个候选实体的描述，最后通过计算余弦相似度值选取最优结果。这是2020年至2024年间的默认方法。
3. **生成式方法（GENRE, 2021；基于大语言模型的方法，2023年起）。** 逐词解码实体的标准名称，同时受限于有效的实体名称字典树，从而确保输出结果为合法的知识库标识符。

**端到端模型与流水线模型。** 现代模型（如ELQ、BLINK、ExtEnD、GENRE）可在单次处理中完成命名实体识别、候选实体生成及消歧操作。而在实际生产环境中，流水线系统仍占据主导地位，因为用户可以灵活更换各个组件。

### 两项测量值

- **提及召回率（候选项生成）**：在所有真实提及中，正确的知识库条目出现在候选列表中的比例。该指标为整个处理流程的下限。
- **消歧准确性 / F1值**：在已有正确候选项的情况下，前1个候选项正确的频率。

必须同时报告这两个指标。若某个系统的消歧准确率为99%，但提及召回率为80%，则该整体处理流程的得分即为80%。

## 构建它

### 步骤 1：根据维基百科的重定向链接构建别名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

维基百科别名数据：约1800万对（别名，实体）记录。可从Wikidata数据导出文件中下载，并以倒排索引形式存储。

### 步骤 2：基于上下文的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

杰卡德重叠度仅是一种简化的衡量方式。建议使用嵌入向量的余弦相似度来替代（Transformer版本的实现见 `code/main.py` 的第 2 步）。

### 步骤 3：基于嵌入的方法（BLINK 风格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

在索引构建时，将每个知识库实体嵌入一次。在查询处理时，将提及内容及其上下文嵌入一次，然后与候选项集合进行点积运算，并选取得分最高的候选项。

### 第 4 步：生成式实体链接（概念）

GENRE 按字符逐个解码实体的维基百科标题。通过受限解码机制（参见第20课）来确保仅能输出有效的标题。该功能与基于知识库的trie结构实现了紧密集成。其现代演进版本为REL-GEN以及具备结构化输出功能的LLM提示式EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

结合白名单机制（即定义 `choice` 的范围），这就是 2026 年最简化的可部署 EL 流水线。

### 第 5 步：在 AIDA-CoNLL 上进行评估

AIDA-CoNLL 是标准的实体链接评估基准：包含 1,393 篇路透社文章、34,000 个提及项以及维基百科中的实体。需报告在知识库内的准确率（`P@1`）以及知识库外的 NIL 检测率。

## 常见陷阱

- **NIL处理。**部分提及的实体并未收录在知识库中（如新兴实体或鲜为人知的个人）。系统必须预测为NIL，而非猜测错误的实体。该指标将单独衡量。
- **提及边界错误。**上游命名实体识别模块会遗漏部分词组范围（例如将“Bank of America”仅标记为“Bank”），从而导致实体链接召回率下降。
- **流行度偏差。**经过训练的系统倾向于过度预测高频出现的实体。在机器学习论文中提及“Michael I. Jordan”时，系统往往将其关联到篮球运动员乔丹。
- **跨语言实体链接。**需将中文文本中的提及映射到英文维基百科的实体。这需要使用多语言编码器或进行翻译处理。
- **知识库过时问题。**新成立的公司、事件及人物可能未包含在去年的维基百科数据集中。生产流程需要设置定期更新机制。

## 使用它

2026年的技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 通用英语内容 + 维基百科 | BLINK或REL |
| 跨语言场景，知识库为维基百科 | mGENRE |
| 适合大语言模型使用，每日提及次数较少 | 向Prompt Claude/GPT-4提供候选列表并配合受限JSON格式输入 |
| 领域专用知识库（医疗、法律等） | 基于BERT的自定义模型，结合知识库感知的检索机制，并采用AIDA风格的数据集进行微调 |
| 极低延迟需求 | 仅使用精确匹配先验模型（Milne-Witten基线模型） |
| 最先进的研究方案 | GENRE / ExtEnD / generative LLM-EL |

2026年投入生产的处理流程：对每个提及项依次执行命名实体识别、共指消解和实体链接操作，随后将各聚类合并为每个聚类中的一个标准实体。最终输出结果为文档中每个实体对应一个知识库ID，而非每个提及项都单独对应一个ID。

## 发布它

保存为 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 练习题

1. **简单。** 在 `code/main.py` 中实现基于先验信息与上下文的消歧器，处理 10 个易产生歧义的提及（Paris、Jordan、Apple）。手动标注正确的实体，并计算准确率。
2. **中等难度。** 使用句子编码器对 50 个易产生歧义的提及进行编码，生成每个候选实体的描述向量。将基于嵌入向量的消歧方法与 Jaccard 上下文重叠度方法进行对比。
3. **高难度。** 构建一个包含 1,000 个实体的领域知识库（例如公司内部的员工和产品信息）。实现端到端的命名实体识别与实体链接功能，并在 100 条保留的句子上评估精确率和召回率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 实体链接（EL） | 链接到维基百科 | 将某个提及映射到唯一的知识库条目。 |
| 候选项生成 | 可能是谁？ | 为某个提及返回一组合理的知识库条目候选列表。 |
| 消歧义处理 | 选择正确的那个 | 利用上下文对候选项进行评分，选出最优结果。 |
| 别名索引 | 查找表 | 建立从表面形式到候选实体的映射关系。 |
| NIL | 不在知识库中 | 明确预测不存在匹配的知识库条目。 |
| KB | 知识库 | 可以是 Wikidata、维基百科、DBpedia 或者自定义领域的知识库。 |
| AIDA-CoNLL | 基准数据集 | 包含 1,393 篇带有标准实体链接的路透社文章。 |

## 延伸阅读

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — 基于先验信息与上下文的方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 基于嵌入技术的常用方法。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 具有约束解码机制的生成式实体链接方法。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — 该领域的基准论文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 开源的生产级实体链接工具栈。
