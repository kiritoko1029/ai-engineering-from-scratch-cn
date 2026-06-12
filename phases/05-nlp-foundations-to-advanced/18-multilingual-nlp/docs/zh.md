# 多语言自然语言处理

> 单一模型，支持100多种语言，其中大多数甚至无需训练数据。跨语言迁移是2020年代最实用的奇迹。

**类型：** 学习
**编程语言：** Python
**先修知识：** 第5阶段 · 04（GloVe、FastText、子词分解），第5阶段 · 11（机器翻译）
**时长：** 约45分钟

## 问题所在

英语拥有数十亿条带标签的示例，乌尔都语仅有数千条，而迈蒂利语则几乎没有任何示例。任何面向全球用户的实用自然语言处理系统都必须能够在那些缺乏特定任务训练数据的语言上运行。

多语言模型通过同时用多种语言对同一个模型进行训练来解决这一问题。这种共享的表示机制使得模型能够将高资源语言中学到的技能迁移到低资源语言中。只需对该模型进行英语情感分析的微调，它就能直接在乌尔都语上给出相当准确的情感预测结果。这就是零样本跨语言迁移技术，它彻底改变了自然语言处理技术的推广方式。

本课将介绍其中的权衡因素、典型的模型类型，以及那些刚开始从事多语言工作的团队常会遇到的一个关键决策：选择用于迁移的源语言。

## 概念概述

![通过共享的多语言嵌入空间实现跨语言迁移](../assets/multilingual.svg)

**共享词汇表。** 多语言模型使用在所有目标语言文本上训练得到的 SentencePiece 或 WordPiece 分词器。这些模型的词汇表是共享的：相同的子词单元在不同相关语言中表示相同的词素。英语和意大利语中的 `anti-` 会对应同一个分词。

**共享表示结构。** 在多种语言上通过掩码语言建模预训练的 Transformer 模型能够识别出不同语言中语义相似的句子会产生相似的隐藏状态。mBERT、XLM-R 和 NLLB 均具备此特性。“cat”在英语中的嵌入向量会聚集在法语“chat”和西班牙语“gato”的嵌入向量附近，整句的嵌入向量也是如此。

**零样本迁移。** 在一种语言（通常是英语）的带标签数据上对模型进行微调，之后即可在该模型支持的其他任何语言上进行推理，无需目标语言的标签。对于类型学上相近的语言，迁移效果较好；而对于差异较大的语言，则效果较差。

**小样本微调。** 为目标语言添加 100-500 个带标签的示例，在分类任务中其准确率可提升至英语基线水平的 95-98%。这是多语言自然语言处理中最具成本效益的方法。

## 模型

| 模型 | 发布年份 | 支持语言数量 | 备注 |
|-------|----------|--------------|------|
| mBERT | 2018 | 104种语言 | 基于维基百科训练。首个实用的多语言大型语言模型，但在低资源语言上的表现较弱。 |
| XLM-R | 2019 | 100种语言 | 基于CommonCrawl数据集训练（规模远大于维基百科），为跨语言任务设立了基准。基础版本参数量为2.7亿，大型版本为5.5亿。 |
| XLM-V | 2023 | 100种语言 | 在XLM-R基础上增加了100万词表的词汇量（原版本为25万），在低资源语言上的表现更佳。 |
| mT5 | 2020 | 101种语言 | 采用T5架构，用于多语言文本生成任务。 |
| NLLB-200 | 2022 | 200种语言 | Meta推出的翻译模型，支持包括55种低资源语言在内的多种语言。 |
| BLOOM | 2022 | 46种语言 + 13种编程语言 | 一款开源的1760亿参数多语言大型语言模型。 |
| Aya-23 | 2024 | 23种语言 | Cohere公司推出的多语言大型语言模型，在阿拉伯语、印地语和斯瓦希里语等语言上的表现优异。 |

可根据具体应用场景选择模型。对于分类任务，XLM-R-base是较为稳妥的默认选择。在文本生成任务中，若需进行翻译则选用mT5，若需开放式生成则选用NLLB。对于类大型语言模型的应用，则可通过明确的多语言提示词来搭配使用Aya-23或Claude。

## 源语言选择（2026年研究方向）

大多数团队默认将英语作为微调的源语言。但2026年的最新研究表明，这种做法往往并不正确。

语言相似度比单纯的语料库规模更能预测迁移效果的质量。对于斯拉夫语系的目标语言，德语或俄语通常优于英语；而对于印度语系的目标语言，印地语往往优于英语。**qWALS**相似度指标（2026年提出，基于《世界语言结构图集》中的特征）可用于量化这一现象。**LANGRANK**方法（由Lin等人于2019年在ACL会议上提出）则是另一种较早的方法，它通过综合语言相似度、语料库规模以及语言亲缘关系来对候选源语言进行排序。

实用原则：如果你的目标语言存在在类型学上相近且资源丰富的语言，建议先尝试使用该语言进行微调，然后再与使用英语微调的结果进行对比。

## 构建它

### 步骤 1：零样本跨语言分类

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一个模型，三种语言，同一套API。基于NLI数据训练的XLM-R可通过蕴含技巧很好地应用于分类任务。

### 步骤 2：多语言嵌入空间

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

在嵌入空间中，不同翻译的向量距离较为接近；而不同的英文句子则其距离会更远。正是这一特性使得跨语言检索、聚类及相似度计算得以实现。

### 步骤 3：少样本微调策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

对于 100 到 500 个目标语言示例，`num_train_epochs=5` 和 `learning_rate=2e-5` 是较为稳妥的默认值。过高的学习率会导致多语言对齐失效，从而生成仅支持英语的模型。

## 真正有效的评估方法

- **在保留集上的各语言准确率。** 不进行汇总，因为汇总会掩盖长尾数据的表现。
- **与单语基线的对比测试。** 对于数据量足够的语言，从零开始训练的单语模型有时能优于多语言模型。需进行测试验证。
- **实体级测试。** 针对目标语言中的命名实体。对于非拉丁字母系的文字，多语言模型的分词能力往往较弱。
- **跨语言一致性。** 两种语言表达相同含义时应产生相同的预测结果。需测量两者之间的差异。

## 使用它

2026年技术栈：

| 任务 | 推荐方案 |
|-----|-------------|
| 分类，支持100种语言 | 经过微调的XLM-R-base（约2.7亿参数） |
| 零样本文本分类 | `joeddav/xlm-roberta-large-xnli` |
| 多语言句子嵌入 | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻译，支持200种语言 | `facebook/nllb-200-distilled-600M`（参见第11课） |
| 生成式多语言模型 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低资源语言NLP处理 | XLM-V或基于相关高资源语言的领域特定微调模型 |

若对性能有较高要求，务必预留目标语言下的微调预算。零样本方法仅作为起点，并非最终解决方案。

### 分词成本（低资源语言中常见的问题）

多语言模型在所有语言中共享同一个分词器。该词汇表是基于以英语、法语、西班牙语、汉语和德语为主的语料库训练而成的。对于那些不属于主流语言集的语言，会有三种隐性成本叠加：

- **生成效率税**。资源匮乏的语言文本每词产生的标记数远高于英语。一句印地语句子所需的标记数可能是对应英语句子的3到5倍。这3到5倍的标记数量会占用更多的上下文窗口、降低训练效率并增加延迟。
- **变体恢复税**。任何拼写错误、变音符号差异、Unicode标准化不一致或大小写变化，都会在嵌入空间中形成无关的冷启动序列。模型无法学习母语者视为理所当然的拼写对应关系。
- **容量溢出税**。前两种成本会消耗上下文位置、层数深度以及嵌入维度。留给实际推理的资源，系统性地少于同一模型为资源丰富语言提供的资源。

实际表现表现为：模型在印地语上的训练看似正常，损失曲线也符合预期，评估时的困惑度看起来合理，但生产环境输出却存在细微错误。句子中的词形变化会中途失效，罕见的词形变化也无法被恢复。**仅靠增加数据量无法解决分词器存在的问题。**

缓解措施：选择对目标语言覆盖良好的分词器（XLM-V的100万标记词汇表就是直接的解决方案）；在训练前使用保留的目标文本验证生成效率；对于真正的长尾脚本，采用字节级回退机制（如SentencePiece的`byte_fallback=True`或GPT-2风格的字节级BPE），确保没有任何内容超出模型处理范围。

## 发布它

保存为 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 练习题

1. **简单。** 对英语、法语、印地语和阿拉伯语，分别使用零样本分类流程处理每种语言的10个句子，并报告各自的准确率。预期结果为：法语表现优异，印地语表现尚可，阿拉伯语表现则参差不齐。
2. **中等难度。** 使用 `paraphrase-multilingual-MiniLM-L12-v2` 在一个小型混合语言语料库上构建跨语言检索器。以英语进行查询，即可检索到任意语言的文档，并测量@5召回率。
3. **高难度。** 比较基于英语源数据和印地语源数据对印地语分类任务进行微调的效果。在两种模式下均使用500个目标语言样本进行少样本微调，报告哪种源数据能带来更高的印地语准确率以及提升幅度。这实际上就是LANGRANK理论的简化版应用。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 多语言模型 | 一个模型，多种语言 | 在不同语言之间共享词汇表和参数。 |
| 跨语言迁移 | 用一种语言训练，用于另一种语言 | 在源语言上进行微调，在目标语言上评估，且无需目标语言的标签数据。 |
| 零样本学习 | 无目标语言标签 | 不在目标语言上进行微调即可实现迁移。 |
| 少样本学习 | 较少的目标语言标签 | 使用100-500个目标语言示例进行微调。 |
| mBERT | 第一个多语言大型语言模型 | 基于维基百科预训练的覆盖104种语言的BERT模型。 |
| XLM-R | 标准跨语言基准模型 | 基于CommonCrawl数据预训练的覆盖100种语言的RoBERTa模型。 |
| NLLB | Meta公司的200语言机器翻译系统 | 名称意为“不让任何语言掉队”，支持55种低资源语言。 |

## 延伸阅读

- [Conneau 等人（2019）。大规模无监督跨语言表示学习](https://arxiv.org/abs/1911.02116) —— 即 XLM-R 论文。
- [Pires、Schlinger、Garrette（2019）。多语言 BERT 的多语言程度究竟如何？](https://arxiv.org/abs/1906.01502) —— 开启跨语言迁移研究方向的分析论文。
- [Costa-jussà 等人（2022）。不让任何语言掉队](https://arxiv.org/abs/2207.04672) —— NLLB-200 论文。
- [Üstün 等人（2024）。Aya 模型：一种经过指令微调的开放获取多语言语言模型](https://arxiv.org/abs/2402.07827) —— 即 Cohere 的多语言大语言模型 Aya。
- [语言相似度可预测跨语言迁移学习性能（2026）](https://www.mdpi.com/2504-4990/8/3/65) —— 即 qWALS / LANGRANK 关于源语言的论文。
