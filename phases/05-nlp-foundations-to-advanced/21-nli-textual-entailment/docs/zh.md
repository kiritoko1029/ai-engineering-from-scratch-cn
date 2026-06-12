# 自然语言推理——文本蕴含关系

> “t entails h”表示：当人类阅读t时，会推断出h为真。NLI任务旨在预测蕴含、矛盾或中立关系。表面上看似乎枯燥，但在实际应用中却至关重要。

**类型：** 学习
**语言：** Python
**先修课程：** 第5阶段 · 05（情感分析）、第5阶段 · 13（问答系统）
**时长：** 约60分钟

## 问题所在

你构建了一个摘要生成器，它输出了摘要。如何确定该摘要中不存在幻觉内容？

你构建了一个聊天机器人，它回答“是”。如何确认这个答案有来源文本作为支撑？

你需要对10,000篇新闻文章进行主题分类，但目前没有训练标签。能否复用现有模型？

这三个问题本质上都可以归结为自然语言推理（NLI）。NLI的核心问题是：给定前提`t`和假设`h`，`h`是蕴含于`t`、与`t`矛盾，还是中立（无关）？

- **幻觉检测**：`t`为源文档，`h`为摘要中的陈述。若不满足蕴含关系，则视为幻觉。
- **基于事实的问答**：`t`为检索到的文本片段，`h`为生成的答案。若不满足蕴含关系，则视为编造内容。
- **零样本分类**：`t`为文档本身，`h`为人工表述的标签（如“这是关于体育的”）。若满足蕴含关系，则预测出的标签即为正确标签。

同一项技术，三种应用场景。正因如此，每一个RAG评估框架都会在底层集成NLI模型。

## 概念概述

![NLI：三向分类，前提与假设](../assets/nli.svg)

**三种标签。**

- **蕴含关系。** `t` → `h`。“猫在垫子上”蕴含“存在一只猫。”
- **矛盾关系。** `t` → ¬`h`。“猫在垫子上”与“没有猫”相互矛盾。
- **中性关系。** 无法推断任何结论。“猫在垫子上”与“猫饿了”之间无逻辑关联。

**并非严格的逻辑蕴含。** NLI属于*自然语言推理*——即普通人类读者的推断方式，而非严格的逻辑推导。“约翰遛了他的狗”在NLI中可视为“约翰有一只狗”的蕴含关系，但在严格的一阶逻辑中，只有当将“拥有”概念形式化后才能成立。

**数据集。**

- **SNLI**（2015年）。包含57万对由人类标注的数据对，前提为图片标题。属于窄领域数据。
- **MultiNLI**（2017年）。涵盖10个领域的43.3万对数据。是2026年的标准训练语料库。
- **ANLI**（2019年）。对抗性NLI。人类专门设计了用于破坏现有模型的示例，难度更高。
- **DocNLI, ConTRoL**（2020–21年）。前提为文档长度级文本。用于测试多跳推理及长距离逻辑推断能力。

**模型架构。** 变换器编码器（如BERT、RoBERTa、DeBERTa）会读取`[CLS] 前提 [SEP] 假设 [SEP]`这样的输入结构。`[CLS]`向量会被送入三向softmax函数进行计算。在MNLI数据集上训练，使用保留的测试集进行评估，通常能在同分布数据对上达到90%以上的准确率。

**通过NLI实现零样本分类。** 给定一段文档及候选标签后，将每个标签转化为一个假设（例如“这段文本是关于体育的”），分别计算其蕴含概率，最终选择概率最高的标签。这正是Hugging Face的`zero-shot-classification`流程所采用的机制。

## 构建它

### 步骤 1：运行预训练的 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

在面向生产环境的自然语言理解任务中，`facebook/bart-large-mnli` 与 `microsoft/deberta-v3-large-mnli` 是常用的开源默认模型。其中，DeBERTa-v3 的性能在各项排行榜上均处于领先地位。

### 步骤 2：零样本分类

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

默认模板为“This example is about {label}.”。可通过`hypothesis_template`进行自定义。无需训练数据，也无需微调，开箱即用。

### 步骤 3：RAG 的一致性校验

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

这正是 RAGAS 准确性的核心所在。需将生成的答案拆分为独立的断言，然后逐一对照检索到的上下文进行验证，并报告其中具有蕴含关系的断言比例。

### 步骤 4：手工实现的自然语言理解分类器（概念性）

请参阅 `code/main.py` 中仅使用标准库实现的示例：通过词汇重叠检测与否定词识别来比较前提和假设。虽然无法与Transformer模型相媲美，但它展示了该任务的架构：输入两段文本，输出三类标签，损失函数为基于 `{entail, contradict, neutral}` 的交叉熵损失。

## 常见陷阱

- **仅基于假设的快捷方法。** 由于“not”、“nobody”、“never”等词汇与矛盾关系相关，模型仅通过假设即可以约60%的准确率预测SNLI任务中的标签，这为检测标签泄漏提供了良好的基准。
- **词汇重叠启发式方法。** 子序列启发式方法（即“任何子序列都蕴含关系”）虽能通过SNLI测试，但在HANS/ANLI任务中表现不佳，应采用对抗性基准进行测试。
- **文档长度带来的性能下降。** 单句结构的NLI模型在处理长篇幅前提时，F1分数会下降20个百分点以上；处理长上下文时应使用经过DocNLI训练的模型。
- **零样本模板敏感性。** 不同的模板表述方式，如“This example is about {label}”、“{label}”以及“The topic is {label}”，会导致准确率出现10个百分点以上的差异，需对模板进行优化调整。
- **领域不匹配问题。** MNLI是在通用英语数据上训练的，而法律、医学和科学文本则需要专门的领域NLI模型（如SciNLI、MedNLI）。

## 使用它

2026年技术栈：

| 应用场景 | 模型 |
|---------|-------|
| 通用NLI任务 | `microsoft/deberta-v3-large-mnli` |
| 高速/边缘设备 | `cross-encoder/nli-deberta-v3-base` |
| 零样本分类（轻量级） | `facebook/bart-large-mnli` |
| 文档级NLI任务 | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多语言处理 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG系统中的幻觉检测 | RAGAS / DeepEval内置的NLI层 |

2026年核心模式：NLI是文本理解领域的通用解决方案。每当需要判断“A是否支持B？”或“A是否与B矛盾？”时，应在调用其他大语言模型之前先使用NLI功能。

## 发布它

保存为 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 练习题

1. **简单级。** 使用 `facebook/bart-large-mnli` 模型处理20组手工构造的三元组（包含前提、假设和标签），这些三元组涵盖了所有三个类别。计算模型的准确率。随后加入对抗性的“子序列启发式”陷阱（如“我没有吃蛋糕”与“我吃了蛋糕”），观察模型性能是否因此下降。
2. **中等级。** 在100条AG News新闻标题上，将零样本模板 `"This text is about {label}"`、`"The topic is {label}"` 以及直接使用 `{label}"` 进行对比，统计准确率的波动情况。
3. **高级别。** 构建一个RAG忠实度检测器：通过原子性声明分解对每个声明进行NLI评估。在50条带有黄金标准上下文的RAG生成答案上进行测试，计算误报率和漏报率，并与手工标注的结果进行对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| NLI | 自然语言推理 | 对前提与假设之间的关系进行三分类。 |
| RTE | 文本蕴含识别 | NLI的旧称；任务相同。 |
| 蕴含 | “t 推出 h” | 在给定 t 的情况下，普通读者会认为 h 为真。 |
| 矛盾 | “t 排除 h” | 在给定 t 的情况下，普通读者会认为 h 为假。 |
| 中立 | “无法判定” | 从 t 无法推断出 h 的真假。 |
| 零样本分类 | 将 NLI 用作分类器 | 将标签表述为假设，选择蕴含度最高的选项。 |
| 准确性 | 答案是否有依据？ | 对（检索到的上下文、生成的答案）进行 NLI 判定。 |

## 延伸阅读

- [Bowman 等人（2015）。用于学习自然语言推理的大型标注语料库](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams、Nangia、Bowman（2017）。用于通过推理实现句子理解的广覆盖挑战语料库](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie 等人（2019）。对抗性自然语言推理](https://arxiv.org/abs/1910.14599) — ANLI 基准测试。
- [Yin、Hay、Roth（2019）。零样本文本分类的基准测试](https://arxiv.org/abs/1909.00161) — NLI-as-classifier。
- [He 等人（2021）。DeBERTa：具有解耦注意力机制的增强版 BERT](https://arxiv.org/abs/2006.03654) — 2026 年自然语言推理领域的核心模型。
