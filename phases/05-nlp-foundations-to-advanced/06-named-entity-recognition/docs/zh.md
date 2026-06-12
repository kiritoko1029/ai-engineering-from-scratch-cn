# 命名实体识别

> 提取实体名称。听起来很简单，但一旦遇到边界模糊、嵌套实体以及领域专用术语，就会变得十分棘手。

**类型：** 构建
**语言：** Python
**先修课程：** 第 5 阶段 · 02（词袋模型 + TF-IDF）、第 5 阶段 · 03（词嵌入）
**耗时：** 约 75 分钟

## 问题所在

“苹果公司因iPhone搜索业务协议问题在美国起诉了谷歌。”涉及五个实体：Apple（ORG，组织）、Google（ORG，组织）、iPhone（PRODUCT，产品）、search deal（可能属于业务协议类别）以及US（GPE，地理区域）。一个优秀的NER系统能够准确提取所有这些实体并标注正确的类型；而劣质的系统则会遗漏iPhone，将水果苹果与公司苹果混淆，并将“US”错误地标记为PERSON（人物）。

NER是所有结构化信息抽取流程的核心基础。无论是简历解析、合规日志扫描、医疗记录匿名化、搜索查询理解、聊天机器人回复的上下文关联，还是法律合同提取，都离不开它的支持。我们往往看不到它的存在，却始终依赖着它。

本课程将带领大家从传统的处理方法（基于规则、HMM、CRF）逐步学习到现代技术（BiLSTM-CRF，再到Transformer模型）。每一种新方法都在解决前一种方法的局限性，这种规律本身便是本课程的核心内容。

## 概念概述

**BIO 标签法**（或称 BILOU）将实体抽取问题转化为序列标注问题。需为每个标记打上 `B-TYPE`（实体起始）、`I-TYPE`（实体内部）或 `O`（非实体区域）的标签。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多标记实体链：`New B-GPE`、`York I-GPE`、`City I-GPE`。能够理解BIO格式的模型可提取任意范围的文本片段。

架构发展历程：

- **基于规则的方法。** 使用正则表达式结合地名辞典进行查询。对已知实体具有高精度，但对新出现的实体完全无法识别。
- **隐马尔可夫模型（HMM）。** 具有标签给定标记的发射概率以及标签间转换的概率。通过维特比算法解码。需在带标签的数据上进行训练。
- **条件随机场（CRF）。** 类似于HMM，但具备判别能力，因此可以融合各种特征（如单词形状、大小写、相邻单词等）。在2026年，它仍是低资源环境下的传统主流方案。
- **BiLSTM-CRF模型。** 使用神经网络特征替代手工设计的特征。LSTM可双向读取句子，顶层的CRF层则用于确保标签序列的一致性。
- **基于Transformer的模型。** 通过为BERT添加标记分类头进行微调。准确率最高，但计算资源消耗最大。

```figure
ner-bio-tagging
```

## 构建它

### 步骤 1：BIO 标签辅助工具

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步骤 2：手工构建特征

对于传统的（非神经网络）命名实体识别任务，特征是决定成败的关键。常用的特征包括：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 的返回值为 `xXxxxx`。`word_shape("USA-2024")` 的返回值为 `XXX-dddd`。专有名词的大小写模式是判断其属性的重要特征。

### 步骤 3：基于规则的简单模型 + 字典基准模型

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

生产级词汇表包含了从维基百科和DBpedia中抓取的数百万条词条，覆盖度相当高。然而，对于同义消歧（如区分“Apple”这家公司与水果）的处理效果极差。正因如此，统计模型反而更胜一筹。

### 步骤 4：CRF 步骤（仅示意图，非完整实现）

若没有概率论基础，仅用50行代码从头实现完整CRF模型并无太大意义。建议改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 分别代表 L1 和 L2 正则化项。设置 `all_possible_transitions=True` 可使模型学习到非法序列（例如在 `O` 之后出现 `I-ORG`）出现的概率极低，这正是 CRF 在无需手动编写约束条件的情况下实现 BIO 一致性的机制。

### 步骤 5：BiLSTM-CRF 的作用

特征为学习得到的。输入为词元嵌入（GloVe或fastText）。LSTM可进行从左到右及从右到左的读取。拼接后的隐藏状态会传递至CRF输出层。CRF依然负责确保标签序列的一致性，而LSTM则用学习到的特征替代手工设计的特征。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

对于 CRF 层，可使用 `torchcrf.CRF`（通过 pip 安装 pytorch-crf）。与手工实现的 CRF 相比，其优势虽可衡量，但除非拥有数以万计的带标签句子，否则这种优势并不会像预期那样显著。

## 使用它

spaCy 开箱即用，即可提供生产级命名实体识别功能。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

请注意，`iPhone` 被标记为 `ORG` 而非 `PRODUCT` —— spaCy 的小型模型在产品实体识别方面的能力较弱。大型模型（`en_core_web_lg`）的表现更好。而基于 Transformer 的模型（`en_core_web_trf`）表现则更为优异。

用于基于 BERT 的命名实体识别的 Hugging Face：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 会将连续的 B-X、I-X 令牌合并为一个区间。如果不设置该参数，系统将生成令牌级别的标签，需手动进行合并操作。

### 基于大语言模型的命名实体识别（2026年方案）

零样本与少样本大语言模型命名实体识别在许多领域已能与微调后的模型相媲美，且在标注数据稀缺时表现更为出色。

- **零样本提示法。** 向大语言模型提供实体类型列表及示例结构，并要求其输出 JSON 格式的结果。该方法开箱即用，但在全新领域的准确率仅为中等水平。
- **ZeroTuneBio 风格的提示法。** 将任务分解为候选项提取 → 含义解释 → 判定 → 重新核查四个步骤。这种多阶段提示方式（非单次提示）能显著提升生物医学领域命名实体识别的准确率，该模式同样适用于法律、金融和科学领域。
- **结合 RAG 的动态提示法。** 在每次推理时，从少量已标注的示例集中检索最相似的案例，并即时构建少样本提示。在 2026 年的基准测试中，该方法使 GPT-4 在生物医学命名实体识别任务中的 F1 分数比静态提示法高出 11-12%。
- **按实体类型分步处理法。** 对于长文档而言，一次性提取所有实体类型的单次调用会随着文档长度的增加而降低召回率。建议为每种实体类型分别执行一次提取操作。虽然推理成本较高，但准确率会有显著提升。这是临床记录和法律合同处理中的标准做法。

2026 年的实战建议：在收集训练数据之前，先使用大语言模型的零样本基线模型进行测试。通常其 F1 分数已足够理想，无需再进行微调。

### 在哪些场景下传统NER仍具有优势

即便在有了大语言模型的情况下，以下场景下传统命名实体识别仍更具优势：

- 延迟预算低于 50 毫秒。
- 您拥有数千个标注过的示例，并且需要 98% 以上的 F1 分数。
- 所属领域具有稳定的本体结构，预训练的 CRF 或 BiLSTM 模型能够很好地迁移应用。
- 监管要求必须使用本地部署的非生成式模型。

### 出现故障的位置

- **领域偏移问题。** 在法律合同数据上使用 CoNLL 标签集训练的命名实体识别模型性能不如通用地名词典。需针对特定领域进行微调。
- **嵌套实体问题。** “Bank of America Tower” 同时属于 ORG（组织）和 FACILITY（设施）类别。标准的 BIO 标签体系无法表示这些重叠的实体范围，需要使用嵌套命名实体识别模型（多轮处理或基于跨度模型的方法）。
- **长实体问题。** 如“United States Federal Deposit Insurance Corporation”这类长名称，基于词级的模型有时会将其拆分。可利用 `aggregation_strategy` 参数进行聚合处理，或通过后处理方式解决。
- **稀疏类型标签问题。** 医学领域的命名实体识别标签如 DRUG_BRAND（药品品牌）、ADVERSE_EVENT（不良事件）、DOSE（剂量）等，通用模型往往无法识别。Scispacy 和 BioBERT 是处理此类问题的常用起点。

## 发布它

保存为 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 练习题

1. **简单。** 实现 `bio_to_spans` 函数（即 `spans_to_bio` 的逆函数），并在 10 句文本上验证该函数的往返一致性。
2. **中等难度。** 使用上述 sklearn-crfsuite CRF 模型在 CoNLL-2003 英语命名实体识别数据集上进行训练。通过 `seqeval` 工具报告每个实体的 F1 分数，典型结果约为 84 分。
3. **高难度。** 在特定领域的命名实体识别数据集（如医学、法律或金融领域）上对 `distilbert-base-cased` 模型进行微调，并将其与 spaCy 的小型模型进行对比。需记录数据泄漏检测的结果，并总结令你感到意外之处。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| NER | 提取名称 | 为词元区间标注类型（PERSON、ORG、GPE、DATE 等）。 |
| BIO | 标注方案 | `B-X` 表示起始位置，`I-X` 表示延续部分，`O` 表示该位置不属于任何实体。 |
| BILOU | 改进版 BIO | 增加了 `L-X`（最后一个词元）和 `U-X`（单元）标注，以更清晰地界定实体边界。 |
| CRF | 结构化分类器 | 不仅考虑输出概率，还建模标签之间的转换关系，从而确保序列的合法性。 |
| Nested NER | 重叠实体 | 某个词元区间属于与其中某个子区间不同的实体。BIO 标注方案无法表示这种情况。 |
| Entity-level F1 | 正确 NER 评估指标 | 预测的词元区间必须与真实区间完全一致。基于词元级别的 F1 值会高估准确率。 |

## 延伸阅读

- [Lample 等人 (2016). 用于命名实体识别的神经网络架构](https://arxiv.org/abs/1603.01360) — 即 BiLSTM-CRF 论文。属于经典文献。
- [Devlin 等人 (2018). BERT：深度双向变换器的预训练](https://arxiv.org/abs/1810.04805) — 提出了后来成为标准的词元分类模式。
- [spaCy 语言特征 — 命名实体](https://spacy.io/usage/linguistic-features#named-entities) — 关于 `Doc.ents` 和 `Span` 中各属性的实用参考资料。
- [seqeval](https://github.com/chakki-works/seqeval) — 正确的评估指标库。应始终使用它。
