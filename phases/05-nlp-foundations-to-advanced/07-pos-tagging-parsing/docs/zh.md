# 词性标注与句法分析

> 有一段时间，语法分析并不流行。后来，由于每个大语言模型流程都需要对结构化提取结果进行验证，语法分析又重新受到了重视。

**类型：** 构建
**编程语言：** Python
**先修课程：** 第5阶段 · 01（文本处理）、第2阶段 · 14（朴素贝叶斯）
**耗时：** 约45分钟

## 问题所在

第 01 课提到，词形还原需要词性标注。如果不知道“running”是动词，词形还原器就无法将其简化为“run”；同样，若不知“better”是形容词，也无法将其简化为“good”。

这一说法实际上掩盖了一个完整的子领域。词性标注用于分配语法类别，而句法分析则用于还原句子的树状结构——即哪些词修饰哪些词，哪个动词支配哪些宾语。传统的自然语言处理技术花费了二十年时间来完善这两项技术。随后，深度学习将它们整合为在预训练变换器之上的标记分类任务，研究界便转向了其他方向。

但实际应用领域并非如此。每一个结构化提取流程仍在底层使用词性标注和依存关系树。由大语言模型生成的 JSON 会依据语法约束进行验证。问答系统则利用依存关系分析来分解查询语句。机器翻译质量评估工具也会检查解析树的对齐情况。

值得了解的是，本课将介绍各种标签集与基准测试方法，以及何时应停止从零实现而转而使用 spaCy。

## 概念概述

**词性标注**是为每个词元分配语法类别的标记方式。**Penn Treebank（PTB）** 标签集是英语领域的默认标准，包含36个细分标签，其复杂的分类方式可能让普通读者感到繁琐：例如 `NN` 表示单数名词，`NNS` 表示复数名词，`NNP` 表示单数专有名词，`VBD` 表示动词过去式，`VBZ` 表示第三人称单数现在时，以此类推。而**通用依存关系（UD）** 标签集的粒度更粗（仅有17个标签），且与具体语言无关；它因此成为跨语言任务的默认选择。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法解析**会生成一棵语法树。主要有两种风格：

- **成分分析法**。名词短语、动词短语及介词短语相互嵌套。其输出为以词语作为叶节点的非终结符类别树（如 NP、VP、PP）。
- **依存分析法**。每个单词都有一个它所依赖的支配词，并通过语法关系进行标注。其输出为一种树结构，其中每条边都表示一个（支配词，被支配词，关系）三元组。

由于能够很好地泛化到不同语言，尤其是自由语序的语言，依存分析法在2010年代占据了优势地位。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

## 构建它

### 步骤 1：最常见标签基线

最简单的可用词性标注器。对于每个单词，预测其在训练数据中出现频率最高的词性标签。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在Brown语料库上，该基准模型的准确率约为85%。虽然不算理想，但也是任何严肃模型都不应低于的最低标准。

### 步骤 2：二元词 HMM 标注器

对序列的联合概率进行建模：

```
P(tags, words) = prod P(tag_i | tag_{i-1}) * P(word_i | tag_i)
```

两个表格：转移概率（给定前一个标签的当前标签）和发射概率（给定标签的单词）。使用拉普拉斯平滑法根据计数值估计这两个概率。通过维特比算法（在标签格上的动态规划）进行解码。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

在Brown语料集上使用二元HMM的准确率可达约93%。从85%提升到93%，主要得益于转移概率的提升——模型识别出“限定词+名词”组合较为常见，而“名词+限定词”组合则较为罕见。

### 步骤 3：为何现代标记器更胜一筹

转移概率与发射概率均为局部性质。它们无法区分“saw”在“I bought a saw”中是名词，而在“I saw the movie”中则是动词。若使用包含任意特征（词尾、单词形状、前后单词以及单词本身）的条件随机场，准确率可达约97%；而BiLSTM-CRF或Transformer模型的准确率则可达到98%以上。

该任务的性能上限取决于标注者之间的分歧程度。在Penn Treebank数据集上，人类标注者的意见一致率约为97%。准确率超过98%的模型很可能是对测试集过拟合了。

### 步骤 4：依赖关系解析概要

从头开始实现完整的依赖关系解析不在本课程的讨论范围内；相关经典教材内容可见 Jurafsky 和 Martin 的著作。需要了解的两类传统解析器如下：

- **基于转换的**解析器（arc-eager、arc-standard）的工作方式类似于移进归约解析器：它们读取标记，将其推入栈中，并执行归约操作以生成连接弧。贪心解码速度较快。经典的实现为 MaltParser；现代的神经网络版本则为 Chen 和 Manning 提出的基于转换的解析器。
- **基于图的**解析器（Eisner 算法、Dozat-Manning 双仿射算法）会为每条可能的头依赖边计算得分，然后选择最大生成树。虽然速度较慢，但精度更高。

对于大多数实际应用场景，建议使用 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

从下往上阅读 `dep` 列，句子的语法结构便会显现出来。

## 使用它

每个生产级 NLP 库都会将词性标注器和依存关系解析器作为标准流程的一部分提供。

- **spaCy**（`en_core_web_sm` / `md` / `lg` / `trf`）。速度快、准确度高，集成了分词、命名实体识别和词形还原功能。提供 `token.tag_`（Penn 标准）、`token.pos_`（UD 标准）以及 `token.dep_`（依存关系）属性。
- **Stanford NLP (stanza)**。CoreNLP 的继任者，支持 60 多种语言，在相关技术领域处于领先水平。
- **trankit**。基于 Transformer 模型，词性标注的准确度较高。
- **NLTK**。提供 `pos_tag` 函数。虽然使用方便但速度较慢且较为老旧，适合用于教学用途。

### 在2026年，这一领域为何依然重要

- **词形还原。** 第 01 课中，必须依赖词性标注才能正确进行词形还原，这是始终不变的规则。
- **从大语言模型输出中提取结构化信息。** 需要验证生成的句子是否符合语法约束（例如主谓一致、必需的修饰语等）。
- **基于方面的情感分析。** 依存关系解析能够明确指出哪些形容词修饰了哪些名词。
- **查询理解。** 如“由韦斯·安德森执导、比尔·默里主演的电影”这类查询，可通过解析分解为结构化的约束条件。
- **跨语言迁移。** 通用词性标注和依存关系是与具体语言无关的，因此能够实现对新语言的零样本结构化分析。
- **低计算开销的处理流程。** 如果无法使用Transformer模型，结合词性标注、依存关系解析以及词汇表依然可以取得相当不错的效果。

## 发布它

保存为 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 练习题

1. **简单。** 使用小型带标签语料库中最常见的标签作为基准（例如 NLTK 的 Brown 子集），在保留的句子上测量准确率，目标值为约 85%。
2. **中等难度。** 训练上述二元 HMM 模型，并报告每个标签的精确度/召回率。该 HMM 模型最容易混淆哪些标签？
3. **困难。** 利用 spaCy 的依存关系解析功能，从 1000 句样本中提取主谓宾三元组。以 50 个手动标注的三元组作为评估标准。记录解析失败的情况（通常发生在被动语态、并列结构以及主语被省略的语句中）。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|-----------------|----------|
| POS标签 | 单词的类型 | 语法类别。PTB包含36种；UD包含17种。 |
| Penn Treebank | 标准标签集 | 专为英语设计，对动词时态和名词数有细致的标注。 |
| Universal Dependencies | 多语言标签集 | 比PTB更粗略，与具体语言无关，是跨语言任务的默认选择。 |
| 依存句法分析 | 句子树结构 | 每个单词都有一个中心词，每条边都表示一种语法关系。 |
| Viterbi算法 | 动态规划方法 | 根据观测值和转移概率，找出概率最高的标签序列。 |

## 延伸阅读

- [Jurafsky 和 Martin 所著《语音与语言处理》第 8 章和第 18 章](https://web.stanford.edu/~jurafsky/slp3/)——关于词性标注与句法分析的经典教材。
- [Universal Dependencies 项目](https://universaldependencies.org/)——所有多语言句法分析器所使用的跨语言标签集与语料库集合。
- [spaCy 语言特征指南](https://spacy.io/usage/linguistic-features)——关于 `Token` 对象中所有属性的实用参考资料。
- [Chen 和 Manning (2014). 基于神经网络的高效精确依存句法分析器](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf)——将神经网络句法分析器引入主流领域的论文。
