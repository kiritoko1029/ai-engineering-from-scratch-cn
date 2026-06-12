# 文本处理——分词、词干提取与词形还原

> 语言是连续的，模型则是离散的。预处理便是连接两者的桥梁。

**类型：** 构建
**语言：** Python
**先修要求：** 第2阶段 · 14（朴素贝叶斯）
**耗时：** 约45分钟

## 问题所在

模型无法理解“猫们在奔跑。”这类句子，它只能识别整数。

每一个自然语言处理系统在启动时都会面临同样的三个问题：单词的起始位置在哪里？单词的词根是什么？为何在某些情况下要将“run”、“running”、“ran”视为相同的内容，而在其他情况下又视为不同的内容？

如果分词处理出错，模型就会从错误的数据中学习。如果你的分词器将`don't`视为一个标记，而将`do n't`视为两个标记，那么训练数据的分布就会发生分裂。如果词干提取器将`organization`和`organ`提取为相同的词干，主题建模功能就会失效。如果词形还原器需要词性上下文，但你没有提供该上下文，动词就会被错误地当作名词处理。

本课将从零开始构建这三个预处理步骤，随后展示NLTK和spaCy是如何实现相同功能的，以便你了解它们之间的权衡关系。

## 概念概述

三种操作。每种操作都有其具体任务及相应的故障模式。

**分词（Tokenization）**：将字符串拆分为多个标记。“标记”的定义较为模糊，因为合适的粒度取决于具体任务。在传统自然语言处理中通常为单词级；在Transformer模型中则为子词级；对于没有空格的语言，则为字符级。

**词干提取（Stemming）**：通过规则去除词尾后缀。该方式速度快、效率高，但较为简单粗暴。例如将 `running` 转换为 `run`，将 `organization` 转换为 `organ`。后者即为该操作的故障模式。

**词形还原（Lemmatization）**：利用语法知识将单词还原为其词典中的标准形式。该方式速度较慢但准确性更高，需要依赖查找表或形态分析器。例如将 `ran` 还原为 `run`（需知晓“ran”是“run”的过去式），将 `better` 还原为 `good`（需知晓比较级形式）。

经验法则：当速度优先且能容忍一定误差时（如搜索索引、粗略分类），使用词干提取；当语义准确性至关重要时（如问答系统、语义搜索，以及任何用户会阅读的内容），则应使用词形还原。

```figure
edit-distance
```

## 构建它

### 步骤 1：正则表达式单词分词器

最简单实用的分词器会在遇到非字母数字字符时进行分割，同时将标点符号视为独立的标记。虽然并不完美，也不是最终方案，但它仅需一行代码即可实现。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

按优先级排列的三种模式。包含可选内部单引号的单词（如“don’t”、“it’s”）。纯数字。任何单个的非空白、非字母数字字符均作为独立标记处理（即标点符号）。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

需注意的错误模式。由于字符序列与数字序列交替出现，`3pm` 会被拆分为 `['3', 'pm']`。对于大多数任务而言这已足够使用，但 URL、电子邮件和标签都会出错。在通用模式之前添加特定模式，方可用于生产环境。

### 步骤 2：Porter 去词干器（仅适用于步骤 1a）

完整的Porter算法包含五个阶段的规则。仅第一步a就涵盖了最常见的英语后缀，并用于教授相关模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

请自上而下地阅读这些规则。正是“ies -> i”这条规则导致了“ponies”被转换为“poni”，而非“pony”。真正的Porter算法包含步骤1b，可以解决这个问题。各规则之间存在竞争关系，先出现的规则会优先生效。规则的排列顺序比任何单一规则都更为重要。

### 步骤 3：基于查表的词形还原器

正确的词形还原需要依赖形态学知识。一个易于教学的版本会使用一个简小的词元表，并设置相应的备用方案。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最后一个案例是重要的教学环节。`watched` 并不在我们的表中，而我们现有的备用方案仅能处理 `ing` 形式。真正的词形还原功能能够涵盖 `ed` 后缀、不规则动词、比较级形容词以及发生音变的多数形式（如 `children -> child`）。正因如此，生产环境中的系统才会使用 WordNet、spaCy 的形态分析器或完整的形态学分析工具。

### 步骤 4：将它们通过管道连接起来

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺失的组件是一个词性标注器。第 5 阶段·07（词性标注）将用于构建该工具。目前，将所有内容的默认词性设置为 `NOUN`，并承认这一局限性。

## 使用它

NLTK 和 spaCy 均提供生产环境版本，每款工具仅几行代码。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 能够处理缩写词、Unicode 字符以及正则表达式无法处理的边缘情况。`PorterStemmer` 会执行全部五个处理阶段。而 `WordNetLemmatizer` 则需要将 NLTK 的 Penn Treebank 标注体系中的词性标签转换为 WordNet 所使用的缩写集。上述的转换逻辑正是大多数教程都会忽略的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 将整个处理流程封装在 `nlp(text)` 函数中。分词、词性标注以及词形还原等操作均在其中完成。与 NLTK 相比，其在大规模数据处理时速度更快，且开箱即用时的准确度也更高。但相应的代价是用户无法轻易地单独替换其中的各个组件。

### 何时选择哪种

| 场景 | 推荐工具 |
|-----------|----------|
| 教学、研究，或需要替换组件时 | NLTK |
| 生产环境、多语言处理且对速度有要求时 | spaCy |
| 使用 Transformer 流水线（反正也会使用模型自带的分词器） | 使用 `tokenizers` / `transformers`，跳过传统的预处理步骤 |

### 没人会提醒你的两种故障模式

大多数教程仅讲解算法本身，便止步不前。在实际的预处理流程中，会有两个问题出现，而这些几乎从未被提及。

**可复现性漂移。** NLTK 和 spaCy 在不同版本之间会改变分词及词形还原的行为。在 spaCy 2.x 版本中能生成 `['do', "n't"]` 的结果，在 3.x 版本中可能会变为 `["don't"]`。你的模型是基于某一版本的分布进行训练的，而推理时却使用的是另一版本的数据。准确率会悄然下降，但没人知道原因所在。应在 `requirements.txt` 中锁定库的版本号。编写一个预处理回归测试，规定 20 条示例句子的分词结果必须保持不变，并在每次升级后运行该测试。

**训练与推理不匹配。** 使用强烈的预处理手段（如转小写、删除停用词、词干提取）进行训练，却在部署时直接使用原始用户输入，从而导致性能急剧下降。这是生产环境中最常见的自然语言处理故障。如果在训练阶段进行了预处理，那么在推理阶段也必须运行完全相同的函数。应将预处理逻辑作为模型包内的函数来提供，而非以服务团队需要重新编写的 Jupyter 单元格形式存在。

## 发布它

一个可复用的提示词，帮助工程师无需阅读三本教材即可选择预处理策略。

请保存为 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 练习题

1. **简单。** 扩展 `tokenize` 函数，将 URL 作为单个标记保留。测试：`tokenize("Visit https://example.com today.")` 应生成一个 URL 标记。
2. **中等难度。** 实现 Porter 分词法的第一步 b。如果某个单词包含元音且以 `ed` 或 `ing` 结尾，则将其移除。同时需处理双辅音规则（例如将 `hopping` 分词为 `hop`，而非 `hopp`）。
3. **高难度。** 构建一个词形还原器，该还原器以 WordNet 作为查询表，但在 WordNet 中找不到对应条目时则回退到自定义的 Porter 茎化器。需在带标注的语料库上，分别对比基于纯 WordNet 和纯 Porter 分词法的准确率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Token | 单词 | 模型所处理的任意单位。可以是单词、子词、字符或字节。 |
| Stem | 单词的词根 | 通过规则去除后缀得到的结果。不一定是真实存在的单词。 |
| Lemma | 字典形式 | 可用于查询的形态。需要语法上下文才能准确计算。 |
| POS tag | 词性标注 | 如 NOUN、VERB、ADJ 等类别。准确进行词形还原需要该信息。 |
| Morphology | 单词形态规则 | 单词根据时态、数、格等发生形式变化的方式。词形还原依赖于这些规则。 |

## 延伸阅读

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始论文，共五页，仍是目前最清晰的解释。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真实处理流程的构建方式。
- [NLTK书籍，第3章](https://www.nltk.org/book/ch03.html) — 你尚未想到的分词边界情况。
