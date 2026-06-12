# 情感分析

> 最具代表性的自然语言处理任务。关于传统文本分类的所有核心知识几乎都体现在此。

**类型：** 构建
**编程语言：** Python
**前置要求：** 第5阶段·02（词袋模型 + TF-IDF），第2阶段·14（朴素贝叶斯）
**耗时：** 约75分钟

## 问题所在

“这食物不太好。”这是正面情感还是负面情感？

情感分析看似简单，评论者只是表达了对某事物的喜爱或不满。请为该句子标注情感类型。它之所以成为经典的自然语言处理任务，是因为每一个看似简单的案例背后都隐藏着复杂问题：否定词会改变原意，讽刺语气则会彻底颠覆其含义；尽管包含两个负面词汇，“还不错”依然属于正面评价。表情符号往往比周围的文字传递出更强的信号，领域专用词汇也很重要（例如音乐评论中的“tight”与时尚评论中的“tight”含义截然不同）。

情感分析是传统自然语言处理领域的实践实验室。如果你理解了为何每一个简单的基准模型都有特定的失效模式，就能明白为何需要开发更复杂的模型。本课程将从零开始构建朴素贝叶斯基准模型，再加入逻辑回归，并指出那些导致实际应用中的情感分析工作面临合规性挑战的陷阱。

## 概念概述

传统的情感分析方法遵循两步流程。

1. **表示。** 将文本转换为特征向量，常用方法包括词袋模型、TF-IDF或n-gram模型。
2. **分类。** 基于带标签的样本训练线性模型（如朴素贝叶斯、逻辑回归、SVM）。

朴素贝叶斯是一种“最简单却有效”的模型。它假设在给定标签的前提下，所有特征都是相互独立的。通过统计频率来估计`P(word | positive)`和`P(word | negative)`的概率值，在推理时则将这些概率相乘。虽然这种“天真”的独立性假设在实际中是严重错误的，但其预测效果却出奇地好。原因在于：在文本特征较为稀疏且数据量适中的情况下，分类器更关注每个词倾向于哪一侧，而非其具体的权重大小。

逻辑回归则解决了独立性假设的问题。它会为每个特征学习一个权重，包括负权重。例如，“not good”作为二元组特征就会获得一个负权重。而朴素贝叶斯则无法对那些它未曾标注过的二元组应用此类负权重处理。

```figure
sentiment-logits
```

## 构建它

### 步骤 1：一个真实的微型数据集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

刻意保持较小规模。实际应用中会使用数以万计的示例（如 IMDb、SST-2、Yelp 极性数据集）。其数学原理是完全相同的。

### 步骤 2：从零实现多项式朴素贝叶斯

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

加性平滑（alpha=1.0）即拉普拉斯平滑。若不使用该方法，类别中未出现过的单词概率将为零，从而导致对数值发散。实际应用中通常采用 alpha=0.01 的参数值，而教学演示的默认值为 alpha=1.0。

### 步骤 3：从零实现逻辑回归

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

此处，L2 正则化非常重要。文本特征具有稀疏性；若不使用 L2，模型会记住训练样本。建议从 `0.01` 开始调整该参数值。

### 第 4 步：处理否定形式（故障模式）

考虑“不好”和“不坏”这两种情况。基于词袋的分类器会看到 `{not, good}` 和 `{not, bad}`，并根据训练数据中出现频率较高的类别进行学习。而二元词组分类器则会将 `not_good` 和 `not_bad` 视为不同的特征来学习。通常这样就已经足够了。

在没有二元词组可用时，有一种较为粗略但有效的解决方法：**否定作用域限定**。即给否定词之后的标记加上 `NOT_` 前缀，直至遇到下一个标点符号为止。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

现在，“good”与“NOT_good”是不同的特征。分类器可以给它们设置相反的权重。经过三步预处理后，在情感分析基准测试中的准确率会有显著提升。

### 第 5 步：重要的评估指标

如果类别分布不均衡，仅凭准确率会具有误导性。真实的情绪分析语料库通常有70-80%为正面情感或70-80%为负面情感；使用单纯多数类分类器时虽然能获得80%的准确率，但实际上毫无价值。需报告以下各项指标：

- **每类的精确率和召回率**：每个类别各提供一组数值。通过宏观平均法计算出一个能够反映类别平衡情况的单一数值。
- **宏观F1值（处理不平衡数据的主要指标）**：对各类别的F1分数取平均值，且各类别权重相等。在类别分布不均衡时，应使用该指标而非准确率。
- **加权F1值（另一种选择）**：与宏观F1类似，但会根据类别出现频率进行加权。当类别不平衡本身具有业务意义时，需与宏观F1值一并报告。
- **混淆矩阵**：以原始计数形式呈现。在依赖任何标量指标之前务必先查看混淆矩阵，因为它能显示模型会将哪些类别对混淆在一起。
- **每类的错误样本**：从每个类别中选取5个错误预测结果并仔细阅读。没有任何方法能替代直接查看实际错误案例。

对于严重不平衡的数据（类别比例超过95:5），应报告**AUROC**和**AUPRC**指标而非准确率。AUPRC对少数类更为敏感，而这正是通常需要关注的方面（如垃圾邮件、欺诈行为、罕见情绪类型）。

**需避免的常见错误**：在处理不平衡数据时若使用微观F1值而非宏观F1值，由于多数类的影响占主导地位，所得数值会显得很高。而宏观F1值能确保人们看到少数类的实际表现情况。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 使用它

scikit-learn 只用六行代码就能正确完成该任务。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

有三点需要注意。`stop_words=None` 可保留否定词。`ngram_range=(1, 2)` 会生成二元语法单元，从而使 `not_good` 成为特征之一。`sublinear_tf=True` 能减弱重复单词的影响。正是这三个参数的差异，使得在 SST-2 数据集上的基准准确率从 75% 提升到了 85%。

### 何时选择使用 Transformer 模型

- 讽刺语检测。传统模型在此类任务上表现不佳，事实如此。
- 长篇评论中情感在文档中途发生变化的情况。
- 基于方面的情绪分析。“相机很棒，但电池很差。”需要将情绪归属于具体的方面。仅支持 Transformer 或结构化输出模型。
- 非英语及资源匮乏的语言。多语言 BERT 能够免费提供零样本的基准方案。

如果需要上述任何功能，请直接跳至第 7 阶段（Transformer 深入学习）。否则，基于 TF-IDF、二元词组以及否定词处理技术的朴素贝叶斯或逻辑回归即可作为 2026 年的实用基准方案。

### 可复现性陷阱（再度来袭）

重新训练情感分析模型是常规操作，但重新评估它们则并非如此。论文中报告的准确率数据是基于特定的数据划分、预处理步骤以及分词器得出的。如果在不使用完全相同的处理流程的情况下将新模型与基准模型进行比较，所得到的差异值将会具有误导性。务必始终依据自身的处理流程来重新生成基准模型，而非直接套用论文中的数值。

## 发布它

保存为 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 练习题

1. **简单。** 在 scikit-learn 的流水线中加入 `apply_negation` 作为预处理步骤，并在小型情感分析数据集上测量 F1 分数的变化量。
2. **中等难度。** 实现带类别加权的逻辑回归（向 scikit-learn 传递参数 `class_weight="balanced"`，或自行推导梯度）。在人为构造的 90-10 比例的类别不平衡数据集上衡量其效果。
3. **高难度。** 通过在该情感分析模型的残差上训练第二个分类器来构建讽刺语检测器。需详细记录实验设置。当模型的准确率低于随机水平时，应向读者发出警告（二分类讽刺语任务的随机准确率约为 50%，大多数初次尝试的模型都会达到这一水平）。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 极性 | 正面或负面 | 二进制标签；有时会扩展为中性或更细粒度的分类（如5星评分）。 |
| 基于方面的情感分析 | 每个方面对应的极性 | 将情感赋予文本中提到的特定实体或属性。 |
| 否定作用域处理 | 反转邻近标记的极性 | 在“not”之后的标记前添加 `NOT_`，直至遇到标点符号为止。 |
| 拉普拉斯平滑处理 | 为计数值加1 | 防止朴素贝叶斯模型中出现概率为零的特征。 |
| L2正则化 | 缩小权重值 | 在损失函数中加入 `lambda * sum(w^2)` 的项。对于稀疏文本特征而言十分重要。 |

## 延伸阅读

- [Pang 和 Lee (2008). 意见挖掘与情感分析](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) —— 该领域的奠基性综述。篇幅较长，但前四节已涵盖了所有传统方法。
- [Wang 和 Manning (2012). 基准模型与二元词组：简单高效的情感分析与主题分类](https://aclanthology.org/P12-2018/) —— 该论文证明了在短文本处理中，二元词组结合朴素贝叶斯的方法难以被超越。
- [scikit-learn 文本特征提取文档](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) —— `CountVectorizer`、`TfidfVectorizer` 及所有需调整的参数的参考资料。
