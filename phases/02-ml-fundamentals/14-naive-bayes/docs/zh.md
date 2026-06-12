# Naive Bayes

> “天真的”假设是错误的，但它仍然有效。这就是它的魅力所在。

**类型：**构建
**语言：**Python
**先决条件：**第二阶段，课程01-07（分类，贝叶斯定理）
**时间：**约75分钟

## 学习目标

- Implement Multinomial Naive Bayes from scratch using Laplace smoothing for text classification
- Explain why the naive independence assumption is mathematically incorrect but produces correct class rankings in practice
- Compare Multinomial, Bernoulli, and Gaussian Naive Bayes variants and select the right one for a given feature type
- Evaluate Naive Bayes against logistic regression on high-dimensional sparse data and explain the bias-variance tradeoff at work

## 问题

You need to classify text. Emails into spam or non-spam. Customer reviews into positive or negative. Support tickets into categories. You have thousands of features (one per word) and limited training data.

Most classifiers struggle here. Logistic regression requires sufficient samples to reliably estimate thousands of weights. Decision trees split on individual words, leading to overfitting. KNN, with 10,000 dimensions, is meaningless because every point is equally distant from all other points.

Naive Bayes handles this situation. It makes a mathematically incorrect assumption—that each feature is independent of all others given the class—yet it still outperforms “more advanced” models in text classification, especially with small training sets. It trains in a single pass through the data and can handle millions of features. It produces probability estimates (though often poorly calibrated due to the independence assumption).

Understanding why a wrong assumption leads to good predictions teaches us something fundamental about machine learning: the best model is not the one that is most correct, but rather the one with the best balance between bias and variance for your specific data.

## 概念

### 贝叶斯定理（快速回顾）

贝叶斯定理改变了条件概率：

```
P(class | features) = P(features | class) * P(class) / P(features)
```

我们希望得到 `P(class | features)`——即给定文档中的词汇时，该文档属于某个类别的概率。我们可以通过以下因素来计算这一概率：
- `P(features | class)`——在该类别的文档中看到这些词汇的可能性
- `P(class)`——该类别的先验概率（垃圾邮件通常有多常见？）
- `P(features)`——所有类别共有的证据，因此在比较时可以忽略它

具有最高 `P(class | features)`值的类别获胜。

### 简单的独立假设

计算 `P(features | class)` 实际上需要估算所有特征共同出现的联合概率。如果词汇量为10,000个单词，那么就需要估算2^10,000种可能组合的分布情况。这是不可能的。

简单的假设是：每个特征在给定类别的情况下都是条件独立的。

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Instead of using one impossible joint distribution, you estimate n separate per-feature distributions. Each distribution only requires a count.

This assumption is obviously incorrect. The terms "machine" and "learning" are not independent in any document. However, the classifier does not need accurate probability estimates. It needs correct rankings—which class has the highest probability. The independence assumption introduces systematic errors, but these errors affect all classes equally, so the ranking remains correct.

### 为什么它仍然有效

Three reasons:

1. **Ranking over calibration.** Classification only needs the top-ranked class to be correct. Even if P(spam) = 0.99999 when the true probability is 0.7, the classifier still picks spam correctly. We don’t need correct probabilities; we need the correct winner.

2. **High bias, low variance.** The independence assumption is a strong prior. It restricts the model heavily, preventing overfitting. With limited training data, a model that is slightly wrong but stable beats a model that is theoretically right but wildly unstable. This is the bias-variance tradeoff in action.

3. **Feature redundancy cancels out.** Correlated features provide redundant evidence. The classifier double-counts this evidence, but it does so for the correct class as well. If “machine” and “learning” always appear together, both provide evidence for the “tech” class. Naive Bayes counts these twice, but it does so for the right class too.

A fourth, practical reason: Naive Bayes is extremely fast. Training involves a single pass through the data to count frequencies. Prediction is a simple matrix multiplication. You can train on a million documents in seconds. This speed allows for faster iteration, more feature testing, and more experimental runs compared to slower models.

### 数学逐步讲解

Let us trace through a concrete example. Suppose we have two classes: spam and not-spam. Our vocabulary has three words: "free", "money", "meeting".

Training data:
- Spam emails mention "free" 80 times, "money" 60 times, "meeting" 10 times (150 total words)
- Not-spam emails mention "free" 5 times, "money" 10 times, "meeting" 100 times (115 total words)
- 40% of emails are spam, 60% are not-spam

With Laplace smoothing (alpha=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

新邮件包含：“免费”（2次），“钱”（1次），“会议”（0次）。

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam wins by a large margin. The appearance of the word “free” twice is strong evidence for spam. Note that the absence of “meeting” contributes zero to both log sums (0 * log(P)) – in Multinomial NB, absent words have no effect. It is Bernoulli NB that explicitly models word absence.

### 三种变体

Naive Bayes has three variants. Each variant models `P(feature | class)` differently.

#### 多项式朴素贝叶斯

Each feature is represented as a count. It is best used for text data where features are word frequencies or TF-IDF values.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha`是拉普拉斯平滑（下文将解释）。这种变体是文本分类的主要工具。

#### 高斯朴素贝叶斯

Each feature is modeled as a normal distribution. Suitable for continuous features.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

每个类别都有自己对应的每个特征的均值和方差。当各个特征在每类内确实遵循钟形曲线时，这种方法效果很好。

#### Bernoulli Naive Bayes

Each feature is represented as a binary value: present or absent. Suitable for short texts or binary feature vectors.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

与多项式模型不同，伯努利模型会明确惩罚单词的缺失。如果“free”通常出现在垃圾邮件中，但在这封电子邮件中缺失，伯努利模型会将此视为反对垃圾邮件的证据。

### 何时使用每种变体

| 变体 | 特征类型 | 适用场景 | 示例 |
|------|----------|----------|---------|
| 多项式 | 计数或频率 | 文本分类、词袋模型 | 电子邮件垃圾邮件、主题分类 |
| 高斯 | 连续值 | 具有近似正态分布特征的表格数据 | Iris分类、传感器数据 |
| 伯努利 | 二进制（0/1） | 短文本、二进制特征向量 | SMS垃圾邮件、存在/不存在特征 |

### 拉普拉斯平滑

当某个单词出现在测试数据中但从未出现在特定类别的训练数据中时会发生什么？

没有平滑处理：`P(word | class) = 0/N = 0`。一个零乘以整个乘积使得`P(class | features) = 0`，无论其他所有证据如何。一个未见过的单词会破坏整个预测结果，无论其他证据有多支持它。

拉普拉斯平滑会在每个特征计数中加入一个小值`alpha`（通常为1）：

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

当alpha值为1时，每个单词至少具有微小的概率。在测试邮件中出现的“discombobulate”一词不再降低垃圾邮件的概率。平滑处理具有贝叶斯解释：它相当于在单词分布上施加均匀狄利克雷先验。

更高的alpha值意味着更强的平滑处理（更均匀的分布）。较低的alpha值意味着模型对数据的信任度更高。alpha是一个需要调整的超参数。

alpha值的效应：

| Alpha | 效应 | 使用时机 |
|-------|--------|-------------|
| 0.001 | 几乎无平滑处理，完全信任数据 | 训练集非常大，预期没有未见特征 |
| 0.1 | 轻度平滑处理 | 训练集较大 |
| 1.0 | 标准拉普拉斯平滑处理 | 默认起始点 |
| 10.0 | 重度平滑处理，使分布平坦 | 训练集非常小，预期有许多未见特征 |

### Log-Space Computation

将数百个概率（每个都小于1）相乘会导致浮点数的下溢。即使真实值是一个非常小的正数，乘积在浮点数中也会变为零。

解决方案：在对数空间中工作。而不是相乘概率，而是加上它们的对数：

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

这将预测结果转换为点积形式：

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

矩阵乘法。这就是朴素贝叶斯预测如此快速的原因——它与单层线性模型的操作相同。

### Naive Bayes vs Logistic Regression

两者都是文本线性分类器。区别在于它们所建模的内容。

| 方面 | 朴素贝叶斯 | 逻辑回归 |
|------|------------|----------|
| 类型 | 生成式（建模 P(X\|Y)） | 判别式（建模 P(Y\|X)） |
| 训练方法 | 计算频率 | 优化损失函数 |
| 小数据 | 更好（强大的先验有助于预测） | 较差（不足以估计权重） |
| 大数据 | 较差（错误的假设会带来问题） | 更好（灵活的边界） |
| 特征处理 | 假设独立性 | 处理相关性 |
| 速度 | 单次遍历，非常快 | 迭代优化 |
| 校准 | 概率较低 | 概率较好 |

经验法则：开始时使用朴素贝叶斯。如果你有足够的数据且朴素贝叶斯模型达到平台期，则改用逻辑回归。

### 分类管道

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

在实践中，我们采用对数空间来避免浮点数的下溢问题。而不是乘以许多小概率，我们添加它们的对数：

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

```figure
naive-bayes
```

## 构建它

`code/naive_bayes.py`中的代码从零开始实现了MultinomialNB和GaussianNB。

### MultinomialNB

从零开始的实现：

1. **fit(X, y)**：对于每个类别，统计每个特征的频率。添加拉普拉斯平滑处理。计算对数概率。存储类别先验（类别频率的对数）。

2. **predict_log_proba(X)**：对于每个样本，计算log P(class)加上所有类别的log P(feature_i | class）之和。这是一个矩阵乘法：X @ log_probs.T + log_priors。

3. **predict(X)**：返回具有最高对数概率的类别。

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

关键见解：在拟合之后，预测只是矩阵乘法加上一个偏置。这就是朴素贝叶斯算法如此快速的原因。

### GaussianNB

对于连续特征，我们估算每个类别每个特征的均值和方差：

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

预测使用每个特征的高斯概率密度函数，将这些概率密度函数相乘（在对数空间中相加）。

### Demo: Text Classification

该代码生成了模拟两个类别（科技文章与体育文章）的合成词袋数据。每个类别都有不同的词频分布。MultinomialNB类使用词计数对这些数据进行分类。

合成数据的生成方式如下：我们创建200个“词”（特征列）。第0-39位的词在科技文章中频率较高，在体育文章中频率较低。第80-119位的词在体育文章中频率较高，在科技文章中频率较低。第40-79位的词在两者中的频率均为中等。这样就能创建一个真实的场景，其中一些词是明显的类别指示符，而另一些则是噪声。

### 演示：持续功能

该代码生成了类似Iris的数据（3类，4特征，高斯分布聚类）。GaussianNB使用每类的均值和方差进行分类。每个类别都有不同的中心（均值向量）和不同的分散度（方差），模拟了现实世界中不同类别之间测量值系统性差异的情况。

代码还展示了：
- **平滑效果比较：**使用不同的alpha值训练MultinomialNB，以展示平滑强度对准确性的影响。
- **训练样本量实验：**当训练数据从20个样本增加到1600个样本时，NB的准确率如何提高。即使样本数量很少，NB也能达到不错的准确率——这是其主要优势。
- **混淆矩阵：**每类的精确度、召回率和F1分数，以显示NB的错误所在。

### 预测速度

Naive Bayes预测是一个矩阵乘法。对于n个样本，每个样本有d个特征，共有k个类别：
- 多项式朴素贝叶斯：一个矩阵乘法（n x d）@（d x k）= O(n * d * k）
- 高斯朴素贝叶斯：n * k次高斯概率密度函数计算，每次针对d个特征 = O(n * d * k)

这两种方法在每个维度上都是线性的。与KNN（需要计算所有训练点之间的距离）或带有RBF核的SVM（需要对所有支持向量进行核计算）相比，Naive Bayes在预测时快几个数量级。

## 使用它

使用sklearn时，两种变体都是简短的语句：

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

用于使用 sklearn 进行文本分类：

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

The code in `naive_bayes.py` compares newly implemented algorithms with those from sklearn using the same data to verify their correctness.

### TF-IDF与朴素贝叶斯

原始单词计数使得每个单词每次出现都具有相同的权重。但是像“the”和“is”这样的常见词汇在每个类别中都会频繁出现——它们不携带任何信息。TF-IDF（词频-逆文档频率）则降低常见词汇的权重，并提升罕见且具有区分性的词汇的权重。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF值是非负的，因此它可以与MultinomialNB一起使用。TF-IDF与MultinomialNB的结合是文本分类中最强大的基线方法之一。在训练样本少于10,000的数据集上，它通常能胜过更复杂的模型。

### BernoulliNB用于短文本

对于短文本（推文、短信、聊天消息），BernoulliNB的表现优于MultinomialNB。短文本的词数较少，因此MultinomialNB依赖的频率信息存在噪声。BernoulliNB只关心词的存在或缺失，这对于短文本来说更为可靠。

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

在CountVectorizer中，`binary=True`标志将所有计数转换为0/1。没有这个标志，BernoulliNB仍然可以工作，但会处理它并不适合处理的计数。

### 校准NB概率

NB模型的概率校准不佳。当NB表示P(垃圾邮件)为0.95时，真实概率可能是0.7。如果你需要可靠的概率估计（例如，设定阈值或与其他模型结合使用），请使用sklearn的CalibratedClassifierCV：

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

This method involves applying logistic regression on the raw scores of NB using cross-validation. The resulting probabilities are much closer to the true class frequencies.

### 常见陷阱

1. **Negative feature values.** MultinomialNB requires non-negative features. If you have negative values (like TF-IDF with certain settings or standardized features), use GaussianNB instead, or shift the features to be positive.

2. **Zero variance features.** GaussianNB divides by variance. If a feature has zero variance for a class (all values identical), the probability computation breaks. The code adds a small smoothing term (1e-9) to all variances to prevent this.

3. **Class imbalance.** If 99% of emails are not-spam, the prior P(not-spam) = 0.99 is so strong that it overwhelms the likelihood evidence. You can set class priors manually or use the class_prior parameter in sklearn.

4. **Feature scaling.** MultinomialNB does not need scaling (it works on counts). GaussianNB also does not need scaling (it estimates per-feature statistics). This is an advantage over logistic regression and SVM, which are sensitive to feature scales.

## 发货

本课程将生成以下文件：
- `outputs/skill-naive-bayes-chooser.md` -- 用于选择正确NB变体的决策技能文档
- `code/naive_bayes.py` -- 从零开始实现MultinomialNB和GaussianNB，并对比sklearn的实现

### 当朴素贝叶斯失败时

当独立性假设导致错误排序时（不仅仅是概率错误），NB模型会失败。这种情况发生在以下情况：

1. **强烈的特征交互。**如果类别依赖于两个特征的组合，而不是任何一个单独的特征（类似异或模式），NB将完全忽略这一特征。每个特征单独来看都无法提供证据，而NB也无法非线性地结合这些特征。

2. **高度相关的特征带来相反的证据。**如果特征A表示“垃圾邮件”，特征B表示“非垃圾邮件”，但A和B之间存在完美相关性（实际上它们总是一致），那么NB会在没有冲突证据的情况下看到冲突。

3. **非常大的训练集。**当数据足够多时，像逻辑回归这样的判别模型会学习到真实的决策边界并优于NB。原本有助于小数据的独立性假设现在反而限制了模型的性能。

在实践中，这些失败模式在文本分类中很少见。文本特征数量众多，单个特征的效果较弱，且独立性假设的误差往往相互抵消。对于具有少量高度相关特征的表格数据，首先考虑使用逻辑回归或基于树的模型。

## 练习

1. **Smoothing experiment.** Train the MultinomialNB algorithm on text data with alpha values of 0.01, 0.1, 1.0, 10.0, and 100. Plot the accuracy against these alpha values. Where does the performance peak? Why does a very high alpha value negatively affect performance?

2. **Feature independence test.** Use a real text dataset. Select two words that are clearly correlated, such as “machine” and “learning”. Calculate P(word1 | class) * P(word2 | class) and compare it to P(word1 AND word2 | class). How accurate is the assumption of feature independence? Does it affect classification accuracy?

3. **Bernoulli implementation.** Extend the code by adding a BernoulliNB class. Convert the bag-of-words data to binary (present/absent) values and compare the accuracy with that of MultinomialNB on the same text data. When does the Bernoulli algorithm outperform MultinomialNB?

4. **NB vs Logistic Regression.** Train both algorithms on text data. Start with 100 training samples and increase to 10,000. Plot the accuracy against the size of the training set for both algorithms. At what point does Logistic Regression surpass Naive Bayes in performance?

5. **Spam filter.** Build a complete spam classification system: tokenize raw email texts, create a vocabulary, generate bag-of-words features, train MultinomialNB, and evaluate the performance using precision and recall (not just accuracy—why?).

## | 关键词 | 翻译 |

| 术语 | 人们怎么说 | 实际含义 |
|------|------------|----------|
| Naive Bayes | “简单的概率分类器” | 一种应用贝叶斯定理的分类器，假设特征在给定类别的情况下是条件独立的 |
| 条件独立 | “特征之间互不影响” | P(A, B | C) = P(A | C) * P(B | C) -- 一旦知道C，知道B并不会提供关于A的新信息 |
| Laplace平滑 | “加一平滑” | 对每个特征添加少量计数，以防止零概率主导预测结果 |
| 先验 | “看到数据之前所相信的” | P(类别) -- 在观察任何特征之前的每个类别的概率 |
| 似然 | “数据拟合程度” | P(特征 | 类别) -- 如果已知类别，观察到这些特征的概率 |
| 后验 | “看到数据之后的信念” | P(类别 | 特征) -- 观察特征后的类别更新概率 |
| 生成模型 | “模拟数据的生成方式” | 一种学习P(X | Y)和P(Y)的模型，然后使用贝叶斯定理得到P(Y | X) |
| 判别模型 | “模拟决策边界” | 一种直接学习P(Y | X)的模型，而不模拟X的生成方式 |
| 对数概率 | “避免下溢” | 使用对数P而不是P来防止许多小数字的乘积在浮点数中变为零 |

## 更多阅读资料

- [scikit-learn Naive Bayes文档](https://scikit-learn.org/stable/modules/naive_bayes.html) -- 三种变体的数学细节
- [McCallum和Nigam，Naive Bayes文本分类的事件模型比较（1998年）](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) -- 多项分布与伯努利分布的经典文本分类比较
- [Rennie等人，解决Naive Bayes文本分类的不良假设（2003年）](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) -- 对文本分类的NB算法的改进
- [Ng和Jordan，关于判别式与生成式分类器（2001年）](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) -- 证明NB算法在更少数据的情况下收敛速度比LR更快
