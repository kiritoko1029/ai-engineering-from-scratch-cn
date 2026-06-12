# 概率与分布

>
> 概率是人工智能用来表达不确定性的语言。

**类型：** 学习
**语言：** Python
**先修要求：** 第一阶段，课程 01-04
**时长：** 约 75 分钟

## 学习目标

- 从零实现伯努利分布、分类分布、泊松分布、均匀分布和正态分布的PMF与PDF函数  
- 计算期望值和方差，并运用中心极限定理解释为何高斯分布占据主导地位  
- 使用数值稳定性技巧（减去最大对数几率）构建softmax函数与log-softmax函数  
- 根据对数几率计算交叉熵损失，并将其与负对数似然联系起来

## 问题所在

分类器输出 `[0.03, 0.91, 0.06]`。语言模型从50,000个候选词中挑选下一个词。扩散模型则通过从学到的概率分布中采样来生成图像。这些都属于概率在实际应用中的体现。

模型做出的每一个预测都对应一个概率分布。每个损失函数用于衡量预测分布与真实分布之间的差异程度。每一步训练都会调整参数，使其中一个分布更接近另一个分布。没有概率论，你就无法阅读任何机器学习论文、调试任何模型，也无法理解为何训练损失会呈现为NaN值。

## 概念概述

### 事件、样本空间与概率

样本空间 S 是所有可能结果构成的集合。事件则是样本空间的子集。概率将事件映射为介于 0 和 1 之间的数值。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三个公理定义了所有的概率概念：
1. 对于任意事件 A，均有 P(A) >= 0。
2. P(S) = 1（必然会发生某事）。
3. 当事件 A 和 B 不能同时发生时，有 P(A 或 B) = P(A) + P(B)。

其余内容（贝叶斯定理、期望值、分布等）均源自这三条规则。

### 条件概率与独立性

P(A|B) 表示在事件 B 已发生的条件下，事件 A 发生的概率。

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

当已知一个事件的信息对另一个事件毫无启示时，这两个事件即为独立事件：

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

抛硬币是独立事件。而不放回地抽牌则不是独立事件。

### 概率质量函数与概率密度函数

离散随机变量具有概率质量函数（PMF）。每个取值都对应一个可直接读取的特定概率。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

连续型随机变量具有概率密度函数（PDF）。某一点处的密度值并不代表概率，概率是通过对该密度在某个区间内进行积分得到的。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

在机器学习中，这一区别具有重要意义。分类任务的输出为概率质量函数（PMF），代表离散的选择；而变分自编码器的潜在空间则使用概率密度函数（PDF），表示连续值。

### 常用发行版

**伯努利分布：** 一次试验，两种结果。用于建模二元分类问题。

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**分类型：** 一次试验，k种结果。用于多类分类建模（输出为 softmax 形式）。

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**均匀分布：** 所有结果的出现概率均等。用于随机初始化。

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**正态分布（高斯分布）：** 即钟形曲线。由均值（mu）和方差（sigma^2）参数化。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**泊松分布：** 用于统计固定时间间隔内稀有事件的发生次数。用于建模事件发生率。

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### 期望值与方差

期望值即加权平均结果。

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

方差用于衡量数据围绕均值的离散程度。

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

在机器学习中，期望值表现为损失函数（即数据分布上的平均损失）。方差则反映了模型的稳定性。梯度方差较高意味着训练过程存在噪声。

### 联合分布与边际分布

联合分布 P(X, Y) 用于描述两个随机变量之间的共同行为。

联合概率质量函数示例（X = 天气，Y = 雨伞）：

| | Y=0（未带雨伞） | Y=1（带雨伞） | 边缘分布 P(X) |
|---|---|---|---|
| X=0（晴天） | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1（雨天） | 0.05 | 0.45 | P(X=1) = 0.50 |
| **边缘分布 P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

边缘分布是通过忽略另一个变量得到的：

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

上表中的行总计与列总计即为边际值。

### 为何正态分布无处不在

中心极限定理：许多独立随机变量的和（或平均值）会收敛为正态分布，且与这些随机变量本身的原始分布无关。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

原因如下：
- 测量误差近似服从正态分布（由众多独立的微小误差源构成）
- 神经网络中的权重初始化通常采用正态分布
- SGD算法中的梯度噪声也近似服从正态分布（由大量样本梯度的总和构成）
- 对于给定的均值和方差，正态分布是熵最大的概率分布

### 对数概率

原始概率值会导致数值问题。将多个较小的概率值相乘时，结果会迅速下溢为零。

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

对数概率可以解决这个问题。乘法运算将转化为加法运算。

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

规则：
- log(a * b) = log(a) + log(b)
- 对数概率始终小于等于 0（因为 0 < P <= 1）
- 数值越负，概率越低
- 交叉熵损失即为正确类别的对数概率的负值

### 作为概率分布的 Softmax 函数

神经网络输出原始分数（逻辑值）。Softmax函数将其转换为有效的概率分布。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

softmax技巧：在对数值取指数之前减去最大对数值，以避免溢出。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

对数软max通过结合软max与对数运算来提升数值稳定性。PyTorch在计算交叉熵损失时会内部使用该算法。

### 采样

采样是指从概率分布中抽取随机值。在机器学习领域：
- Dropout 会随机选择某些神经元并将其值设为零
- 数据增强则是通过随机变换来生成新数据
- 语言模型会根据预测的分布来采样下一个标记
- 扩散模型则通过采样噪声并逐步去除噪声来实现

若需从任意概率分布中采样，则需要使用逆变换采样、拒绝采样或重参数化技巧（如变分自编码器中所使用的）等技术。

```figure
gaussian-pdf
```

## 构建它

### 步骤 1：概率基础

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 步骤 2：从零构建 PMF 和 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 步骤 3：期望值与方差

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 步骤 4：从分布中采样

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 步骤 5：Softmax 及对数概率计算

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 步骤 6：中心极限定理演示

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步骤 7：可视化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

包含所有可视化内容的完整实现代码位于 `code/probability.py` 中。

## 使用它

使用 NumPy 和 SciPy，以上所有操作均可通过单行代码完成：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

您是从零开始构建这些的。现在您已经了解了库中的各个函数在做什么。

## 练习题

1. 实现指数分布的逆变换采样方法。通过抽取10,000个样本，并将生成的直方图与真实的概率密度函数进行对比，以验证该方法的正确性。

2. 为两个标准骰子的投掷结果构建联合分布表。计算各自的边际分布，并判断这两个骰子是否相互独立。

3. 计算一个5类分类器的交叉熵损失值，该分类器在正确类别的索引为3时输出对数几率 `[2.0, 0.5, -1.0, 3.0, 0.1]`。随后使用PyTorch中的 `nn.CrossEntropyLoss` 函数来验证计算结果。

4. 编写一个函数，该函数接收一组对数概率值，返回最可能的序列、总对数概率以及对应的原始概率值。使用一个包含50个单词的句子进行测试，假设每个单词的概率均为0.01。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 样本空间 | “所有可能的情况” | 实验中每种可能结果的集合 S |
| PMF | “概率函数” | 给出每个离散结果精确概率的函数，其值之和为 1 |
| PDF | “概率曲线” | 连续变量的密度函数。通过对区间积分可得到概率 |
| 条件概率 | “在已知某条件的情况下的概率” | P(A\|B) = P(A 和 B) / P(B)。这是贝叶斯思维与贝叶斯定理的基础 |
| 独立性 | “它们互不影响” | P(A 和 B) = P(A) * P(B)。知道一个事件无法推断另一个事件的信息 |
| 期望值 | “平均值” | 所有结果按概率加权的总和。损失函数即为期望值 |
| 方差 | “数据的离散程度” | 值与均值的平方偏差的期望值。方差较高意味着估计结果噪声大且不稳定 |
| 正态分布 | “钟形曲线” | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2))。由于中心极限定理，该分布随处可见 |
| 中心极限定理 | “平均值趋近于正态分布” | 无论来源如何，多个独立样本的均值都会收敛为正态分布 |
| 联合分布 | “两个变量一起的概率” | P(X, Y) 描述了 X 和 Y 的每组结果组合出现的概率 |
| 边缘分布 | “忽略另一个变量后的分布” | P(X) = sum_y P(X, Y)。从联合分布中可得到某个变量的分布 |
| 对数概率 | “概率的对数” | log P(x)。它将乘积运算转化为求和运算，从而避免长序列中的数值下溢问题 |
| Softmax | “将分数转换为概率” | softmax(z_i) = exp(z_i) / sum(exp(z_j))。它将实数值的逻辑斯蒂得分映射为有效的概率分布 |
| 交叉熵 | “损失函数” | -sum(p_true * log(p_predicted))。用于衡量两个概率分布之间的差异，值越小越好 |
| 逻辑斯蒂得分 | “模型的原始输出” | 在应用 Softmax 之前的未标准化分数。其名称源自逻辑斯蒂函数 |
| 抽样 | “随机抽取数值” | 根据概率分布生成数值。这也是模型产生输出的方式 |

## 延伸阅读

- [3Blue1Brown：什么是中心极限定理？](https://www.youtube.com/watch?v=zeJD6dqJ5lo) —— 用可视化方式解释平均值为何会呈现正态分布  
- [斯坦福 CS229 概率论复习资料](https://cs229.stanford.edu/section/cs229-prob.pdf) —— 涵盖本课程内容及更多知识的简明参考文档  
- [对数和指数技巧](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) —— 说明数值稳定性为何重要以及如何实现它
