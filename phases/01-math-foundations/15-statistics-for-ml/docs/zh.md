# 机器学习统计

统计是判断模型是否真正有效还是只是运气好的方法。

**类型：** 构建
**语言：** Python
**先决条件：** 第一阶段，课程06（概率与分布），课程07（贝叶斯定理）
**时间：** 约120分钟

## 学习目标

- Calculate descriptive statistics, Pearson/Spearman correlation, and covariance matrices from scratch
- Conduct hypothesis tests (t-test, chi-squared) and correctly interpret p-values and confidence intervals
- Use bootstrap resampling to construct confidence intervals for any metric without distributional assumptions
- Distinguish statistical significance from practical significance using effect size measures

## 问题

您训练了两个模型。模型A在测试集上的得分是0.87，模型B的得分为0.89。您选择了模型B进行部署。三周后，生产环境中的指标比之前更差了。发生了什么？

实际上，模型B并没有超过模型A的性能。那0.02的差值只是噪声。您的测试集太小，或者方差太大，或者两者都有。您将随机性伪装成了改进。

这种情况经常发生。Kaggle排行榜的变动、无法复现的研究论文、基于几百个样本就宣布获胜的A/B测试。根本原因总是相同的：有人跳过了统计学分析。

统计学为您提供区分信号和噪声的工具。它告诉您何时存在差异是真实的、应该有多大的信心，以及需要多少数据才能信任结果。每个机器学习流程、每次模型比较、每项实验都需要统计学。没有它，你就是在猜测。

## 概念

### 描述性统计：数据的总结

在您进行任何建模之前，您需要了解数据的具体形态。描述性统计将数据集压缩为几个数字，以捕捉其结构。

**中心趋势度量**回答了“中间在哪里？”的问题。

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

均值是平衡点。中位数是中间值。当它们偏离时，你的分布就会偏斜。收入分布的均值远大于中位数（从亿万富翁到普通人的右偏态分布）。训练过程中的损失分布通常均值远小于中位数（从容易获得的样本到难以获得的样本的左偏态分布）。

**离散度指标**回答了“数据有多分散？”这个问题。

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**百分位数**将排序后的数据分为100个相等的部分。第25百分位数（Q1）意味着有25%的值低于这一点。第50百分位数是中位数。第75百分位数是Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

In machine learning, you need to consider percentiles for inference latency, prediction confidence distributions, and understanding error distributions. A model with low average error but poor P99 error might be ineffective for safety-critical applications.

**Sample vs population statistics.** When calculating variance from a sample, divide by (n-1) instead of n. This is Bessel's correction. It accounts for the fact that the sample mean is not equal to the true population mean. Using n in the denominator systematically underestimates the true variance, while using (n-1) provides an unbiased estimate.

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

在实践中：如果样本数量很大（数千个样本），差异可以忽略不计。如果样本数量很少（几十个样本），则差异就很重要了。

### 相关性：变量如何相互关联

相关性衡量两个变量之间线性关系的强度和方向。

**皮尔逊相关系数**衡量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson assumes that the relationship is linear and that both variables are approximately normally distributed. It is sensitive to outliers. A single extreme point can shift the correlation coefficient from 0.1 to 0.9.

**Spearman rank correlation** measures monotonic associations:

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**何时使用每种方式：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黄金法则：**相关性并不意味着因果关系。冰淇淋销量和溺水死亡人数之间存在关联，因为两者都在夏季增加。你的模型的准确性和参数数量之间存在关联，但添加参数并不会自动提高准确性（参见：过拟合）。

### 协方差矩阵

两个变量之间的协方差衡量它们一起的变化情况：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于d个特征，协方差矩阵C是一个d x d的矩阵，其中C[i][j] = 协方差（feature_i, feature_j）。对角线上的元素C[i][i]是每个特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**Connection to PCA.** PCA eigendecomposes the covariance matrix. The eigenvectors represent the principal components, which are the directions with maximum variance. The eigenvalues indicate the amount of variance captured by each component. This is exactly what was covered in Lesson 10, but now you understand why the covariance matrix is the appropriate choice for decomposition: it encodes all pairwise linear relationships in your data.

**Connection to correlation.** The correlation matrix is the covariance matrix of standardized variables (each divided by its standard deviation). Correlation normalizes the covariance so that all values fall within the range [-1, 1].

### 假设检验

假设检验是一种在不确定性情况下做出决策的方法。你从一个假设开始，收集数据，然后确定这些数据是否与该假设一致。

**设置：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p值**是假设H0为真时，观察到如此极端数据的概率。它并不是H0为真的概率。这是统计学中最常见的误解之一。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

置信区间给出了参数的一个合理值范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间的宽度反映了其精确度。宽度较大的区间意味着不确定性较高。宽度较小的区间则表明你的估计较为精确（但如果数据存在偏差，则可能不一定准确）。

### t-test

t检验用于比较均值。它有多种类型。

**单样本t检验：** 总体均值是否与假设值不同？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```

**两样本t检验（独立）：** 两组均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**配对t检验：**当测量数据成对出现时（同一模型在相同数据分割上进行评估）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在机器学习中，配对t检验很常见：你在相同的10个交叉验证折叠上运行两个模型，并成对比较它们的得分。

### 卡方检验

卡方检验用于检查观测到的频率是否与预期的频率相符。适用于分类数据。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### A/B Testing for Machine Learning Models

在机器学习中进行的A/B测试与网页上的A/B测试不同。模型比较存在特定的挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**程序：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性与实际显著性

一个结果可能在统计上显著，但在实际意义上却毫无意义。只要有足够的数据，即使是微小的差异也会变得在统计上显著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

效应量量化了差异的大小，与样本大小无关：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

始终报告p值和效应大小。p值告诉你差异是否真实存在。效应大小则表明其是否具有重要性。

### 多比较问题

当你测试许多假设时，有些会因偶然性而变得“显著”。如果你在α=0.05的显著性水平下测试20个假设，即使没有什么是真实的，你仍然会预期有1个假阳性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni校正：**将alpha除以测试次数。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在机器学习中，当您比较多个模型的指标、测试多种超参数配置或在多个数据集上进行评估时，这一点非常重要。

### Bootstrap Methods

自举法通过重新采样带有替换功能的数据来估计统计量的抽样分布。不需要对底层分布做出假设。

**算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap置信区间（百分位数法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**为什么引导对于机器学习很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

模型比较的引导工具：

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比配对t检验更为稳健，因为它不做出分布假设。

### 参数化检验与非参数化检验

参数化测试假设特定的分布（通常是正态分布）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

非参数检验不做出分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何时使用非参数方法：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何时使用参数化：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在机器学习实验中，通常n值较小（5或10个交叉验证折叠），因此像Wilcoxon符号秩检验这样的非参数检验比t检验更为合适。

### 中心极限定理：实际意义

CLT表明，随着n的增加，样本均值的分布会趋近于正态分布，无论底层总体分布如何。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

为什么这对机器学习很重要：

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT does NOT do:**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### 常见的机器学习论文中的统计错误

1. **Testing on the training set.** This leads to overfitting. Always include data that the model has never seen during training.

2. **No confidence intervals.** Reporting a single accuracy number without uncertainty makes the results unreproducible and unverifiable.

3. **Ignoring multiple comparisons.** Testing 50 configurations and reporting the best one without correction increases the false positive rate.

4. **Confusing statistical and practical significance.** A p-value of 0.001, even if it indicates a 0.01% improvement in accuracy, is not significant.

5. **Using accuracy on imbalanced data.** Achieving 99% accuracy on a dataset with 99% negative class means the model has learned nothing. Use precision, recall, F1 score, or AUC instead.

6. **Cherry-picking metrics.** Reporting only the metric that best suits your model is not honest evaluation. A thorough evaluation should include all relevant metrics.

7. **Leaking information across train/test splits.** Normalizing data before splitting or using future data to predict past results is unethical and misleading.

8. **Small test sets with no variance estimates.** Evaluating on a small number of samples and claiming a 2% improvement is not meaningful; it’s noise.

9. **Assuming independence when data is not independent.** This applies to medical images from the same patient, multiple sentences in the same document, and correlated observations within a group.

10. **P-hacking.** Trying different tests, subsets, or exclusion criteria until you get a p < 0.05 is unethical and misleading. The result is just an artifact of the search process.

## 构建它

你将实现：

1. **从零开始构建描述性统计**（均值、中位数、众数、标准差、百分位数、四分位距）
2. **相关函数**（皮尔逊和斯皮尔曼，附带协方差矩阵）
3. **假设检验**（单样本t检验、双样本t检验、卡方检验）
4. **自助法置信区间**（适用于任何统计量，无需假设）
5. **A/B测试模拟器**（生成数据、进行测试、检查第一类错误和第二类错误）
6. **统计意义与实际意义的演示**（展示大样本数量使所有结果都变得“显著”）

全部从零开始，仅使用`math`和`random`。不需要numpy或scipy。

## | 关键词 | 翻译 |

| 术语 | 定义 |
|---|---|
| 均值 | 数值之和除以数量。对异常值敏感。 |
| 中位数 | 排序数据中的中间值。对异常值有鲁棒性。 |
| 标准差 | 方差的平方根。以原始单位衡量分布范围。 |
| 百分位数 | 低于该值的数据占给定百分比。 |
| 四分位距 | 第三四分位数减去第一四分位数。表示中间50%数据的分布范围。 |
| 皮尔逊相关系数 | 测量两个变量之间的线性关联。范围为[-1, 1]。 |
| 斯皮尔曼相关系数 | 使用排名测量单调关联。 |
| 协方差矩阵 | 所有特征之间成对协方差的矩阵。 |
| 零假设 | 没有效应或差异的默认假设。 |
| p值 | 在零假设为真的情况下，数据达到这种极端程度的概率。 |
| 置信区间 | 在给定置信水平下参数的合理值范围。 |
| t检验 | 测试均值是否显著不同。使用t分布。 |
| 卡方检验 | 测试观察频率是否与预期频率不同。 |
| 效应大小 | 与样本大小无关的差异幅度。常用Cohen's d表示。 |
| 博福里尼校正 | 将显著性阈值除以测试数量以控制假阳性。 |
| 自助法 | 带有替换的重采样方法，用于估计抽样分布。 |
| 第一类错误 | 假阳性。在零假设为真时拒绝它。 |
| 第二类错误 | 假阴性。在零假设为假时未拒绝它。 |
| 统计功效 | 正确拒绝错误零假设的概率。功效 = 1减去第二类错误率。 |
| 中心极限定理 | 随着样本大小的增加，样本均值收敛于正态分布。 |
| 参数检验 | 假设数据具有特定分布（通常是正态的）。 |
| 非参数检验 | 不假设任何分布。适用于排名或符号。 |
