# 规范与距离

> Your distance function defines what "similar" means. Choose wrong and everything downstream breaks.

**类型：**构建
**语言：**Python
**先决条件：**第一阶段，课程01（线性代数直觉），课程02（向量、矩阵与运算）
**时间：**约90分钟

## 学习目标

- Implement the L1, L2, cosine, Mahalanobis, Jaccard, and edit distance functions from scratch.
- Select the appropriate distance metric for a given machine learning task and explain why alternative metrics fail.
- Connect the L1 and L2 norms to LASSO and Ridge regularization and their geometric constraint regions.
- Demonstrate how the same dataset produces different nearest neighbors under different distance metrics.

## 问题

您有两个向量。它们可能是词嵌入，也可能是用户资料，或者像素数组。您需要知道：它们之间的相似程度如何？

答案完全取决于您选择的距离函数。根据一种度量标准，两个数据点可能是最近邻；而根据另一种度量标准，它们可能相距甚远。您的KNN分类器、推荐引擎、向量数据库、聚类算法、损失函数——所有这些都依赖于这个选择。如果选错了，模型就会优化错误的内容。

没有普遍适用的最佳距离函数。L2距离适用于空间数据。余弦相似度在自然语言处理中占主导地位。Jaccard度量适用于集合。编辑距离适用于字符串。马氏距离考虑了相关性。Wasserstein移动概率质量。每一种距离函数都反映了关于“相似”的不同假设。

本课程从基础开始讲解每一个主要的距离函数，说明何时使用每种函数是正确的选择，并演示了相同的数据根据所使用的度量标准会产生完全不同的最近邻。

## 概念

### Norms: Measuring Vector Magnitude

范数衡量了向量的“大小”。两个向量之间的任何距离函数都可以表示为它们差值的范数：d(a, b) = ||a - b||。因此，理解范数就是理解距离。

### L1范数（曼哈顿距离）

L1范数对所有分量的绝对值进行求和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为曼哈顿距离，因为它衡量了在只能沿轴移动的城市网格上行走的距离。不允许对角线移动。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

When to use L1 regularization:
- High-dimensional sparse data (text features, one-hot encodings)
- When you want robustness to outliers (a single huge difference does not dominate)
- Feature selection problems (L1 regularization promotes sparsity)

Connection to L1 regularization (Lasso): Adding ||w||_1 to your loss function penalizes the sum of absolute weight values. This pushes small weights to exactly zero, performing automatic feature selection. The L1 penalty creates diamond-shaped constraint regions in weight space, and the corners of diamonds lie on the axes where some weights are zero.

Connection to loss functions: Mean Absolute Error (MAE) is the average L1 distance between predictions and targets. It penalizes all errors linearly, making it robust to outliers compared to Mean Squared Error (MSE).

### L2 norm (Euclidean distance)

L2范数是直线距离。即各分量平方之和的平方根。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这是你在几何课上学到的距离。在n维空间中，这就是毕达哥拉斯定理的应用。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

When to use L2 regularization:
- Low-to-medium dimensional continuous data
- When the feature scales are comparable
- Physical distances (spatial data, sensor readings)
- Image similarity at the pixel level

Connection to L2 regularization (Ridge): Adding ||w||_2^2 to your loss function penalizes large weights. Unlike L1, it does not push weights to zero. It shrinks all weights toward zero proportionally. The L2 penalty creates circular constraint regions, so there are no corners on axes. Weights get small but rarely exactly zero.

Connection to loss functions: Mean Squared Error (MSE) is the average of L2 distances squared. Squaring penalizes large errors more heavily than small ones.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp Normes：一般家族

L1和L2是Lp范数的特殊情况：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的p值会产生不同形状的“单位球”（即距离原点距离为1的所有点组成的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-无穷范数（切比雪夫距离）

随着 p 趋近于无穷大，Lp范数收敛到最大的绝对分量。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两点之间的距离由它们差异最大的那个维度决定。其他所有维度都被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用L无穷大：
- 当任何单一维度中的最坏情况偏差重要时
- 游戏棋盘（国际象棋中的国王移动距离为L无穷大：任意方向一步的代价为1）
- 制造公差（每个维度必须符合规格要求）

### 余弦相似度和余弦距离

余弦相似度用于衡量两个向量之间的角度，而不考虑它们的大小。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它的范围是从-1（相反方向）到+1（相同方向）。垂直向量的余弦相似度为0。

余弦距离将其转换为距离：余弦距离 = 1 - 余弦相似度。这个范围从0（相同方向）到2（相反方向）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Why cosine dominates NLP and embeddings: In text, the length of documents should not affect similarity. A document about cats that is twice as long as another document about cats should still be “similar.” Cosine similarity ignores magnitude (length) and only cares about direction. Two documents with the same word distribution but different lengths point in the same direction and get cosine similarity 1.0.

When to use cosine similarity:
- Text similarity (TF-IDF vectors, word embeddings, sentence embeddings)
- Any domain where magnitude is noise and direction is signal
- Recommendation systems (user preference vectors)
- Embedding search (vector databases almost always use cosine or dot product)

### 点积相似性与余弦相似性

两个向量的点积为：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度是点积除以两个向量的模长后的归一化结果。当两个向量都已经单位归一化（模长=1）时，点积和余弦相似度是相同的。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

当它们存在差异时：点积包含大小信息。大小较大的向量会得到更高的点积得分。这在某些检索系统中很重要，因为在这些系统中，希望“热门”项目获得更高的排名。大小实际上起到了隐含的质量或重要性信号的作用。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

在实践中：
- 当需要纯粹的方向相似性时，使用余弦相似度。
- 当幅度具有有意义的信息时，使用点积。
- 许多向量数据库（如Pinecone、Weaviate、Qdrant）允许你在这两种方法之间选择。
- 如果你的嵌入数据已经进行了L2归一化，那么选择哪种方法并不重要。

### Mahalanobis Distance

欧几里得距离对所有维度平等对待。但如果你的特征之间存在相关性或具有不同的尺度，L2距离会给出误导性的结果。

马氏距离则考虑了数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中S是数据的协方差矩阵。

直观地说：马氏距离首先去相关并规范化数据（白化），然后在转换后的空间中计算L2距离。如果S是单位矩阵（不相关、方差为单位的特征值），则马氏距离简化为欧几里得距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用马氏距离：
- 异常值检测（与均值有较大马氏距离的点为异常值）
- 特征具有不同尺度和相关性时的分类
- 当有足够的数据来估计可靠的协方差矩阵时
- 制造业中的质量控制（多变量过程监控）

### Jaccard Similarity (for sets)

Jaccard相似度用于衡量两个集合之间的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

它的范围是从0（无重叠）到1（相同的集合）。Jaccard距离等于1减去Jaccard相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

When to use Jaccard:
- Comparing sets of tags, categories, or features
- Document similarity based on word presence (not frequency)
- Near-duplicate detection (MinHash approximation of Jaccard)
- Comparing binary feature vectors (presence/absence data)
- Evaluating segmentation models (Intersection over Union = Jaccard)

### 编辑距离（莱文斯坦距离）

编辑距离计算将一个字符串转换为另一个字符串所需的最少单字符操作次数。这些操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Calculated using dynamic programming. Fill a matrix where the entry (i, j) represents the edit distance between the first i characters of string A and the first j characters of string B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：
- 拼写检查和纠正
- DNA序列比对（带有加权操作）
- 模糊字符串匹配
- 杂乱文本数据的去重

### KL散度（不是距离，但像距离一样使用）

KL散度用于衡量一个概率分布与另一个概率分布的差异。这一内容在第09课中有所介绍，但它也属于本次讨论的范围，因为人们将其作为“距离”来使用，尽管它实际上并非真正的距离。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键特性：KL散度不是对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

This means it fails the basic requirement of a distance metric. It also does not satisfy the triangle inequality. It is a divergence, not a distance.

Forward KL (D_KL(P || Q)) is “mean-seeking”: Q tries to cover all modes of P.
Reverse KL (D_KL(Q || P)) is “mode-seeking”: Q focuses on a single mode of P.

When you see KL divergence:
- VAEs (the KL term in the ELBO pushes the latent distribution toward a prior)
- Knowledge distillation (the student tries to match the teacher’s distribution)
- RLHF (the KL penalty keeps the fine-tuned model close to the base model)
- Policy gradient methods (constraining policy updates)

### Wasserstein Distance (Earth Mover's Distance)

Wasserstein距离用于衡量将一个概率分布转换为另一个概率分布所需的最小“工作量”。可以这样理解：如果一个分布是一堆泥土，另一个分布是一个洞，那么你需要移动多少泥土以及移动的距离是多少？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，可以简化为累积分布函数的绝对差值的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么Wasserstein很重要：
- 它是一个真正的度量标准（对称，满足三角不等式）
- 即使分布不重叠时也能提供梯度（KL散度会趋于无穷大）
- 这一特性使其成为Wasserstein生成对抗网络的核心，解决了原始生成对抗网络的训练不稳定问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

When to use Wasserstein:
- GAN training (WGAN, WGAN-GP)
- Comparing distributions that may not overlap
- Optimal transport problems
- Image retrieval (comparing color histograms)

### 为什么不同的任务需要不同的距离

| 任务 | 最佳距离算法 | 原因 |
|------|--------------|-----|
| 文本相似度 | 余弦距离 | 数值是噪声，方向才有意义 |
| 图像像素比较 | L2距离 | 空间关系很重要，特征具有可比规模 |
| 稀疏高维特征 | L1距离 | 稳健，不会放大罕见的大差异 |
| 集合重叠（标签、类别） | Jaccard相似度 | 数据本质上是集合值，而非向量 |
| 字符串匹配 | 编辑距离 | 操作符合人类编辑直觉 |
| 异常检测 | 马氏距离 | 考虑特征相关性和规模 |
| 分布比较 | KL散度 | 通过Q而不是P衡量信息损失 |
| GAN训练 | Wasserstein距离 | 即使分布不重叠也能提供梯度 |
| 嵌入（向量数据库） | 余弦距离或点积 | 嵌入用于编码方向中的意义 |
| 推荐 | 点积 | 数值可以表示流行度或置信度 |
| DNA序列 | 加权编辑距离 | 替换成本随核苷酸对不同而变化 |
| 制造质量控制 | L-无穷距离 | 任何维度上的最坏情况偏差都很重要 |

### 与损失函数的连接

损失函数是应用于预测与目标的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的连接

正则化在损失函数中对权重施加了规范惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么L1会产生稀疏性而L2则不会：想象在二维权重空间中的约束区域。L1是一个菱形，L2是一个圆形。损失函数的轮廓（椭圆）最有可能在其中一个权重为零的角落与菱形接触。它们会在两个权重都不为零的平滑点处与圆形接触。

### 最近邻搜索

Every distance function implies a nearest neighbor search problem: given a query point, find the closest points in a dataset.

Exact nearest neighbor search is O(n * d) per query in a dataset of n points with d dimensions. For large datasets, this is too slow.

Approximate Nearest Neighbor (ANN) algorithms trade a small amount of accuracy for massive speed gains:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW（分层可导航小世界）是现代向量数据库中的主导算法。它构建了一个多层图，其中每个节点都连接到其最近的邻居。搜索从顶层开始（稀疏，长跳跃），然后向下到底层（密集，短跳跃）。

```figure
norm-unit-balls
```

## 构建它

### 步骤1：所有规范与距离函数

请参阅`code/distances.py`以获取完整的实现。每个函数都是使用基本的Python数学算法从头开始构建的。

### 步骤2：相同的数据，不同的距离，不同的邻居

`distances.py`中的演示创建了一个数据集，选取了一个查询点，并展示了根据距离度量标准，最近邻如何发生变化。在L1距离下“最远”的点可能在L2距离或余弦距离下并非最远。

### 步骤3：嵌入相似性搜索

该代码包含了一个模拟的嵌入相似性搜索功能，它使用余弦相似度而非L2距离来查找与查询最相似的“文档”，这表明排名可能会有所不同。

## 使用它

最常见的实际用途：在向量数据库中查找相似项。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用`model.encode(text)`然后搜索向量数据库时，底层会发生以下过程。嵌入模型将文本映射到向量上。向量数据库使用ANN算法计算你的查询向量与每个存储的向量之间的余弦相似度（或点积），以避免检查所有向量。

## 练习

1. Calculate the L1, L2, and L-infinity distances between the points (1, 2, 3) and (4, 0, 6). Verify that L-inf <= L2 <= L1 always holds for any pair of points. Prove why this ordering is guaranteed.

2. Create two vectors where the cosine similarity is high (> 0.9) but the L2 distance is large (> 10). Explain geometrically what is happening. Then create two vectors where the cosine similarity is low (< 0.3) but the L2 distance is small (< 0.5).

3. Implement a function that takes a dataset and a query point and returns the nearest neighbor under L1, L2, cosine, and Mahalanobis distance. Find a dataset where all four methods disagree on which point is the nearest.

4. Calculate the Wasserstein distance between [0.5, 0.5, 0, 0] and [0, 0, 0.5, 0.5] by hand using the CDF method. Then calculate it between [0.25, 0.25, 0.25, 0.25] and [0, 0, 0.5, 0.5]. Which one is larger and why?

5. Implement MinHash for approximate Jaccard similarity. Generate 100 random sets, compute exact Jaccard for all pairs, and compare with MinHash approximations using 50, 100, and 200 hash functions. Plot the approximation error.

## | 关键词 | 翻译 |

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------|
| 范数 | “向量的大小” | 一个将向量映射到非负标量的函数，满足三角不等式、绝对齐次性，且零向量时为零 |
| L1范数 | “曼哈顿距离” | 绝对分量值之和。在优化中产生稀疏性。对异常值具有鲁棒性 |
| L2范数 | “欧几里得距离” | 分量平方和的平方根。欧几里得空间中的直线距离 |
| Lp范数 | “广义范数” | p次幂的绝对值之和的p次根。L1和L2是特殊情况 |
| L-无穷大范数 | “最大范数”或“切比雪夫距离” | 最大的绝对分量值。当p趋于无穷大时Lp的极限值 |
| 余弦相似度 | “向量之间的角度” | 点积除以两者的大小。范围从-1到+1。忽略向量长度 |
| 余弦距离 | “1减去余弦相似度” | 将余弦相似度转换为距离。范围从0到2 |
| 点积 | “未归一化的余弦” | 分量乘积之和。等于余弦相似度乘以两者的大小 |
| 马氏距离 | “相关意识距离” | 使用数据协方差矩阵进行白化（去相关和归一化）后的L2距离 |
| 杰卡德相似度 | “集合重叠” | 交集大小除以并集大小。适用于集合，不适用于向量 |
| 编辑距离 | “莱文斯坦距离” | 将一个字符串转换为另一个字符串所需的最小插入、删除和替换次数 |
| KL散度 | “分布之间的距离” | 不是真正的距离（不对称）。衡量使用Q编码P所增加的额外比特数 |
| 瓦瑟斯坦距离 | “地球搬运者距离” | 将质量从一种分布运输到另一种分布所需的最小工作量。真正的度量标准 |
| 近似最近邻 | “ANN搜索” | 比精确搜索更快找到近似最近的点的算法（HNSW、LSH、IVF） |
| HNSW | “向量DB算法” | 分层可导航小世界图。多层图用于快速近似最近邻搜索 |
| L1正则化 | “Lasso” | 将权重的L1范数加到损失函数中。使权重归零（稀疏性） |
| L2正则化 | “Ridge”或“权重衰减” | 将权重的平方L2范数加到损失函数中。使权重趋于零但不产生稀疏性 |
| 弹性网络 | “L1 + L2” | 结合L1和L2正则化。比单独使用更能处理相关特征组 |

## 更多阅读资料

- [FAISS：一种高效相似性搜索库](https://github.com/facebookresearch/faiss) - Meta用于数十亿级ANN搜索的库  
- [Wasserstein GAN（Arjovsky等人，2017年）](https://arxiv.org/abs/1701.07875) - 介绍GANs中地球搬运者距离概念的论文  
- [局部敏感哈希（Indyk & Motwani，1998年）](https://dl.acm.org/doi/10.1145/276698.276876) - ANN算法的基础  
- [词表示的高效估计（Mikolov等人，2013年）](https://arxiv.org/abs/1301.3781) - Word2Vec，其中余弦相似성이嵌入的默认标准  
- [sklearn.neighbors文档说明](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn中距离度量与邻居算法的实用指南
