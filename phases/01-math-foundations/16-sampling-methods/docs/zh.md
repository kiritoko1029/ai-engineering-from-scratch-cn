# 采样方法

> 采样是人工智能探索可能性空间的方式。

**类型：** 构建
**语言：** Python
**先决条件：** 第一阶段，课程06-07（概率论，贝叶斯定理）
**时间：** 约120分钟

## 学习目标

- Implement inverse CDF, rejection sampling, and importance sampling from scratch using only uniform random numbers
- Build temperature, top-k, and top-p (nucleus) sampling for language model token generation
- Explain the reparameterization trick and why it enables backpropagation through sampling in VAEs
- Run Metropolis-Hastings MCMC to sample from an unnormalized target distribution

## 问题

语言模型完成对提示的处理，生成了一个包含50,000个logits的向量，每个logit对应其词汇表中的每一个标记。现在它必须选择一个。如何做到这一点？

如果它总是选择概率最高的标记，那么每个响应都相同。这是确定性的，令人乏味。如果它均匀随机选择，输出将是无意义的。答案介于这两个极端之间，而这种中间状态是由采样控制的。

采样不仅限于文本生成。强化学习通过采样路径来估计策略梯度。VAE通过从已学习的分布中采样并通过随机性进行反向传播来学习潜在表示。扩散模型通过采样噪声并迭代去噪来生成图像。蒙特卡洛方法估计没有封闭形式解的积分。MCMC算法探索无法枚举的高维后验分布。

每个生成式AI系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本课程从基础开始，逐步介绍所有主要的采样方法，从均匀随机数到现代LLM和生成模型所使用的技术。

## 概念

### 为什么采样很重要

采样在人工智能和机器学习中扮演着四个基本角色：

**生成。**语言模型、扩散模型和生成对抗网络都是通过采样来产生输出的。采样算法直接控制创造力、连贯性和多样性。温度采样、Top-K采样和核采样是工程师日常使用的工具。

**训练。**随机梯度下降通过采样小批量数据。Dropout技术通过采样神经元来实现停用。数据增强通过采样随机变换来实现。在强化学习中，重要性采样通过重新加权样本来减少梯度方差（如PPO、TRPO）。

**估计。**机器学习中的许多量没有封闭形式的解决方案。数据分布上的期望损失、基于能量的模型的配分函数、贝叶斯推断中的证据。蒙特卡洛估计通过对样本进行平均来近似所有这些。

**探索。**在贝叶斯推断中，MCMC算法探索后验分布。进化策略通过采样参数扰动来实现探索。汤普森抽样在风险游戏中平衡探索和利用。

核心挑战是：你只能直接从简单分布（如均匀分布、正态分布）中进行采样。对于其他所有情况，你需要一种方法将简单样本转换为目标分布的样本。

### 均匀随机抽样

所有采样方法都从这里开始。均匀随机数生成器产生的值位于[0, 1)范围内，其中每个等长的子区间都有相同的概率被选中。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

To uniformly sample from a discrete set of n items, generate U and return floor(n * U). To sample from a continuous range [a, b], calculate a + (b - a) * U.

The key insight: A single uniform random number contains exactly the right amount of randomness to produce one sample from any distribution. The trick is finding the right transformation.

### 逆累积分布函数方法（逆变换采样）

累积分布函数（CDF）将值映射到概率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

逆累积分布函数将概率映射回数值。如果U服从均匀分布(0, 1)，那么X = F_inverse(U)就会遵循目标分布。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

指数分布示例：

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

当你可以以封闭形式写出F_inverse时，这种方法非常有效。对于正态分布，没有封闭形式的累积分布函数逆函数，因此我们使用其他方法（如Box-Muller法或数值近似）。

**离散版本：**对于离散分布，将累积分布函数构建为累积和，生成U，然后找到累积和超过U的第一个索引。这就是第06课中`sample_categorical`的工作方式。

### 拒绝采样

当您无法反转CDF，但可以将目标PDF评估到一个常数时，拒绝采样方法有效。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

The tighter the bound M, the higher the acceptance rate. In low dimensions (1-3), rejection sampling works well. In high dimensions, the acceptance rate drops exponentially because most of the proposal volume gets rejected. This is the curse of dimensionality for rejection sampling.

**Example: sampling from a truncated normal.** Use a uniform proposal over the truncated range. The bound M is the maximum of the normal probability distribution in that range.

**Example: sampling from a semicircle.** Propose uniformly in the bounding rectangle. Accept if the point falls inside the semicircle. This is how Monte Carlo computes pi: the acceptance rate equals the area ratio pi/4.

### 重要性抽样

有时你不需要目标分布p(x)的样本。你需要估计在p(x)下的期望值，而你有来自不同分布q(x)的样本。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

This is crucial in reinforcement learning. In PPO (Proximal Policy Optimization), you collect trajectories under an old policy pi_old but want to optimize a new policy pi_new. The importance weight is pi_new(a|s) / pi_old(a|s). PPO clips these weights to prevent the new policy from diverging too far from the old one.

The variance of the importance sampling estimator depends on how similar q is to p. If q is very different from p, a few samples get enormous weights and dominate the estimate. Self-normalized importance sampling divides by the sum of weights to reduce this problem:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### 蒙特卡洛估计

蒙特卡洛估计通过平均随机样本来近似积分。大数定律保证收敛性。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

错误率与维度无关。这就是为什么在高维空间中，蒙特卡洛方法占主导地位，因为基于网格的积分在高维空间中是不可能的。

**估算圆周率：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估计期望值：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### 马尔可夫链蒙特卡洛方法（MCMC）：Metropolis-Hastings

MCMC构建了一个马尔可夫链，其稳态分布为目标分布p(x)。经过足够多的步骤后，该链中的样本（近似地）就是来自p(x)的样本。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

对于对称提案（q(x'|x) = q(x|x')），比率简化为p(x')/p(x)。这是原始的Metropolis算法。

**其工作原理。**接受规则确保了详细平衡：处于x状态并移动到x'的概率等于处于x'状态并移动到x的概率。详细平衡意味着p(x)是链的稳态分布。

**实际考虑因素：**
- 燃烧期：在链达到平衡之前丢弃早期样本
- 稀疏化：保留每k个样本以减少自相关性
- 提案规模：太小则链移动缓慢（接受率高，探索速度慢）；太大则大多数提案被拒绝（接受率低，停滞不前）
- 在高维度下，高斯提案的最佳接受率约为0.234

### Gibbs Sampling

Gibbs抽样是多变量分布的MCMC的一种特殊情况。它不是一次性在所有维度上提出移动，而是从其条件分布中一次更新一个变量。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

吉布斯抽样要求能够从每个条件分布 p(x_i | x_{-i}) 中进行抽样。对于许多模型来说，这是可行的：
- 贝叶斯网络：条件概率基于图结构确定
- 高斯混合模型：条件概率为高斯分布
- 伊辛模型：每个自旋的条件概率仅取决于其邻居

接受率始终为1（每个提议都被接受），因为从精确条件分布中进行抽样自动满足详细平衡。

**限制。**当变量高度相关时，吉布斯抽样混合速度较慢，因为一次更新一个变量无法在分布中做出大的对角移动。

### 温度采样（用于大语言模型）

语言模型为每个词汇表中的每个词输出logits z_1, ..., z_V。Softmax将这些概率转换为概率值。在softmax之前，温度参数会重新调整logits的值：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**为何有效。**将logits除以T < 1可以放大logits之间的差异。如果z_1 = 2且z_2 = 1，除以T = 0.5后得到z_1/T = 4和z_2/T = 2，从而使差距更大。经过softmax处理后，logit值最高的令牌会获得更大的份额。

**实际应用：**
- T = 0.0：贪婪解码，适用于事实性问答
- T = 0.3-0.7：稍具创造性，适用于代码生成
- T = 0.7-1.0：平衡，适用于一般对话
- T = 1.0-1.5：创意写作，头脑风暴
- T > 1.5：随机性增加，很少有用

温度并不改变哪些令牌是可能的。它改变了分配给每个令牌的概率质量。

### Top-k Sampling

Top-k采样将候选集限制为概率最高的k个令牌，然后重新归一化并从该受限集合中进行采样。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k防止模型选择那些存在于词汇分布长尾中的极不可能出现的标记（如拼写错误、无意义词）。问题在于：k的值是不受上下文影响的固定值。当模型确信某个标记出现的概率为95%时，k=40仍然允许有39种替代选项。而当模型不确定时（概率分布在1000个标记中），k=40则会排除合理的选项。

### Top-p (Nucleus) Sampling

Top-p采样动态调整候选集的大小。它不是保持固定数量的标记，而是保持累积概率超过p的最小标记集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

当模型具有信心时，核采样会保留较少的标记（可能是2-3个）。当模型不确定时，它会保留较多标记（可能是200个）。这种自适应行为是为什么核采样通常比top-k方法产生更好的文本的原因。

**常见组合：**
- 温度0.7 + top-p 0.9：通用的良好设置
- 温度0.0（贪婪）：适用于确定性任务的最佳设置
- 温度1.0 + top-k 50：Fan等人（2018）原始论文中的设置

可以结合使用top-k和top-p。先应用top-k，然后在剩余的数据集上应用top-p。

### 重参数化技巧（用于VAEs）

变分自编码器（VAEs）通过将输入编码到潜在空间中，从该分布中采样，然后将样本解码回来来学习。问题是在采样操作之后无法进行反向传播。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

这种重新参数化的技巧将随机性与参数分离开来：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

This works because N(mu, sigma^2) has the same distribution as mu + sigma * N(0, 1). The key insight: move the randomness to a parameter-free source (epsilon), then express the sample as a differentiable transformation of the parameters.

**In the VAE training loop:**
1. The encoder outputs mu and log(sigma^2) for each input.
2. Sample epsilon ~ N(0, 1).
3. Compute z = mu + sigma * epsilon.
4. Decode z to reconstruct the input.
5. Backpropagate through steps 4, 3, 2, 1 (possible because step 3 is differentiable).

Without this reparameterization trick, VAEs cannot be trained with standard backpropagation. This single insight made VAEs practical.

### Gumbel-Softmax（可微分类采样）

这种重新参数化的技巧适用于连续分布（高斯分布）。对于离散分类分布，我们需要采用不同的方法。Gumbel-Softmax提供了一种可微分的近似方法来处理分类采样。

**Gumbel-Max技巧（不可微分）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax实现了对离散样本的连续松弛处理。其输出是一个概率向量（软一热编码），而不是硬一热编码。梯度通过softmax传递。在训练的前向过程中，可以使用“直通”估计器：使用前向过程中的硬argmax，而后向过程则使用软Gumbel-Softmax梯度。

**应用：**
- VAE中的离散潜在变量
- 神经架构搜索（选择离散操作）
- 硬注意力机制
- 带有离散动作的强化学习

### 分层抽样

标准蒙特卡洛抽样可能会随机导致样本空间出现空白。分层抽样通过将空间划分为不同的层并从每个层中抽样，确保覆盖所有区域。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

与标准蒙特卡洛方法相比，分层抽样始终具有更低或相等的方差：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**应用：**
- 数值积分（准蒙特卡洛方法）
- 训练数据分割（确保每个折叠中的类别平衡）
- 分层重要性采样（结合两种技术）
- NeRF（神经辐射场）使用沿相机光线的分层采样

### 与扩散模型的连接

扩散模型通过采样过程生成图像。前向过程中，在T个步骤内对图像添加高斯噪声，直到其变为纯噪声。反向过程中，模型学习去噪技术，逐步恢复原始图像。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

本课程中方法的关联：
- 每个去噪步骤都使用重新参数化技巧（采样噪声，应用确定性变换）
- 噪声调度{alpha_t}控制一种温度退火过程
- 训练过程中使用蒙特卡洛估计来近似ELBO（证据下界）
- 扩散模型中的祖先采样是一种马尔可夫链（每一步仅依赖于当前状态）

整个图像生成过程是迭代采样：从噪声开始，在每一步中，根据学习到的去噪模型采样出一个稍微不那么嘈杂的版本。

```figure
monte-carlo-pi
```

## 构建它

### 步骤1：均匀和逆CDF采样

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

生成10,000个指数样本，并验证其均值等于1/λ。

### 步骤2：拒绝抽样

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

使用拒绝抽样从截断的正态分布中抽取样本。通过绘制样本的直方图来验证其形状。

### 步骤3：重要性抽样

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

使用均匀分布估计正态分布下的E[X^2]，并与已知答案(mu^2 + sigma^2)进行比较。

### 步骤4：π的蒙特卡洛估计

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步骤5：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

样本来自双峰分布（两个高斯分布的混合）。可视化链条的轨迹。

### 步骤6：吉布斯抽样

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步骤7：温度采样

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示温度如何改变一组令牌逻辑向量的输出分布。

### 步骤8：Top-k和top-p采样

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步骤9：重新参数化技巧

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

证明梯度通过重新参数化的样本流动，但不通过直接采样流动。

### 步骤10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示温度下降如何使输出趋近于一个独热向量。所有实现及可视化内容位于`code/sampling.py`中。

## 使用它

使用NumPy和SciPy，生产版本如下：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

对于大规模MCMC，使用专用库：
- PyMC：基于NUTS的完整贝叶斯建模（自适应HMC）
- emcee：集成MCMC采样器
- NumPyro/JAX：GPU加速MCMC

这些库都是你从零开始构建的。现在你知道它们各自的功能了。

## 练习

1. Implement inverse CDF sampling for the Cauchy distribution. The CDF is F(x) = 0.5 + arctan(x)/pi. Generate 10,000 samples and plot the histogram against the true PDF. Notice the heavy tails (extreme values far from center).

2. Use rejection sampling to generate samples from a Beta(2, 5) distribution using a Uniform(0, 1) proposal. Plot the accepted samples against the true Beta PDF. What is the theoretical acceptance rate?

3. Estimate the integral of sin(x) from 0 to pi using Monte Carlo with 1,000, 10,000, and 100,000 samples. Compare the error at each level. Verify that the error scales as O(1/sqrt(N)).

4. Implement Metropolis-Hastings to sample from a 2D distribution p(x, y) proportional to exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Plot the samples and the chain trajectory. Experiment with different proposal standard deviations.

5. Build a complete text generation demo: given a vocabulary of 10 words with logits, generate sequences of 20 tokens using (a) greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9. Compare the diversity of outputs across 5 runs.

## | 关键词 | 翻译 |

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------|
| 采样 | “抽取随机值” | 根据概率分布生成值。所有生成式人工智能背后的机制 |
| 均匀分布 | “所有值等可能” | [a, b]区间内的每个值都有相等的概率密度1/(b-a)。所有采样方法的起点 |
| 逆累积分布函数 | “概率变换” | F_inverse(U)将均匀样本转换为来自已知累积分布函数的任何分布的样本。精确且高效 |
| 拒绝采样 | “提出并接受/拒绝” | 从简单提议中生成，以目标/提议比例的概率接受。精确但浪费样本 |
| 重要性采样 | “重新加权样本” | 使用q(x)中的样本通过p(x)/q(x)对每个样本进行加权，以估计p(x)下的期望。是强化学习中PPO的核心 |
| 蒙特卡洛 | “平均随机样本” | 作为样本平均值近似积分。无论维度如何，误差为O(1/sqrt(N)) |
| MCMC | “收敛的随机游走” | 构建其稳态分布为目标分布的马尔可夫链。Metropolis-Hastings是基础算法 |
| Metropolis-Hastings | “有时接受上升路径，有时接受下降路径” | 提出移动，基于密度比接受。详细平衡确保收敛到目标分布 |
| Gibbs采样 | “一次一个变量” | 在固定其他变量的情况下，根据其条件分布更新每个变量。100%接受率 |
| 温度 | “置信度旋钮” | 在softmax之前将logits除以T。T<1使结果更自信，T>1使结果更多样化 |
| Top-k采样 | “保留k个最佳” | 消除除概率最高的k个令牌外的所有令牌，重新归一化后采样。固定候选集大小 |
| 核心采样（top-p） | “保留可能的那些” | 保留累积概率超过p的最小令牌集。自适应调整候选集大小 |
| 重参数化技巧 | “将随机性移出” | 写出z = mu + sigma * epsilon，其中epsilon ~ N(0,1)。使采样可微分。对VAE训练至关重要 |
| Gumbel-Softmax | “软分类采样” | 使用Gumbel噪声和温度进行softmax的分类采样的可微近似 |
| 分层采样 | “强制覆盖” | 将样本空间划分为层，从每层抽样。总是比简单的蒙特卡洛方法具有更低的方差 |
| 燃烧期 | “预热期” | 在马尔可夫链达到其稳态分布之前丢弃初始MCMC样本 |
| 详细平衡 | “可逆条件” | p(x) * T(x->y) = p(y) * T(y->x)。是p成为马尔可夫链的稳态分布的充分条件 |
| 扩散采样 | “迭代去噪” | 从噪声开始，应用学习到的去噪步骤生成数据。每一步都是条件抽样操作 |

## 更多阅读资料

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - detailed tutorial on MCMC foundations  
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - original Gumbel-Softmax paper  
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus (top-p) sampling paper  
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - VAE paper introducing the reparameterization trick  
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM connects sampling to image generation
