# 数值稳定性

浮点数是一种容易出错的抽象概念。在训练过程中它会给你带来麻烦，而你根本不会察觉到。

**类型：**构建
**语言：**Python
**先决条件：**第一阶段，课程01-04
**时间：**约120分钟

## 学习目标

- Implement numerically stable softmax and log-sum-exp using the max-subtraction trick
- Identify overflow, underflow, and catastrophic cancellation in floating-point computations
- Verify analytical gradients against numerical gradients using centered finite differences
- Explain why bfloat16 is preferred over float16 for training and how loss scaling prevents gradient underflow

## 问题

你的模型训练了三个小时，然后损失值变成了NaN。你添加了打印语句。在9000步时，logits正常。但在9001步时，它们变为`inf`。到了9002步，每个梯度都变成`nan`，训练就此终止。

或者：你的模型完成了训练，但准确率比论文中声称的低2%。你检查了所有方面。架构、超参数和数据都一致。问题是论文使用了float32类型，而你使用的是float16类型，且没有正确的缩放方式。32位的累积舍入误差悄悄降低了你的准确率。

或者：你从零开始实现了交叉熵损失函数。它在较小的logits上有效。当logits超过100时，它会返回`inf`。softmax溢出是因为`exp(100)`超出了float32能表示的范围。每个机器学习框架都用一个两行代码的方法来处理这个问题。你不知道这个技巧的存在。

数值稳定性不是一个理论问题。它是训练成功与默默失败之间的区别。你最终需要调试的每个严重的人工智能错误都与浮点数有关。

## 概念

### IEEE 754：计算机如何存储实数

计算机将实数存储为遵循IEEE 754标准的浮点值。一个浮点数由三部分组成：符号位、指数和尾数（ significand ）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

尾数决定了精度（有多少位有效数字）。指数决定了范围（数字可以有多大或小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

`float32`提供大约7位小数的精度。这意味着它可以区分1.0000001和1.0000002，但无法区分1.00000001和1.00000002。超过7位数字后，所有数值都只是四舍五入的误差。

`float16`提供大约3位小数。它所能表示的最大数字是65,504。对于机器学习来说，这个数字太小了，因为逻辑值、梯度和激活值通常远远超过这个数值。

`bfloat16`是Google对`float16`范围问题的解决方案。它与`float32`具有相同的8位指数（相同的值域，最高可达3.4e38），但只有7位尾数位（精度低于`float16`）。对于训练神经网络来说，值域比精度更重要，因此`bfloat16`通常更适用。

### 为什么 0.1 + 0.2 不等于 0.3

数字0.1无法在二进制浮点数中精确表示。在二进制中，它是一个重复分数：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 truncates this to 23 bits of mantissa. The stored value is approximately 0.100000001490116. Similarly, 0.2 is stored as approximately 0.200000002980232. Their sum is 0.300000004470348, not 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

这对于机器学习很重要，因为：

1. 像“如果损失小于阈值”这样的损失比较可能会得出错误的答案。
2. 累积许多小值（数千步的梯度更新）会导致结果偏离真实总和。
3. 如果你用“==”比较浮点数，校验和和可重复性测试会失败。

解决方法：永远不要使用“==”来比较浮点数。应使用`abs(a - b) < epsilon`或`math.isclose()`。

### 灾难性取消

当您减去两个几乎相等的浮点数时，有效数字会相互抵消，最终剩下的只是被四舍五入产生的噪声，这些噪声会变成前导数字。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

That is a 19% relative error from a single subtraction. In machine learning, this occurs whenever you:

- Compute variance of data with a large mean: `E[x^2] - E[x]^2` when E[x] is large
- Subtract nearly equal log-probabilities
- Compute finite-difference gradients with too-small epsilon

The fix: rearrange formulas to avoid subtracting large, nearly equal numbers. For variance, use the Welford algorithm or center the data first. For log-probabilities, work in log-space throughout.

### Overflow and Underflow

当结果过大无法表示时，会发生溢出。当结果过小（接近零而不是最小可表示的正数）时，会发生下溢。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()`函数是机器学习中溢出现象的主要来源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()`函数的作用方向是相反的：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在机器学习中，`exp()`出现在softmax、sigmoid和概率计算中。`log()`出现在交叉熵、对数似然和KL散度中。组合`log(exp(x))`如果没有正确的技巧就会成为一个难题。

### Log-Sum-Exp Trick

直接计算 `log(sum(exp(x_i)))` 在数值上存在风险。如果任何 `x_i` 的值很大，`exp(x_i)` 会发生溢出。如果所有 `x_i` 都非常负，那么每个 `exp(x_i)` 都会下溢为零，而 `log(0)` 的值为 `-inf`。

解决方法：先减去最大值，再进行指数运算。

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

为什么这有效：在减去 `max(x)` 之后，最大的指数是 `exp(0) = 1`。不可能发生溢出。求和中的至少一项为1，因此总和至少为1，而 `log(1) = 0`。也不可能发生下溢到 `-inf`。

证明：

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

设置 `c = max(x)` 即可消除溢出问题。

这种技巧在机器学习中随处可见：
- Softmax归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合模型
- 变分推断

### 为什么Softmax需要最大值减法技巧

Softmax将logits转换为概率：

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Without the trick, logits of [100, 101, 102] cause overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

使用这个技巧，减去最大值 x 的 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

The probabilities are identical. The computation is safe. This is not an optimization. It is a requirement for correctness.

### NaN和Inf：检测与预防

`nan`（非数字）和`inf`（无穷大）会在计算过程中通过病毒式传播。梯度更新中的一个`nan`会使权重变为`nan`，从而导致所有后续输出也变为`nan`。训练在一步之内就会失败。

`inf`出现的情形：
- 大正数的`exp()`运算
- 除以零：`1.0 / 0.0`
- `float32`累加时的溢出

`nan`出现的情形：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的`sqrt()`运算
- 负数的`log()`运算
- 任何涉及现有`nan`的算术运算

检测：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

预防策略：

1. 将输入限制为 `exp()` 的输出：`exp(clamp(x, -80, 80))`
2. 在分母中添加epsilon：`x / (y + 1e-8)`
3. 在 `log()` 内部添加epsilon：`log(x + 1e-8)`
4. 使用稳定实现（如log-sum-exp、stable softmax）
5. 通过梯度裁剪防止权重爆炸
6. 在调试期间，每次前向传播后检查 `nan`/`inf`

### Numerical Gradient Checking

分析梯度（来自反向传播）可能存在错误。数值梯度检查通过使用有限差分来计算梯度来验证它们。

中心差分公式：

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

This method has an accuracy of O(h^2), which is much better than the forward difference `(f(x+h) - f(x)) / h`, which is only O(h).

Choosing the value of h: if it is too large, the approximation will be incorrect. If it is too small, catastrophic cancellation will destroy the result. The typical range for h is from 1e-5 to 1e-7.

The check: calculate the relative difference between the analytical and numerical gradients.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

经验法则：
- 相对误差 < 1e-7：完美，梯度正确
- 相对误差 < 1e-5：可接受，可能正确
- 相对误差 > 1e-3：有问题
- 相对误差 > 1：梯度完全错误

在实现新层或损失函数时，务必检查梯度。PyTorch提供了`torch.autograd.gradcheck()`用于此目的。

### 混合精度训练

现代GPU拥有专门的硬件（张量核心），其进行float16矩阵乘法的速度比float32快2到8倍。混合精度训练利用了这一特性：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

纯float16训练的问题在于：梯度通常非常小（1e-8或更小）。float16会将低于约6e-8的值归零。你的模型会停止学习，因为所有的梯度更新都为零。

解决办法是损失缩放：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

动态损失缩放会自动调整比例因子。初始值设为较大值（65536）。如果梯度溢出到`inf`，则将其减半。如果经过N步未发生溢出，则将其加倍。

### bfloat16 vs float16: Why bfloat16 is preferred for training

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16具有更高的精度（10位尾数比特，而float32为7位），但范围有限（最大约65,504）。bfloat16的精度较低，但其范围与float32相同（最大约3.4e38）。

对于训练神经网络：

- 在训练高峰期，激活值和逻辑值经常超过65,504。float16会发生溢出；bfloat16可以处理这种情况。
- 使用float16时需要损失缩放，但使用bfloat16通常不需要，因为其范围涵盖了梯度幅度的整个频谱。
- bfloat16是float32的简单截断版本：丢弃尾数的最后16位。转换过程简单且无损失。

当值有界且精度更为重要时，优选使用float16进行推理。当范围更为重要时，则优选使用bfloat16进行训练。这就是为什么TPU和现代NVIDIA GPU（A100、H100）具有bfloat16原生支持的原因。

### 梯度裁剪

爆炸式梯度发生在梯度通过多层呈指数增长时（常见于RNN、深度网络和Transformer模型中）。一个巨大的梯度可以在一步中破坏所有权重。

两种类型的裁剪：

**按值裁剪：**独立地限制每个梯度的元素。

```
grad = clamp(grad, -max_val, max_val)
```

简单但可以改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过阈值。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Preserves the direction of the gradient. This is what `torch.nn.utils.clip_grad_norm_()` does. It is the standard choice.

Typical values: `max_norm=1.0` for transformers, `max_norm=0.5` for RL, `max_norm=5.0` for simpler networks.

Gradient clipping is not a hack. It is a safety mechanism. Without it, a single outlier batch can produce a gradient large enough to ruin weeks of training.

### 规范化层作为数值稳定器

批量归一化、层归一化和RMS归一化通常被作为正则化器来使用，帮助训练收敛。它们也是数值稳定器。

如果没有归一化，激活值可能会通过层呈指数级增长或缩小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

在每一层对激活值进行标准化、重新定位和重新缩放：

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon`（通常为1e-5）可以防止在所有激活函数相同的情况下发生除以零的情况。学习到的参数`gamma`和`beta`使网络能够恢复所需的任何比例。

这确保了整个网络中的值都在数值安全的范围内，从而防止了前向传播中的溢出和后向传播中的梯度爆炸。

### 常见的机器学习数值错误

**Bug: Loss becomes NaN after a few epochs.**  
Cause: Logits become too large, causing softmax to overflow. Or the learning rate is too high, leading to weight divergence.  
Fix: Use stable softmax (max subtraction), reduce the learning rate, and add gradient clipping.  

**Bug: Loss remains at log(num_classes).**  
Cause: Model outputs are nearly uniform probabilities, indicating that gradients are zero or the model is not learning at all.  
Fix: Verify that data labels are correct, examine the loss function, and check for dead ReLU neurons.  

**Bug: Validation accuracy is 1-3% lower than expected.**  
Cause: Mixed precision without proper loss scaling causes small updates to be silently eliminated by underflow of gradients.  
Fix: Enable dynamic loss scaling or switch to bfloat16.  

**Bug: Gradient norms are 0.0 for some layers.**  
Cause: Dead ReLU neurons (all inputs negative) or underflow of float16.  
Fix: Use LeakyReLU or GELU, apply gradient scaling, and check weight initialization.  

**Bug: The model works on one GPU but produces different results on another.**  
Cause: Non-deterministic accumulation order of floating point operations. GPU parallel computations sum in different orders on different hardware, and floating point addition is non-associative.  
Fix: Accept small differences (1e-6) or set `torch.use_deterministic_algorithms(True)` to accept the speed penalty.  

**Bug: `exp()` returns `inf` in loss computation.**  
Cause: Raw logits are passed to `exp()` without the max-subtraction trick.  
Fix: Use `torch.nn.functional.log_softmax()`, which internally implements log-sum-exp.  

**Bug: Training diverges after switching from float32 to float16.**  
Cause: Float16 cannot represent gradient magnitudes below 6e-8 or activations above 65,504.  
Fix: Use mixed precision with loss scaling (AMP) or switch to bfloat16.

```figure
logsumexp-stability
```

## 构建它

### 步骤1：展示浮点数的精度限制

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Step 2: Implementing Naive vs Stable Softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 步骤3：实现稳定的对数和展开函数

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 步骤4：实现稳定的交叉熵损失函数

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步骤5：渐变检查

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 使用它

### 混合精度仿真

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### 梯度裁剪

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf detection

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

请参阅 `code/numerical.py`，其中展示了所有边缘情况的完整实现。

## 发货

本课程将生成以下文件：
- `code/numerical.py`，包含稳定的softmax、log-sum-exp、交叉熵、梯度检查以及混合精度模拟功能
- `outputs/prompt-numerical-debugger.md`，用于诊断训练过程中的NaN/Inf数值问题

这些稳定实现将在第三阶段构建训练循环时再次出现，在第四阶段实现注意力机制时也会使用到。

## 练习

1. **Catastrophic cancellation.** Calculate the variance of the range [1000007.0, 1000002.0, 1000003.0] using the naive formula `E[x^2] - E[x]^2` in float32 format. Then calculate it using Welford's online algorithm. Compare the error with the true variance (0.6667).

2. **Precision hunt.** Find the smallest positive float32 value `x` such that `1.0 + x == 1.0` in Python. This is known as the machine epsilon. Verify that it matches `numpy.finfo(numpy.float32).eps`.

3. **Log-sum-exp edge cases.** Test your `logsumexp_stable` function with: (a) all values being equal, (b) one value being much larger than the rest, and (c) all values being very negative (-1000). Verify that it provides correct results where the naive version fails.

4. **Gradient checking a neural network layer.** Implement a single linear layer `y = Wx + b` and its analytical backward pass. Use `numerical_gradient` to verify the correctness for a 3x2 weight matrix.

5. **Loss scaling experiment.** Simulate training with float16: generate random gradients in the range [1e-9, 1e-3], convert them to float16, and measure what fraction of the gradients become zero. Then apply loss scaling (multiply by 1024), convert back to float16, scale down again, and measure the new fraction of zero gradients.

## | 关键词 | 翻译 |

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------|
| IEEE 754 | “浮点标准” | 定义二进制浮点格式、舍入规则及特殊值（无穷大、NaN）的国际标准。所有现代CPU和GPU都支持它。 |
| 机器epsilon | “精度极限” | 在给定浮点格式下，使得1.0 + e != 1.0的最小e值。对于float32，约为1.19e-7。 |
| 灾难性抵消 | “减法导致的精度损失” | 当减去几乎相等的浮点数时，有效位数会被抵消，结果中的舍入噪声占主导。 |
| 溢出 | “数字太大” | 结果超过可表示的最大值并变为无穷大。exp(89)会使float32溢出。 |
| 下溢 | “数字太小” | 结果接近零而不是最小的可表示正数，变为0.0。exp(-104)会使float32下溢。 |
| Log-sum-exp技巧 | “先减去最大值” | 通过提取exp(max(x))来避免溢出和下溢，从而计算log(sum(exp(x))。用于softmax、交叉熵和概率对数运算。 |
| 稳定softmax | “不会爆炸的softmax” | 在指数化之前减去max(logits)。结果数值上相同，不可能发生溢出。 |
| 梯度检查 | “验证反向传播” | 通过有限差分法比较分析梯度与数值梯度，以发现实现中的错误。 |
| 混合精度 | “用float16进行前向计算，用float32进行反向计算” | 在需要高速运算时使用低精度浮点数，在数值敏感操作中使用高精度浮点数。通常加速2-3倍。 |
| 损失缩放 | “防止梯度下溢” | 在反向传播前将损失乘以大常数，使梯度保持在float16的可表示范围内，然后在权重更新前除以同一常数。 |
| bfloat16 | “大脑浮点” | Google的16位格式，包含8个指数位（与float32相同范围）和7个尾数位（精度低于float16）。适用于训练。 |
| 梯度裁剪 | “限制梯度范数” | 调整梯度向量的范数不超过阈值，防止梯度爆炸破坏权重。 |
| NaN | “非数字” | 来自未定义运算的特殊浮点值（0/0、inf-inf、sqrt(-1)）。会传播到所有后续算术运算中。 |
| 无穷大 | “Infinity” | 来自溢出或除以零的特殊浮点值。可以组合产生NaN（inf - inf、inf * 0）。 |
| 数值梯度 | “暴力导数” | 通过评估f(x+h)和f(x-h)并除以2h来近似导数。虽然缓慢但可靠用于验证。 |

## 更多阅读资料

- [Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- the definitive reference, dense but complete
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- the NVIDIA paper that introduced loss scaling for float16 training
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- practical guide to mixed precision in PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- why Google chose this format for TPUs
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- algorithm for reducing rounding error in floating point sums
