# Introduction to JAX

> PyTorch用于修改张量。TensorFlow用于构建图。JAX则编译纯函数。最后这一点改变了你对深度学习的看法。

**类型：**构建
**语言：**Python
**先决条件：**第03阶段课程01-10，基本的NumPy知识
**时间：**约90分钟

## 学习目标

- Write pure-function neural network code using JAX's functional API (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Explain the key design difference between PyTorch's eager mutation and JAX's functional compilation model
- Apply jit compilation and vmap vectorization to accelerate training loops compared to naive Python
- Train a simple network in JAX and contrast the explicit state management with PyTorch's object-oriented approach

## 问题

您知道如何在 PyTorch 中构建神经网络。您可以定义 `nn.Module`，调用 `.backward()`，调整优化器。这一切都运作良好，有数百万人使用它。

但 PyTorch 的DNA中内置了一个限制：它在 Python 中一次只追踪一个操作。每个 `tensor + tensor` 都是独立的内核启动。每次训练步骤都会重新解释相同的 Python 代码。这看起来没问题，直到您需要用 2,048 个 TPU 训练一个拥有 5400 亿参数的模型。这时，这种开销就会让您崩溃。

Google DeepMind 使用 JAX 训练 Gemini。Anthropic 也使用 JAX 训练 Claude。这些并不是小型操作——它们是地球上最大的神经网络训练任务。他们选择 JAX 是因为它将训练循环视为可编译的程序，而不是一系列 Python 调用。

JAX 是 NumPy 的增强版，拥有三个强大的功能：自动微分、JIT 编译到 XLA 以及自动向量化。您编写一个处理单个示例的函数。JAX 则为您提供了一个处理批次的函数，计算梯度，编译为机器代码，并在多个设备上运行。所有这些都不需要修改原始函数。

## 概念

### JAX哲学

JAX是一个函数式框架。没有类，没有可变状态，也没有`.backward()`方法。相反：

| PyTorch | JAX |
|---------|-----|
| 带有状态的`nn.Module`类 | 纯函数：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| 急切执行 | 通过XLA进行JIT编译 |
| `for x in batch:`手动循环 | `jax.vmap(f)`自动向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)`自动并行化 |
| 可变的`model.parameters()` | 不可变的数组结构 |

这不是风格偏好，而是编译器约束。JIT编译要求使用纯函数——相同的输入总是产生相同的输出，没有副作用。这一限制使得速度提升可达100倍成为可能。

### jax.numpy: The Familiar Surface

JAX重新实现了NumPy API在加速器上的使用：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函数名称。相同的广播规则。相同的切片语义。但是这些数组位于GPU/TPU上，且每个操作都可以被编译器追踪。

一个关键的区别是：JAX数组是不可变的。不能执行`a[0] = 5`。相反，应该执行`a = a.at[0].set(5)`。一开始可能会觉得不太顺手，但之后就会习惯——不可变性使得像`grad`、`jit`和`vmap`这样的转换可以组合使用。

### jax.grad: Functional Autodiff

PyTorch attaches gradients to tensors (`.grad`). JAX attaches gradients to functions.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` takes a function and returns a new function that computes the gradient. No `.backward()` call is required. No computation graph is stored on tensors. The gradient is simply another function that can be called, composed with other functions, or JIT-compiled.

This functionality allows for arbitrary composition:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二阶导数。三阶导数。雅可比矩阵。赫斯矩阵。所有这些都可以通过调用 `grad` 来实现。PyTorch 也可以实现这一点（`torch.autograd.functional.hessian`），但它是附加的。在 JAX 中，这是基础。

约束条件：`grad` 仅适用于纯函数。内部不能包含打印语句（它们在追踪过程中运行，而不是执行时）。不得改变外部状态。未经明确管理的情况下不得生成随机数。

### jit: Compiled into XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

在第一次调用时，JAX会追踪函数执行过程——它记录发生的操作，但不执行这些操作。然后，它将这一追踪结果传递给XLA（加速线性代数），这是Google为TPU和GPU设计的编译器。XLA将操作融合在一起，消除多余的内存复制，并生成优化的机器代码。

后续调用完全跳过Python部分。编译后的代码在加速器上以C++的速度运行。

当JIT有帮助时：
- 训练步骤（相同的计算重复数千次）
- 推理（相同模型，不同输入）
- 任何多次调用的具有相似输入形状的函数

当JIT有害时：
- 依赖值的Python控制流函数（`if x > 0`，其中x是被追踪的数组）
- 一次性计算（编译开销超过运行时间）
- 调试（追踪隐藏了实际执行过程）

控制流限制是真实的。`jax.lax.cond`取代了`if/else`。`jax.lax.scan`取代了`for`循环。这些不是可选的——它们是编译的代价。

### vmap：自动向量化

您编写一个处理一个示例的函数：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap`用于批量处理：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`inaxes=(None, 0)` 的含义是：不要对 `params`（共享）进行批量处理，而是对 `x` 的轴 0 进行批量处理。无需手动使用 `for` 循环。不进行重塑。也不进行批量维度线程化。JAX 会自动确定批量维度并将整个计算向量化。

这不是语法糖。`vmap` 生成的向量化代码运行速度比 Python 循环快 10-100 倍。而且它与 `jit` 和 `grad` 结合使用：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

按示例梯度。一行代码。在PyTorch中，没有特殊技巧的话这几乎是不可能的。

### pmap：设备间的数据并行处理

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 会在所有可用的设备（GPU/TPU）上复制函数，并分割批次。在函数中，`jax.lax.pmean` 和 `jax.lax.psum` 会同步各设备的梯度。

谷歌使用 `pmap`（及其继任者 `shard_map`）在数千个 TPU v5e 芯片上训练 Gemini。编程模型如下：编写单设备版本，然后使用 `pmap` 进行分割。

### Pytrees：通用数据结构

JAX operates on “pytrees” – nested combinations of lists, tuples, dictionaries, and arrays. Your model parameters are a pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Each JAX transformation – `grad`, `jit`, `vmap` – knows how to traverse pytrees. `jax_tree.map(f, tree)` applies `f` to every leaf. This is how optimizers update all parameters at once:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

没有`.parameters()`方法。没有参数注册。树结构就是模型。

### 函数式与面向对象

PyTorch stores state within objects:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX使用具有明确状态的纯函数：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

参数已传入。没有数据进行存储，也没有数据发生修改。这使得每个函数都具备可测试性、可组合性和可编译性。这也意味着你需要自己管理参数——或者使用像Flax或Equinox这样的库。

### JAX Ecosystem

JAX提供基础接口。库则提供易用性：

| 库 | 角色 | 风格 |
|----|------|-------|
| **Flax** (Google) | 神经网络层 | 使用`nn.Module`并明确状态 |
| **Equinox** (Patrick Kidger) | 神经网络层 | 基于Pytree，符合Python风格 |
| **Optax** (DeepMind) | 优化器+LR调度 | 可组合的梯度变换 |
| **Orbax** (Google) | 检查点保存 | 保存/恢复pytrees |
| **CLU** (Google) | 指标与日志 | 训练循环工具 |

Optax是标准优化器库。它将梯度变换（Adam、SGD、裁剪）与参数更新分离，使得组合使用非常简单：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### When to Use JAX vs PyTorch

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU支持 | 一流（Google自行构建） | 社区维护（torch_xla） |
| GPU支持 | 良好（通过XLA使用CUDA） | 顶尖水平（原生CUDA） |
| 调试 | 困难（追踪+编译） | 简单（即时，逐行处理） |
| 生态系统 | 以研究为重点（Flax, Equinox） | 庞大（HuggingFace, torchvision等） |
| 招聘 | 小众领域（Google/DeepMind/Anthropic） | 主流领域（ everywhere） |
| 大规模训练 | 优越（XLA, pmap, mesh） | 良好（FSDP, DeepSpeed） |
| 原型开发速度 | 较慢（功能开销较大） | 更快（即时生成） |
| 生产环境推理 | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| 使用者 | DeepMind（Gemini），Anthropic（Claude） | Meta（Llama），OpenAI（GPT），Stability AI |

诚实的回答是：除非有特定原因需要使用JAX，否则应使用PyTorch。这些原因包括：TPU访问、需要每个示例的梯度、大规模多设备训练，或在Google/DeepMind/Anthropic工作。

### JAX中的随机数

JAX没有全局随机状态。每个随机操作都需要一个明确的PRNG密钥：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

At first, this may seem annoying. However, it ensures reproducibility across different devices and compilation environments—a feature that PyTorch’s `torch.manual_seed` cannot guarantee in multi-GPU settings.

```figure
batchnorm-effect
```

## 构建它

### 步骤1：设置和数据

我们将使用JAX和Optax在MNIST上训练一个三层MLP。输入为784个神经元，两个隐藏层分别有256和128个神经元，输出类别为10个。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步骤2：初始化参数

无课程。只是一个返回 pytree 的函数：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

手动进行初始化。从同一个种子生成三个伪随机数生成器密钥。每个权重都是一个不可变的数组，位于嵌套字典中。

### 步骤3：前传

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

纯函数。参数输入，预测输出。没有`self`，也没有存储的状态。`loss_fn`从零开始计算交叉熵——使用softmax、对数和负均值。

### 步骤4：即时编译培训步骤

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and Grad` 在同一次调用中返回损失值和梯度。`@jax.jit` 装饰器将这两个函数编译为 XLA 格式。第一次调用后，每次训练步骤的执行都不会涉及 Python。

### 步骤5：训练循环

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 epochs. Test accuracy of ~97%. The first epoch is slow due to JIT compilation. Epochs 2-10 are fast.

Note what is missing: no `.zero_grad()`, no `.backward()`, no `.step()`. The entire update process is a single function call. Gradients are calculated, transformed using Adam, and applied to parameters—all within `train_step`.

## 使用它

### 亚麻：谷歌标准

Flax是JAX神经网络库中最常用的一种。它添加了`nn.Module`，但具有明确的状态管理机制：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

结构与PyTorch相同，但`params`与模型分离。`model.init()`用于创建`params`。`model.apply(params, x)`执行前向传播。模型对象没有状态。

### Equinox：Python风格的替代方案

Equinox (by Patrick Kidger) represents models as pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

The model itself is a pytree. No `.apply()` is required. The parameters are simply the model’s leaves. This is more in line with how JAX operates.

### Optax: 可组合优化器

Optax separates the gradient transformation from the update process:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

梯度裁剪、学习率预热、权重衰减——所有这些都构成了一个转换链。每个转换都会处理梯度，对其进行修改，并将其传递给下一个转换。没有单一的优化器类。

## 发货

**安装：**

```bash
pip install jax jaxlib optax flax
```

支持GPU：

```bash
pip install jax[cuda12]
```

针对TPU（谷歌云）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**性能注意事项：**

- 第一次JIT调用速度较慢（编译过程）。在进行基准测试之前进行预热。
- 在JIT环境中，避免对JAX数组使用Python循环。应使用`jax.lax.scan`或`jax.lax.fori_loop`。
- `jax.debug.print()`可以在JIT环境下使用，而普通的`print()`则不行。
- 可以使用`jax.profiler`或TensorBoard进行性能分析。XLA编译可以揭示瓶颈。
- JAX默认预分配75%的GPU内存。设置`XLA_PYTHON_CLIENT_PREALLOCATE=false`以禁用此功能。

**检查点：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

本课程将生成以下文件：
- `outputs/prompt-jax-optimizer.md` —— 用于选择合适JAX优化器配置的提示词
- `outputs/skill-jax-patterns.md` —— 关于JAX中函数式模式的技能文档

## 练习

1. Add dropout to the MLP. In JAX, dropout requires a PRNG key -- thread a key through the forward pass and split it for each dropout layer. Compare test accuracy with and without.

2. Use `jax.vmap` to compute per-example gradients for a batch of 32 MNIST images. Compute the gradient norm for each example. Which examples have the largest gradients, and why?

3. Replace the manual forward function with a generic `mlp_forward(params, x)` that works for any number of layers. Use `jax.tree.leaves` to determine the depth automatically.

4. Benchmark the training step with and without `@jax.jit`. Time 100 steps of each. How large is the speedup on your hardware? What is the compilation overhead on the first call?

5. Implement gradient clipping by composing `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Train with and without clipping. Plot the gradient norm over training to see the effect.

## | 关键词 | 翻译 |

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|----------------|
| XLA | “让JAX变得快速的工具” | 加速线性代数——一种将操作融合并从计算图中生成优化后的GPU/TPU内核的编译器 |
| JIT | “即时编译” | JAX在首次调用时追踪函数，将其编译为XLA，然后在后续调用中运行已编译版本 |
| 纯函数 | “无副作用” | 输出仅依赖于输入的函数——没有全局状态，没有变异，没有无需明确键的随机性 |
| vmap | “自动批处理” | 将处理单个示例的函数转换为处理批次的函数，无需重写 |
| pmap | “自动并行化” | 在多个设备上复制函数并分割输入批次 |
| Pytree | “嵌套数组字典” | 任何JAX可以遍历和转换的嵌套列表、元组、字典和数组结构 |
| 追踪 | “记录计算过程” | JAX使用抽象值执行函数以构建计算图，而不计算实际结果 |
| 函数自动微分 | “函数的梯度” | 通过转换函数来计算导数，而不是将梯度存储附加到张量上 |
| Optax | “JAX的优化器库” | 可组合的梯度变换库——Adam、SGD、裁剪、调度——将它们串联起来 |
| Flax | “JAX的nn.Module” | Google为JAX提供的神经网络库，在保持状态明确性的同时添加层抽象 |

## 更多阅读资料

- JAX文档：https://jax.readthedocs.io/ -- 官方文档，包含关于grad、jit和vmap的优秀教程
- “JAX：Python+NumPy程序的可组合转换”（Bradbury等，2018）-- 解释设计哲学的原始论文
- Flax文档：https://flax.readthedocs.io/ -- Google用于JAX的神经网络库
- Patrick Kidger，“Equinox：通过可调用PyTrees和过滤转换实现JAX中的神经网络”（2021）-- Flax的Python风格替代方案
- DeepMind，“Optax：可组合梯度转换和优化”-- 标准优化器库
- “你不知道JAX”（Colin Raffel，2020）-- 来自T5作者之一的关于JAX常见问题和模式的实用指南
