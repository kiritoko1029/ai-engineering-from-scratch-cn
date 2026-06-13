# 梯度检查点与激活重计算

> 反向传播保留每个中间激活。在 70B 参数和 128K 上下文下，每 rank 3 TB 激活。检查点用 FLOPs 换内存：重算而非保存。问题是丢弃哪些段，答案不是"全部"。

**类型：** 构建
**语言：** Python（使用 numpy，可选 torch）
**前置要求：** 第 10 阶段课程 04（预训练 Mini-GPT），第 10 阶段课程 05（扩展与分布式）
**所需时间：** 约70分钟

## 问题所在

训练 Transformer 时，为每一层存储反向传播中每个被微分操作的输入：注意力输入、Q/K/V 投影、softmax 输出、FFN 输入、归一化输出和残差流。对于隐藏大小 `d`、序列长度 `L`、批大小 `B` 的层，每层约 `12 * B * L * d` 个浮点数。

对于 `d=8192, L=8192, B=1`，BF16 下每层 800 MB。64 层模型是 51 GB 激活——这还没乘以微批次大小，还没加注意力-softmax 中间结果（每头 `L^2`），还没算张量并行的部分副本。

两面账：BF16 权重加优化器状态可能放进 80GB，但激活把你推过去了。梯度检查点（又名激活重计算）是标准修复。丢弃大部分激活；在反向期间重做前向来找回它们。代价：额外 FLOPs。收益：内存按检查点段与总层数的比率下降。

朴素做的话，检查点每步多花约 33% 前向 FLOPs。做得好——按 Korthikanti 等人的"智能选择"做选择性检查点——你用不到 5% FLOP 开销节省 5 倍内存。配合 FP8 矩阵乘法、FSDP offload 和专家并行 MoE，这真的重要：你既负担不起内存也负担不起浪费的计算。

## 概念说明

### 反向传播实际需要什么

`output = layer(input)`。反向需要 `grad_input` 和 `grad_params`。要计算它们需要：

- `input`（用于计算线性层的 `grad_params = input.T @ grad_output`）
- 一些激活导数中间结果（ReLU/GELU/softmax 的导数依赖激活值）

前向传播在自动求导图中自动存储这些。每个 `tensor.retain_grad()` 和每个需要其输入的操作都保留引用。

### 朴素全量检查点

将网络分成 N 个段。前向期间只存储每个段的*输入*。当反向需要中间结果时，重跑该段的前向传播来物化它们，然后微分。

示例：32 层 Transformer 分成 32 个段，每段 1 层。

- 内存：32 个层输入（小）vs 32 x（每层激活量）（巨大）。
- 额外计算：每段 1 次额外前向，即总共约 33% 更多前向 FLOPs（因为反向是前向的 2 倍，完整步骤从 1 + 2 = 3 变为 1 + 1 + 2 = 4）。

这是 Chen 等 2016 的原始配方：每 `sqrt(L)` 层一个检查点以平衡内存和计算。L=64 时是 8 个检查点。

### 选择性检查点（Korthikanti 2022）

并非所有激活成本相同。注意力 softmax 输出是 `B*L*L*heads`，随序列长度*二次*增长。FFN 隐藏激活是 `B*L*4d`，线性增长。长序列下 softmax 主导。

选择性检查点保留存储成本低的激活（线性投影、残差），只重算昂贵的（注意力）。你付最小 FLOPs 重算但节省 O(L^2) 内存。

Megatron-Core 实现此为"选择性"激活重计算。用于大多数 2024+ 前沿训练。

### Offload

重算的替代：在前向和反向之间将激活送到 CPU RAM。需要 PCIe 带宽；当空闲带宽超过重物化的成本时有益。混合策略常见：检查点某些层，offload 其他。

FSDP2 将 offload 作为一等选项。当 GPU 内存受限但 CPU-GPU 传输有余量时，offload 闪光。

### 重算成本模型

在 L 层中每 k 层朴素检查点的每步 FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

选择性检查点只重算注意力 kernel，不重算整层：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 内存节省模型

每层激活量：A。L 层总激活内存：L * A。

全量检查点（段大小 1）：只存储 L 个 input_volume（标准 Transformer 约 L * 1/10 A）。节省约 9 * L * A * 1/10。

每 k 层检查点：存储 L/k * A 加活跃段内 k-1 层的量。

在 k = sqrt(L) 时，内存和重算成本都按 sqrt(L) 缩放——均匀成本层的最优权衡。

### 何时不检查点

- 已在执行中的流水线阶段的最内层。反正它们要完成。
- 第一层和最后一层如果主导阶段计算（在 Transformer 中少见）。
- 已使用 FlashAttention 的注意力 kernel——Flash 已经快速重算 softmax，额外的层级检查点在此之上增加很少。

### 实现模式

1. **函数包装器：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包装段。PyTorch 只存储 `input`，反向时重算其余一切。

2. **基于装饰器：** 标记层为可检查点；训练器在配置时决定哪些段被包装。

3. **手动显式重算：** 自己写反向传播，调用用存储输入复制前向的自定义 `recompute_forward`。

三种给出相同的功能结果。包装器是标准惯用法。

### 与 TP / PP / FP8 的交互

- **张量并行：** 检查点输入在重算时需要被 gather 或重新 scatter；处理通信成本。
- **流水线并行：** 典型模式是检查点每个流水线阶段的前向，使反序微批次可以复用激活内存。
- **FP8 重算：** 重算期间更新的 amax 历史必须与原始前向匹配，否则 FP8 缩放会漂移。大多数框架快照缩放因子。

## 开始构建

### 第 1 步：带段的玩具模型

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 第 2 步：需要所有激活的朴素反向

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 第 3 步：每 k 层检查点内存

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 第 4 步：成本模型

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 第 5 步：内存估算器

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 第 6 步：最优段大小

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 第 7 步：选择性检查点决策

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 使用它

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`——PyTorch 的标准包装器。包装函数；只存储输入，反向时重算。
- **Megatron-Core 激活重计算**：支持 `selective`、`full` 和 `block` 模式。2024+ 前沿训练标准。
- **FSDP2 offload**：`module.to_empty(device="cpu")` 配合 FSDP2 中的 `offload_policy` 将激活 offload 到 CPU 而非重算。
- **DeepSpeed ZeRO-Offload**：优化器状态和激活的 CPU offload，补充检查点。

## 交付它

本课程产出 `outputs/prompt-activation-recompute-policy.md`——一个接收你的模型配置（层、隐藏、序列、批大小）和可用 GPU 内存并输出逐层重算策略（无 / 选择性 / 全量 / offload）的提示。

## 练习

1. 验证正确性。运行 `model_forward` + `model_backward`（全激活）vs `model_forward_checkpointed` + `model_backward_checkpointed`（段）。参数梯度必须在机器精度内相同。

2. 将段大小 `k` 从 1 扫描到 `L`。绘制 FLOP 开销和内存。找到曲线的拐点。

3. 实现选择性检查点：存储注意力模块输入但不存储其中间结果。在 32 层模型 seq=8192 下度量 vs 全层检查点的 FLOP 开销。

4. 添加 offload。将段输入保存到模拟的"CPU 缓冲区"（单独列表）。度量"PCIe 带宽"为字节/时间，找到 offload 和重算之间的收支平衡点。

5. 在有和没有 `torch.utils.checkpoint` 的情况下基准测试真实 PyTorch Transformer。度量内存（通过 `torch.cuda.max_memory_allocated`）和步时间。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 梯度检查点 | "通过重做前向节省内存" | 只存储段输入；反向时重算中间结果以获得梯度支持张量 |
| 激活重计算 | "与检查点相同" | HPC 风格的同一技术名称 |
| 段大小（k） | "每检查点多少层" | 其中间结果被丢弃并一起重物化的层数 |
| 选择性检查点 | "Korthikanti 的技巧" | 只重算存储昂贵的激活（注意力 softmax）；保留便宜的 |
| 全量检查点 | "朴素版本" | 重算每段每层的中间结果 |
| 块检查点 | "粗粒度" | 检查点整个 Transformer 块；最大粒度 |
| FLOP 开销 | "计算税" | 每步额外 FLOPs =（重算 FLOPs）/（前向 + 反向 FLOPs）；朴素 33%，选择性 5% |
| 激活 offload | "送到 CPU" | 在前向->反向之间将激活移到 CPU RAM；重算的替代 |
| sqrt-L 规则 | "经典最优" | 对于均匀成本层，最优检查点间距是 sqrt(L) 层 |
| 注意力-softmax 体积 | "O(L^2) 问题" | L^2 * heads * batch 浮点数；长上下文中主导激活内存 |

## 延伸阅读

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- 形式化梯度检查点的原始论文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- 选择性激活重计算和形式化成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- 通过反向模式重物化的替代恒定内存方案
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- 大规模激活 offload
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- 标准 API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- 选择性、全量和块模式
