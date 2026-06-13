# KV缓存、Flash Attention与推理优化

> 训练过程为并行执行，受浮点运算次数限制。推理过程为串行执行，受内存限制。瓶颈不同，应对策略也不同。

**类型：** 构建
**语言：** Python
**先修知识：** 第 7 阶段 · 02（自注意力机制）、第 7 阶段 · 05（完整 Transformer 模型）、第 7 阶段 · 07（GPT）
**时长：** 约 75 分钟

## 问题所在

一个简单的自回归解码器在生成 `N` 个标记时需要执行 `O(N²)` 复杂度的计算：在每一步中，它都需要重新对整个前缀序列计算注意力权重。对于长度为 4K 标记的响应而言，这将产生 1600 万次注意力运算，其中大部分都是冗余的。一旦计算出某个前缀标记的所有隐藏状态，这些状态就是确定的——此时只需将新标记的查询向量与之前缓存的所有键值进行匹配即可。

除此之外，注意力机制本身也会涉及大量数据的传输。传统的注意力算法会生成一个 N×N 的得分矩阵、一个 N×d 维度的 softmax 输出以及一个 N×d 维度的最终输出，这会导致对高带宽内存（HBM）进行过多的读写操作。当 `N` 大于 2K 时，计算瓶颈往往首先出现在内存访问上，而非浮点运算能力上。传统的注意力内核的利用率仅为现代 GPU 的 4 到 10 倍。

Dao 等人提出的两项优化技术将推理速度从“缓慢”提升到了“极快”：

1. **KV 缓存**：存储每个前缀标记的 K 向量和 V 向量。对于每一个新标记，只需将其查询向量与缓存中的键进行一次匹配即可。这样一来，每生成一个标记所需的计算复杂度就从 `O(N²)` 降到了 `O(N)`。
2. **Flash Attention**：通过将注意力计算分解为多个小块，确保完整的 N×N 矩阵永远不会被加载到 HBM 中。所有的 softmax 计算和矩阵乘法都在静态随机存取存储器（SRAM）中完成。在 A100 上可实现 2 到 4 倍的速度提升；在使用 FP8 编码的 H100 上则可实现 5 到 10 倍的提升。

到 2026 年，这两种技术已变得普遍应用。所有的生产级推理框架（如 vLLM、TensorRT-LLM、SGLang、llama.cpp）都默认支持它们。而且，所有前沿模型在发布时也都启用了 Flash Attention 功能。

## 概念概述

![KV缓存增长与Flash Attention分块示意图](../assets/kv-cache-flash-attn.svg)

### KV 缓存数学原理

每个解码器层、每个令牌、每个头：

```
bytes_per_token_per_layer = 2 * d_head * dtype_size
                          ^
                          K and V
```

对于一个具有32层、32个注意力头，d_head=128，并采用fp16精度的70亿参数模型：

```
per token per layer = 2 * 128 * 2 = 512 bytes
per token (32 layers) = 16 KB
per 32K context = 512 MB
```

针对 Llama 3 70B（80 层，d_head=128，带有 8 个 KV 头的 GQA）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KB)
per 32K context = 10.4 GB
```

正是这 10 GB 的原因，导致在批次大小为 1 时，Llama 3 70B 模型在 128K 上下文长度下的 KV 缓存就需要占用几乎整个 40 GB 的 A100 显卡。

**GQA 模型在 KV 缓存方面具有优势。** 使用 64 个注意力头的 MHA 模型的 KV 缓存需求则为 32 GB。而 MLA 模型还能进一步压缩缓存大小。

调整模型维度，观察缓存大小的变动。增加序列长度或批次大小，看看其消耗的显存速度会多快超出单张 GPU 的承载能力：

```figure
kv-cache-sizer
```

### Flash Attention——分块技巧

标准注意力机制：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次HBM数据往返传输。在H100上，HBM的带宽为3 TB/s，而SRAM的带宽则为30 TB/s。与将所有数据保留在芯片内部相比，每次HBM数据传输都会导致速度下降10倍。  

闪存注意力机制：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每个 tile 执行一次 HBM 访问。总内存占用从 `O(N²)` 降至 `O(N)`。反向传播会重新计算前向传播中的一些数值而非将其存储起来——这进一步节省了内存。

**数值技巧。** 运行 softmax 时会跨 tile 维护 `(max, sum)` 值，从而实现精确的最终归一化。这不是近似值——在忽略 fp16 的非结合性之外，Flash Attention 能够生成与标准注意力机制完全相同的输出。

**版本演进：**

| 版本 | 年份 | 主要变更 | 在参考硬件上的加速比 |
|---------|------|-----------|------------------------|
| Flash 1 | 2022 | 分块 SRAM 核心 | A100 上提升 2 倍 |
| Flash 2 | 2023 | 更佳的并行性及因果优先排序 | A100 上提升 3 倍 |
| Flash 3 | 2024 | Hopper 异步机制及 FP8 支持 | H100 上提升 1.5–2 倍（约 740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 五阶段流水线及软件 exp2 算法 | 仅支持推理（初始阶段仅为前向传播） |

Flash 4 发布时仅支持前向传播。训练仍使用 Flash 3。Flash 4 对 GQA 和 varlen 的支持预计在 2026 年中期推出。

### 推测性解码——另一项降低延迟的策略

低成本模型会生成 N 个令牌。大型模型则并行验证这 N 个令牌中的所有内容。如果验证通过 k 个令牌，那么只需为这 k 次生成支付一次大型模型的前向传播费用。在代码和散文处理中，典型的 k 值为 3–5。

2026 年的默认配置：
- **EAGLE 2 / Medusa**：集成式草稿头，可与验证器的隐藏状态共享信息，在不损失质量的前提下可实现 2–3 倍的速度提升。
- **结合草稿模型的推测解码**：在消费级硬件上可实现 2–4 倍的速度提升。
- **前瞻解码**：采用雅可比迭代法，无需草稿模型。虽应用范围较窄，但完全免费。

### 连续批处理

经典的批量推理方式：需等待最慢的序列处理完成后再启动新批次。当短响应提前处理完毕时，会浪费 GPU 资源。

连续批处理（最初在 Orca 中推出，现亦应用于 vLLM、TensorRT-LLM、SGLang）：一旦旧请求处理完成，便立即将新请求加入批次中。对于典型的聊天类任务，其吞吐量可提升 5–10 倍。

### PagedAttention — 以虚拟内存形式实现的KV缓存

vLLM的核心特性。KV缓存以16个标记为一个块进行分配；页表则负责将逻辑地址映射到物理块上。该机制支持在并行样本（如束搜索、并行采样）之间共享KV数据，可快速更换前缀以实现提示词缓存，并能对内存进行碎片整理。相较于简单的连续内存分配方式，其处理效率提升了4倍。

```figure
flash-attention-memory
```

## 构建它

请参阅 `code/main.py`。我们实现了以下内容：

1. 一种简单的、时间复杂度为 `O(N²)` 的增量解码器。
2. 一种基于 KV 缓存的、时间复杂度为 `O(N)` 的解码器。
3. 一种分块 softmax 算法，用于模拟 Flash Attention 中的 running-max 算法。

### 步骤 1：KV 缓存

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

简单方法：在每一层、每一个头的列表中持续扩展按令牌计数的K和V向量。

### 步骤 2：分块软最大值算法

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

可在单次计算中输出与 `softmax(qK) V` 完全相同的结果，但任何时刻的工作集均为 `tile × d_head` 大小的块，而非完整的 `N × d_head` 大小。

### 步骤 3：在生成 100 个令牌的测试中，对比朴素解码与缓存解码的性能。

统计注意力操作次数。朴素方法：`O(N²)`，结果为 5050 次；缓存方法：`O(N)`，结果为 100 次。代码会输出这两种方法的计算结果。

## 使用它

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 生产环境部署：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨请求的前缀缓存是 2026 年的一项重大突破——相同的系统提示、少量示例或长上下文文档都可以在多次调用之间复用键值对。对于需要重复发送工具提示的智能体任务，前缀缓存通常能带来 5 倍的吞吐量提升。

## 发布它

请参阅 `outputs/skill-inference-optimizer.md`。该技能会为新的推理部署选择注意力机制实现、KV缓存策略、量化方式以及推测性解码技术。

## 练习题

1. **简单。** 运行 `code/main.py`，验证朴素解码器和缓存解码器是否产生相同输出；注意操作次数的差异。
2. **中等难度。** 实现前缀缓存机制：给定一个提示词 P 和若干个补全内容，先对 P 执行一次前向传播以填充 KV 缓存，随后针对每个补全内容分别进行分支处理。测量与重新编码 P 相比的加速效果。
3. **高难度。** 实现一个简化的分页注意力机制：将 KV 缓存划分为大小固定的 16 个令牌块，并使用自由列表管理这些块。当某个序列处理完毕后，将其块归还到池中。模拟长度各异的 1,000 次聊天补全场景，比较碎片化内存分配与连续内存分配的差异。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| KV缓存 | “让解码变快的技巧” | 存储每个前缀令牌的K和V值；新查询可直接调用这些值，无需重新计算。 |
| HBM | “GPU主内存” | 高带宽内存；H100型号为80 GB，B200型号为192 GB。带宽约为3 TB/s。 |
| SRAM | “片上内存” | 用于每个SM的快速内存，H100上每个SM约256 KB。带宽约为30 TB/s。 |
| Flash Attention | “分块注意力核” | 在不将N×N矩阵加载到HBM中的情况下计算注意力。 |
| 连续批处理 | “无等待批处理” | 将已处理的序列移出，再将新序列加入，且不会耗尽当前批次。 |
| PagedAttention | “vLLM的亮点技术” | 通过页表将KV缓存划分为固定块，从而消除碎片化问题。 |
| 前缀缓存 | “复用长提示词” | 为多个请求中共用的前缀缓存K和V值；可大幅降低智能体的计算成本。 |
| 推测解码 | “先草拟再验证” | 先使用低成本模型生成令牌草案，再由大模型一次性完成验证。 |

## 延伸阅读

- [Dao 等人 (2022). FlashAttention：具备 I/O 感知能力的快速且节省内存的精确注意力机制](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2：具有更好并行性与工作划分能力的更快注意力机制](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah 等人 (2024). FlashAttention-3：支持异步处理与低精度计算的快速且精确的注意力机制](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 发布说明（Dao-AILab，2026）](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 阶段流水线及 software-exp2 技巧；请阅读仓库的 README 以了解本课程提及的仅支持前向计算的限制。
- [Kwon 等人 (2023). 基于 PagedAttention 的大型语言模型高效内存管理方法](https://arxiv.org/abs/2309.06180) — vLLM 论文。
- [Leviathan 等人 (2023). 通过推测解码实现 Transformer 的快速推理](https://arxiv.org/abs/2211.17192) — 推测解码技术。
- [Li 等人 (2024). EAGLE：推测采样需要重新思考特征不确定性问题](https://arxiv.org/abs/2401.15077) — 本课程所引用的集成草稿方法的 EAGLE-1/2 论文。
- [Cai 等人 (2024). Medusa：具备多个解码头的简单大型语言模型推理加速框架](https://arxiv.org/abs/2401.10774) — 与 EAGLE 并列提及的 Medusa 方法。
- [vLLM 文档 — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 关于 16 个标记块及页表设计的权威深度解析。
