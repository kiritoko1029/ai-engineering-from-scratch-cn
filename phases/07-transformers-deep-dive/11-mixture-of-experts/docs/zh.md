# 专家混合模型（MoE）

> 高密度的 70B 参数Transformer模型会为每个标记激活所有参数。而 671B 参数的MoE模型仅需为每个标记激活 37B 参数，且在各项基准测试中均优于前者。稀疏性是本十年最重要的扩展思路。

**类型：** 构建
**语言：** Python
**先修课程：** 第7阶段 · 05（完整Transformer模型），第7阶段 · 07（GPT模型）
**耗时：** 约45分钟

## 问题所在

在推理阶段，密集型 Transformer 的 FLOPs 数值等于其参数总量（前向传播时还需再乘以 2）。当扩大密集型模型的规模时，每个 token 都需要承担全部的计算开销。到 2024 年，这一技术发展已遭遇计算瓶颈：若要实现显著的性能提升，就需要为每个 token 提供呈指数级增长的 FLOPs。

专家混合架构打破了这种关联。该架构用 `E` 个独立的专家网络替换原有的每个 FFN，并加入一个路由器来为每个 token 选择 `k` 个专家。总参数量为 `E × FFN_size`，而每个 token 的活跃参数量为 `k × FFN_size`。2026 年常见的配置为 `E=256`、`k=8`。模型的存储需求随 `E` 增长，而计算需求则随 `k` 增长。

2026 年的顶尖模型几乎全是专家混合架构：DeepSeek-V3（总参数量 671B，活跃参数量 37B）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的独立排行榜上，排名前 10 的开源模型也全部属于专家混合架构。

## 概念概述

![MoE层：路由器为每个标记选择k个专家模型](../assets/moe.svg)

### FFN 交换操作

密集Transformer块：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE 块：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每位专家都是一个独立的FFN（通常为SwiGLU）。路由器为一个单层线性结构。每个令牌会自行选择`k`位专家，并获得这些专家输出经过门控机制混合后的结果。

### 负载均衡问题

如果路由器将 90% 的令牌分配给专家 3，其他专家将会处于闲置状态。目前已尝试过三种解决方案：

1. **辅助负载均衡损失机制**（Switch Transformer、Mixtral）。引入与专家使用率方差成正比的惩罚项。该方法有效，但会增加一个超参数以及第二个梯度信号。
2. **专家容量限制 + 令牌丢弃机制**（早期 Switch 版本）。每个专家最多处理 `C × N/E` 个令牌；超出部分的令牌将被跳过 해당层。但这会降低模型质量。
3. **无辅助损失均衡机制**（DeepSeek-V3）。为每个专家引入一个可学习的偏置值，以此调整路由器的 top-k 选择策略。该偏置值在训练损失之外进行更新，不会对主目标函数造成惩罚。这是 2024 年的一项重大突破。

DeepSeek-V3 的实现方式如下：在每个训练步骤之后，针对每个专家检查其使用率是高于还是低于目标值，然后将偏置值调整 `±γ`。路由选择时会使用 `scores + bias` 的结果；而用于控制流控制的专家概率则仍为原始的 `scores` 值。这种方式实现了路由决策与模型表达之间的解耦。

### 共享专家资源

DeepSeek-V2/V3同样将专家模型分为*共享型*和*路由型*两种。每个token都会依次经过所有的共享型专家模型。而路由型专家模型则是通过top-k算法进行选取的。共享型专家负责处理通用知识，而路由型专家则专注于特定领域。V3版本会启用1个共享型专家模型，以及从256个路由型专家中选出的前8个。

### 细粒度专家模型

经典 MoE（GShard、Switch）：每个专家层的宽度与完整的 FFN 相同。`E` 的值较小（8–64），`k` 的值也较小（1–2）。

现代细粒度 MoE（DeepSeek-V3、Qwen-MoE）：每个专家层的宽度更窄，仅为 FFN 宽度的 1/8。`E` 的值较大（256+），`k` 的值更大（8+）。虽然总参数量保持不变，但组合数量的增长速度要快得多。每个 token 可能存在的“专家”组合数为 `C(256, 8) = 400万亿` 种。模型质量得以提升，而延迟则基本保持不变。

### 成本结构

按令牌数及层级划分：

| 配置 | 活跃参数量 / 令牌数 | 总参数量 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | 约 390 亿 | 1410 亿 |
| Llama 3 70B（密集型） | 700 亿 | 700 亿 |
| DeepSeek-V3 | 370 亿 | 6710 亿 |
| Kimi K2（MoE架构） | 约 320 亿 | 1万亿 |

在几乎所有基准测试中，DeepSeek-V3 的表现均优于 Llama 3 70B（密集型），且**每令牌的活跃 FLOPs 更低**。参数越多，模型掌握的知识越丰富；活跃 FLOPs 越高，每处理一个令牌所需的计算资源就越多。MoE 架构则实现了这两者之间的解耦。

### 问题所在：内存

无论启用哪些专家网络，所有专家节点均运行在 GPU 上。一个 671B 规模的模型，其 fp16 格式的权重需要约 1.3 TB 的显存。Frontier MoE 的部署依赖于专家并行机制——即将各个专家节点分配到不同的 GPU 上，并通过网络传输令牌。延迟的主要来源是节点间的全连接通信，而非矩阵乘法运算。

## 构建它

请参阅 `code/main.py`。这是一个使用纯标准库实现的紧凑型 MoE 层，具备以下特性：

- `n_experts=8` 个类似 SwiGLU 的专家模块（为便于说明，每个模块仅包含一个线性层）
- top-k=2 路由机制
- 经 softmax 归一化的门控权重
- 通过各专家模块的偏置实现无辅助损失平衡

### 步骤 1：路由器

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

偏差影响的是数据选择，而非门控权重。这正是 DeepSeek-V3 的巧妙之处——它能够纠正数据分布不均的问题，而不会改变模型的预测结果。

### 步骤 2：将 100 个令牌通过路由器处理

跟踪各专家的调用频率。若没有偏差校正，使用率会出现偏斜。通过引入偏差更新循环（过度使用的专家设为 `-γ`，使用不足的专家设为 `+γ`），经过几次迭代后，使用率即可收敛至均匀分布状态。

### 步骤 3：参数数量比较

打印 MoE 配置的“密集等效”形式。DeepSeek-V3 的结构为：256 个路由层 + 1 个共享层，激活层数量为 8 层，d_model 值为 7168。其总参数量极为庞大。该模型的激活层数量仅为密集版 Llama 3 70B 的七分之一。

## 使用它

HuggingFace 加载中：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026年生产环境推理：vLLM原生支持MoE路由功能。SGLang则拥有最快的专家并行处理路径。两者均能自动实现top-k筛选与专家并行化处理。

**何时选择MoE：**
- 您希望在较低的每Token推理成本下获得顶尖的质量。
- 您具备足够的VRAM及专家并行基础设施。
- 您的工作负载以Token量为重（如聊天、代码），而非上下文量较大（如长文档）。

**何时不应选择MoE：**
- 边缘部署场景——任何处于活跃状态的FLOP都会占用全部存储空间。
- 对延迟要求极高的单用户服务场景——专家路由会增加额外开销。
- 小型模型（<7B参数）——MoE的质量优势仅在计算阈值之上才会显现（约6B活跃参数）。

## 发布它

请参阅 `outputs/skill-moe-configurator.md`。该技能会根据参数预算、训练令牌数量以及部署目标，为新的混合专家模型选择 E、k 以及共享专家布局。

## 练习题

1. **简单级。** 运行 `code/main.py`，观察在50次迭代过程中，无辅助损失偏置更新如何使各专家的使用频率趋于均衡。
2. **中等级。** 用基于哈希的路由器（确定性、无需学习）替换已学习的路由器，比较两者在质量与平衡性方面的表现。为何已学习的路由器更优？
3. **高级别。** 实现类似GRPO的“滚动匹配路由”机制（DeepSeek-V3.2中的技巧）：记录推理过程中激活的专家，并在梯度计算时强制使用相同的路由策略。在一个简化的策略梯度实验环境中衡量其效果。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Expert | “众多 FFN 中的一个” | 独立的前馈网络；其参数专门用于 FFN 计算中的某一小部分。 |
| Router | “门控机制” | 一个极小的线性层，用于为每个标记计算在各个 Expert 上的得分，并进行 top-k 选择。 |
| Top-k routing | “每个标记使用 k 个活跃 Expert” | 每个标记的 FFN 计算会通过恰好 k 个 Expert，其权重由门控机制决定。 |
| Auxiliary loss | “负载均衡惩罚项” | 用于惩罚 Expert 使用不均衡情况的额外损失项。 |
| Auxiliary-loss-free | “DeepSeek-V3 的技巧” | 仅通过在路由器选择时为每个 Expert 设置偏置来实现平衡，无需额外的梯度计算。 |
| Shared expert | “始终启用”的 Expert | 一个额外的 Expert，所有标记都会经过它；用于捕获通用知识。 |
| Expert parallelism | “按 Expert 分片处理” | 将不同的 Expert 分配到不同的 GPU 上，并在网络中路由标记。 |
| Sparsity | “活跃参数 < 总参数数” | 比率为 `k × expert_size / (E × expert_size)`；DeepSeek-V3 的该比值为 37/671，约为 5.5%。 |

## 延伸阅读

- [Shazeer 等人（2017）。规模惊人的神经网络：稀疏门控专家混合层](https://arxiv.org/abs/1701.06538) —— 相关理念。  
- [Fedus、Zoph、Shazeer（2022）。Switch Transformer：利用简单高效的稀疏性实现万亿参数模型的扩展](https://arxiv.org/abs/2101.03961) —— Switch，经典的专家混合架构。  
- [Jiang 等人（2024）。Mixtral 专家混合模型](https://arxiv.org/abs/2401.04088) —— Mixtral 8×7B 模型。  
- [DeepSeek-AI（2024）。DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437) —— MLA 架构 + 无辅助损失专家混合模型 + 多任务并行技术。  
- [Wang 等人（2024）。专家混合模型的无辅助损失负载均衡策略](https://arxiv.org/abs/2408.15664) —— 基于偏差的负载均衡方法相关论文。  
- [Dai 等人（2024）。DeepSeekMoE：迈向专家混合语言模型中的极致专家专业化](https://arxiv.org/abs/2401.06066) —— 本课程中路由器所采用的细粒度专家划分与共享专家机制。  
- [Kim 等人（2022）。DeepSpeed-MoE：提升专家混合模型的推理与训练性能](https://arxiv.org/abs/2201.05596) —— 原始的共享专家相关论文。
