# 推测解码——起草、验证、重复

> 自回归解码属于串行处理方式，每个标记的生成都需要等待前一个标记的处理完成。而推测性解码则打破了这一顺序：先用一个计算成本较低的模型草拟出 N 个标记，再使用计算成本较高的模型通过一次前向传播来验证这 N 个标记的正确性。如果草拟结果正确，只需支付一次针对 N 代生成的大规模前向传播费用即可。

**类型：** 构建
**语言：** Python
**先修要求：** 第 7 阶段 · 07（GPT 因果语言模型）、第 7 阶段 · 12（KV 缓存与 Flash Attention）
**耗时：** 约 60 分钟

## 问题所在

在 H100 上，70B 大语言模型对每个 token 进行采样大约需要 30 毫秒，而 3B 草稿模型则仅需约 3 毫秒。如果让 3B 草稿模型先生成 5 个 token，然后再运行一次 70B 模型来验证这 5 个 token 的正确性，那么处理最多 5 个被接受的 token 所需的总时间将为 `5×3 + 30 = 45 毫秒`——相比直接生成则需要 `5×30 = 150 毫秒`。这就是推测解码技术的核心理念：通过使用占用少量额外 GPU 内存的草稿模型，将解码延迟降低 2 到 4 倍。

关键在于必须保持输出序列的分布特性不变。Leviathan 等人（2023 年）以及 Chen 等人几乎同时提出的推测采样技术，能够确保生成的输出序列与大型模型独立生成的结果具有**完全相同的分布特征**，不会牺牲任何质量，仅能提升速度。

截至 2026 年，有四类常见的草稿验证器组合主导着推理领域：

1. **传统推测解码（Leviathan 2023）**：采用独立的草稿模型（例如 Llama 3 1B）与验证器（例如 Llama 3 70B）相结合。
2. **Medusa（Cai 2024）**：验证器上配备多个解码头，可并行预测 `t+1..t+k` 位置的 token，无需单独的草稿模型。
3. **EAGLE 系列（Li 2024, 2025）**：这类轻量级草稿模型会复用验证器的隐藏状态，其正确率高于传统方案，解码速度通常为普通方案的 3 到 4 倍。
4. **前瞻解码（Fu 2024）**：基于雅可比迭代算法，完全不需要草稿模型，属于自推测类型。虽然应用范围较窄，但无需依赖其他组件。

到 2026 年，所有商用推理框架都会默认支持推测解码功能。vLLM、TensorRT-LLM、SGLang 以及 llama.cpp 均至少支持传统推测解码与 EAGLE-2 方案。

## 概念概述

### 核心算法

给定一个验证器 `M_q` 和一个成本更低的草稿模型 `M_p`：

1. 设 `x_1..x_k` 为已解码的前缀。
2. **生成草稿**：使用 `M_p` 通过自回归方式依次提出 `d_{k+1}, d_{k+2}, ..., d_{k+N}`，这些候选项对应的草稿概率分别为 `p_1..p_N`。
3. **并行验证**：对 `x_1..x_k, d_{k+1}, ..., d_{k+N}` 仅运行一次 `M_q`，从而得到位置 `k+1..k+N+1` 对应的验证概率 `q_1..q_{N+1}`。
4. **从左到右逐个接受/拒绝草稿令牌**：对于每个索引 `i`，以概率 `min(1, q_i(d_i) / p_i(d_i))` 决定是否接受该候选项。
5. 当在位置 `j` 首次出现拒绝时：从经过归一化的“残差”分布 `(q_j - p_j)_+` 中采样一个令牌 `t_j`。此后所有的草稿均被丢弃。
6. 若所有 `N` 个候选项均被接受，则从 `q_{N+1}` 分布中再随机抽取一个额外的令牌 `t_{N+1}`（即免费的奖励令牌）。

所谓“残差分布”技巧，其数学本质在于确保最终输出的概率分布与直接让 `M_q` 从头开始采样时的分布完全一致。

### 什么决定了加速比

令 `α` 表示每个草稿标记的预期接受率，`c` 表示草稿生成到验证器的成本比。具体步骤如下：

- 朴素生成方式：每个标记都会触发一次大模型调用。
- 推测生成方式：当 `α` 值较高时，每 `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` 个标记才会触发一次大模型调用。

以 `α = 0.75`、`N = 5` 为例的典型经验值是：大模型调用次数减少3倍，草稿生成成本降低5倍，整体耗时大约缩短2.5倍。

**`α` 的影响因素包括：**

- 草稿与验证器的匹配程度。属于同一系列或使用相同训练数据的模型能显著提升 `α` 值。
- 解码策略。贪婪生成的草稿搭配贪婪的验证器时，`α` 值较高；而温度采样则更难实现精准匹配，导致接受率下降。
- 任务类型。代码和结构化输出的任务更容易被接受（因为更具可预测性），而自由形式的创意写作则较难通过验证。

### Medusa — 无草图模型的草图版本

Medusa 通过在验证器上增加额外的输出头来替代草稿模型。在位置 `t` 处：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每个头都会输出各自的逻辑斯蒂值。在推理阶段，会从每个头中采样以生成候选序列，随后通过一次前向传播并采用树注意力机制来验证这些序列，该机制能够同时考虑所有候选的后续内容。

优点：无需额外的模型。缺点：会增加可训练参数；需要经过监督微调阶段（约10亿个标记）；其接受率略低于基于优质初稿的常规推测解码方法。

### EAGLE — 通过复用隐藏状态实现更优的草图生成

EAGLE-1/2/3（Li等人，2024–2025年）将草案模型设计为极小的Transformer结构（通常仅1层），该模型会输入验证器的最后一层隐藏状态。由于该草案能够获取验证器的特征表示，其预测结果与验证器的输出分布具有高度相关性。因此，模型的通过率可从普通版本的约0.6提升至0.85以上。

EAGLE-3（2025年）在候选续写方案中加入了树搜索算法。vLLM和SGLang将EAGLE-2/3设置为Llama 3/4及Qwen 3的默认配置路径。

### KV 缓存机制详解

验证过程会在单次前向传播中将 `N` 个草稿令牌输入到验证器中。这会使验证器的 KV 缓存增加 `N` 条记录。如果部分草稿被拒绝，就必须将缓存回滚至已被接受的前缀长度。

生产环境中的实现方案（如 vLLM 的 `--speculative-model`、TensorRT-LLM 的 LookaheadDecoder）则通过临时 KV 缓冲区来处理这一问题：先进行写入，在内容被接受后再提交。从概念上来说这并不复杂，但操作起来较为繁琐。

## 构建它

请参阅 `code/main.py`。我们在此实现了核心的推测采样算法（拒绝步骤 + 残差分布），具体包括：

- 一个“大型模型”，它是对手工编写的概率分布应用确定性 softmax 函数得到的（这样我们可以从数学上验证接受概率）。
- 一个“草稿模型”，它是大型模型的微调版本。
- 一个接受/拒绝循环，其生成的边际分布与直接采样结果相同。

### 步骤 1：拒绝处理步骤

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 表示一个均匀随机数。`q_prob` 是验证者对所生成令牌的置信概率，而 `p_prob` 则是生成模型的置信概率。根据利维坦定理，通过该伯努利决策机制，并在拒绝时从残差中采样，即可精确保持验证者的分布不变。

### 步骤 2：残差分布

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素从 `q` 中减去 `p`，将负值限制为零，然后重新归一化。在发生任何拒绝情况时，从此数据中采样。

### 步骤 3：一个假设性步骤

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

一次验证器遍历可处理5个已通过验证的项、1个奖励项，生成6个令牌。

### 步骤 4：测量接受率

在不同的草稿质量级别下运行 10,000 次推测性步骤。绘制接受率与草稿分布和验证器分布之间的 KL 散度之间的关系图。你应该会看到一条清晰的单调关系曲线。

### 步骤 5：验证分布等价性

从实证角度来看：推测循环生成的令牌直方图应与直接从验证器中采样得到的直方图保持一致。这便是实际应用中的利维坦定理。卡方检验可证明二者在抽样误差范围内相符。

## 使用它

生产环境：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

截至2026年中期，TensorRT-LLM拥有最快的Medusa路径。`faster-whisper`为Whisper-large模型封装了基于小规模草稿的推测性解码功能。

**选择草稿方案：**

| 策略 | 适用场景 | 加速倍数 |
|----------|--------------|---------|
| 原生草稿（1B/3B Llama系列） | 快速原型开发，无需训练 | 1.8–2.3倍 |
| Medusa头部结构 | 可对验证器进行微调时 | 2–3倍 |
| EAGLE-2 / 3 | 生产环境，追求最高速度 | 3–4倍 |
| 前瞻解码 | 无需草稿、无需训练且无额外参数时 | 1.3–1.6倍 |

**何时不应使用推测性解码：**

- 单序列生成长度为1–5个标记的情况。此时开销占主导地位。
- 需要极高创造性或采用高温度采样（α值降低）的场景。
- 内存资源受限的部署环境（草稿模型会增加VRAM占用）。

## 发布它

请参阅 `outputs/skill-spec-decode-picker.md`。该技能会为新的推理任务选择一种推测性解码策略（vanilla / Medusa / EAGLE / lookahead）以及相应的调优参数（N、draft temperature）。

## 练习题

1. **简单。** 运行 `code/main.py`，验证在 50,000 个令牌范围内，推测得到的令牌分布与验证器的直接采样分布的卡方检验 p 值是否大于 0.05。
2. **中等难度。** 将每次大模型前向传播处理的令牌数作为函数绘制出随 `N` 变化的加速比，其中 `α = 0.5, 0.7, 0.85`。为每个 `α` 值确定最优的 `N` 值。（提示：每次验证调用预期的令牌数为 `(1 - α^{N+1}) / (1 - α)`。）
3. **高难度。** 实现一个简化的 Medusa 模型：采用第 14 课中的核心 GPT 模型，再添加 3 个额外的语言模型头，用于预测位置 t+2、t+3 和 t+4 的内容。使用 tinyshakespeare 数据集，并通过联合多头损失函数进行训练。将该模型的接受率与通过对同一模型进行截断处理得到的普通草稿的接受率进行比较。
4. **高难度。** 实现回滚机制：首先创建一个包含 10 个令牌前缀的 KV 缓存，输入 5 个草稿令牌后，在第 3 个位置模拟一次拒绝情况。验证在下一次迭代时，缓存中的读取内容是否正确匹配“前缀 + 前两个被接受的草稿”内容。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 草稿模型 | “便宜的那个” | 用于生成候选标记的较小模型；通常比验证模型的成本低10–50倍。 |
| 验证模型 | “大的那个” | 我们需要保持其分布的目标模型；在每次推测步骤中运行一次。 |
| 接受率（α） | “草稿正确的频率” | 验证模型接受草稿的每个标记概率，典型值为0.7–0.9。 |
| 残差分布 | “拒绝时的备用方案” | 经过归一化的`(q - p)_+`；在拒绝时从该分布中采样可保持验证模型的分布不变。 |
| 额外标记 | “免费的那个” | 当所有N个草稿都被接受后，再从验证模型下一步的分布中随机抽取一个标记。 |
| Medusa | “无草稿的推测方式” | 在验证模型上使用多个语言模型头并行预测t+1..t+k位置。 |
| EAGLE | “基于隐藏状态的草稿” | 基于验证模型最底层隐藏状态生成的微型Transformer草稿。 |
| 展望解码 | “雅可比迭代法” | 通过固定点迭代实现自我推测，无需使用草稿模型。 |
| 树形注意力 | “同时验证多个候选项” | 采用分支式验证方式，可同时考虑多个草稿的延续内容。 |
| KV回滚 | “撤销被拒绝的草稿” | 清空KV缓冲区；在草稿被接受时保存数据，在被拒绝时丢弃。 |

## 延伸阅读

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心算法及等价性定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 并行引入方法；严谨的伯努利拒绝证明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa论文；树注意力机制验证。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1版本；基于隐藏状态条件的草案机制。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2版本；动态树深度调整机制。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3版本。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — 基于前瞻解码的方案，无需草案机制。
- [vLLM文档 — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 综合四种策略的标准生产环境参考资料。
- [SafeAILab / EAGLE参考实现](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3版本的参考代码。
