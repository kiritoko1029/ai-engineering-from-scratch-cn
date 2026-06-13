# 推测解码与 EAGLE

> 前沿 LLM 生成一个 token 需要对数十亿参数进行完整的前向传播。该前向传播是严重过度配置的：大部分时间一个更小的模型能正确猜测接下来的 3-5 个 token，而大模型只需*验证*猜测。当猜测正确时，你用一个 token 的价格获得了 5 个 token。推测解码（Leviathan 等 2023）使之精确，EAGLE-3（2025）将接受率推到每次验证约 4.5 个 token——在匹配输出分布下 4-5 倍加速。

**类型：** 构建
**语言：** Python（使用 numpy）
**前置要求：** 第 10 阶段课程 12（推理优化），第 10 阶段课程 04（预训练 Mini-GPT）
**所需时间：** 约75分钟

## 问题所在

70B 级模型在 H100 上的解码吞吐通常是 40-80 token/秒。每个 token 需要从 HBM 读取所有模型权重的完整前向传播。你不能在不改变输出的情况下缩小模型。你不能在超过内存的情况下增加批大小。你卡住了——除非你能让模型每次前向传播输出多于一个 token。

自回归生成看起来本质上是串行的：`x_{t+1} = sample(p(· | x_{1:t}))`。但有一个并发机会。如果你有一个廉价的预测器说"接下来的 4 个 token 可能是 [a, b, c, d]"，你可以在**大模型的一次前向传播**中验证所有 5 个位置，并接受最长的匹配前缀。

Leviathan、Kalai、Matias（2023，"Fast Inference from Transformers via Speculative Decoding"）通过巧妙的接受/拒绝规则使之精确，保持目标模型的采样分布不变。相同的输出分布，2-4 倍更快。

## 概念说明

### 双模型设置

- **目标模型** `M_p`：你真正想从中采样的又大又慢的高质量模型。分布：`p(x)`。
- **草稿模型** `M_q`：又小又快的低质量模型。分布：`q(x)`。小 5-30 倍。

每步：

1. 草稿模型自回归提出 K 个 token：`x_1, x_2, ..., x_K ~ q`。
2. 目标模型在所有 K+1 个位置并行运行一次前向传播，为每个提出的 token 产出 `p(x_k)`。
3. 通过修改的拒绝采样规则从左到右接受/拒绝每个 token。接受最长的匹配前缀。
4. 如果任何 token 被拒绝，从修正分布中采样替换并停止。否则从 `p(· | x_1...x_K)` 中采样一个额外 token。

如果草稿与目标完美匹配，你每次目标前向获得 K+1 个 token。如果草稿在位置 1 就错了，你只获得 1 个 token。

### 精确性规则

推测解码在分布上**可证明等价于从 p 采样**。拒绝规则：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐点差的正部。当草稿和目标一致（`p ≈ q`）时接受率接近 1。当不一致时，残差分布被构造使得整体样本仍然精确服从 `p`。

**贪心情况。** 对于 temperature=0 采样，只需检查 `argmax(p) == x_t`。如果是，接受；如果不是，输出 `argmax(p)` 并停止。

### 预期加速

如果草稿模型的 token 级接受率是 `α`，每次目标前向传播的预期产出 token 数为：

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

在 `α = 0.8, K = 4` 时：`(1 - 0.8^5)/(1 - 0.8) = 3.36` token 每次前向。单次目标前向大约成本 `cost_q * K + cost_p`（K 次草稿加一次目标验证）。如果 `cost_p >> cost_q * K`，加速比是吞吐上的 `3.36× / 1 = 3.36×`。

唯一真正的参数是 `α`，它完全取决于草稿-目标对齐。好的草稿就是一切。

### 训练草稿：蒸馏

随机小模型做不好草稿。标准配方是从目标蒸馏：

1. 选择一个小架构（70B 目标约 1B，7B 目标约 500M）。
2. 在大型文本语料上运行目标模型；存储其下一个 token 分布。
3. 用 KL 散度对目标分布（而非真实 token）训练草稿。

结果：编码上 `α` 通常 0.6-0.8，自然语言聊天 0.7-0.85。生产中 2-3 倍加速。

### EAGLE：树草稿 + 特征复用

Li、Wei、Zhang、Zhang（2024，"EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"）观察到标准推测解码的两个低效：

1. 草稿做 K 个串行步骤，每步全栈。但草稿可以复用目标最近一次验证的特征（隐藏状态）——目标已经计算了丰富的表示，草稿却在从头重新推导。
2. 草稿输出线性链。如果草稿能输出候选的*树*（每个节点多个猜测），目标的单次前向传播可以通过树注意力掩码并行验证多个候选路径，并选择最长的接受分支。

EAGLE-1 变化：
- 草稿输入 = 目标在位置 t 的最终隐藏状态，而非原始 token。
- 草稿架构 = 1 个 Transformer 解码器层（不是独立的小模型）。
- 输出 = 每深度 K = 4-8 候选的树，深度 4-6。

EAGLE-2（2024）添加动态树拓扑：树在草稿不确定的地方变宽，在确信的地方保持窄。在不增加验证成本的情况下提高 `α_effective`。

EAGLE-3（Li 等 2025，"EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"）去除了固定的顶层特征依赖，并用新的"测试时模拟"损失训练草稿——草稿在匹配目标测试时分布的输出上训练，而非教师强制训练分布。接受率从 0.75（EAGLE-2）升至 0.82（EAGLE-3），平均每次验证 token 从 3.0 升至 4.5。

### 树注意力验证

当草稿输出树时，目标模型在一次前向传播中使用**树注意力掩码**验证——编码树拓扑而非纯线性的因果掩码。每个 token 只关注树中的祖先。验证传播仍然是一次前向、一次矩阵乘法；拓扑掩码只多花几个额外 KV 条目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是竞争的首个 token 候选，`c, d, e, f` 是第二个 token 候选，所有六个位置在一次前向传播中验证。输出是沿任何接受路径的最长前缀。

### 何时赢、何时不赢

**赢：**
- 带可预测文本的聊天/补全（代码、常见英语、结构化输出）。`α` 高。
- 解码期间有未使用 GPU 计算的设置（内存受限阶段）。树草稿利用可用 FLOPs。

**不赢/无收益：**
- 高度随机输出（高温度创意写作）。`α` 趋向 `1/|vocab|`。
- 非常高并发的批服务——批处理已填满 FLOPs，树验证空间小。
- 非常小的目标模型，草稿并不小多少。

生产团队通常报告聊天 2-3 倍时钟加速，代码生成 3-5 倍，创意写作接近零。

```figure
speculative-decoding
```

## 开始构建

`code/main.py`：

- 一个参考 `speculative_decode(target, draft, prompt, K, temperature)`，实现精确拒绝规则并验证其保持目标分布（经验 KL < 0.01 vs 纯目标采样）。
- 一个 EAGLE 风格树草稿器，构建带 top-p 分支的深度 K 树。
- 一个树注意力掩码构建器，为验证器产出正确的因果模式。
- 一个接受率测试工具，在小型 LM 上运行两者（从 GPT-2-medium 目标蒸馏一个 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 使用它

- **vLLM** 和 **SGLang** 发布一等推测解码支持。标志：`--speculative_model`、`--num_speculative_tokens`。通过 `--spec_decoding_algorithm eagle` 标志支持 EAGLE-2/3。
- **NVIDIA TensorRT-LLM** 原生支持 Medusa 和 EAGLE 树。
- **参考草稿模型**：`Qwen/Qwen3-0.6B-spec`（Qwen3-32B 的草稿）、`meta-llama/Llama-3.2-1B-Instruct-spec`（70B 的草稿）。
- **Medusa 头**（Cai 等 2024，"Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"）：不使用草稿模型，而是在目标本身上添加 K 个并行预测头。部署更简单，接受率略低于 EAGLE。

## 交付它

本课程产出 `outputs/skill-speculative-tuning.md`——一个分析目标模型工作负载并选择：草稿模型、K（草稿长度）、树宽度、温度、以及何时回退到纯解码的技能。

## 练习

1. 实现精确拒绝规则并实证验证。运行 10K 次 `speculative_decode` 和纯目标采样；计算两个输出分布之间的 TV 距离。应 < 0.01。

2. 计算加速公式。给定固定 `α` 和 `K`，绘制每次目标前向的预期 token 数。为 α ∈ {0.5, 0.7, 0.9} 找到最优 K。

3. 训练小型草稿。取 124M GPT-2 目标，在 100M token 上用 KL 损失蒸馏 30M GPT-2 草稿。在保留文本上度量 `α`。预期：0.6-0.7。

4. 实现 EAGLE 风格树草稿。不使用链，让草稿在每深度输出 top-3 分支。构建树注意力掩码。验证目标接受最长的正确分支。

5. 度量失败模式。在 temperature=1.5（高随机性）下运行推测解码。展示 α 崩溃，算法因草稿开销比纯解码更慢。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 目标模型 | "大模型" | 你想从中采样的又慢又高质量的模型（p 分布） |
| 草稿模型 | "推测器" | 又小又快的预测器（q 分布）；小 5-30 倍 |
| K / 草稿长度 | "前瞻" | 每次验证传播推测的 token 数 |
| α / 接受率 | "命中率" | 草稿提议被接受的每 token 概率 |
| 精确拒绝规则 | "接受测试" | r < p/q 比较，保持目标分布 |
| 残差分布 | "修正的 p-q" | (p - q)+ / ||(p - q)+||_1，拒绝时的采样分布 |
| 树草稿 | "分支推测" | 草稿输出候选树，用树结构注意力掩码在一次传播中验证 |
| 树注意力掩码 | "拓扑掩码" | 编码树拓扑的因果掩码，使每个节点只关注其祖先 |
| Medusa 头 | "并行头" | 目标本身的 K 个额外预测头；无独立草稿模型 |
| EAGLE 特征复用 | "隐藏状态草稿" | 草稿输入是目标的最后隐藏状态，而非原始 token，缩小草稿 |
| 测试时模拟损失 | "EAGLE-3 训练" | 在匹配目标测试时分布的输出上训练草稿，而非教师强制 |

## 延伸阅读

- [Leviathan, Kalai, Matias, 2023 -- "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) -- 精确拒绝规则和理论加速分析
- [Chen, Borgeaud, Irving et al., 2023 -- "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) -- DeepMind 同期推测采样论文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 -- "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) -- 草稿模型的并行头替代
- [Li, Wei, Zhang, Zhang, 2024 -- "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) -- 特征复用和树草稿
- [Li et al., 2024 -- "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) -- 动态树拓扑
- [Li et al., 2025 -- "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) -- 训练时测试时匹配
- [Fu, Haotian, Peng et al., 2024 -- "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) -- Jacobi/前瞻解码，无推测器的替代方案
