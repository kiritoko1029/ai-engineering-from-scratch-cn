# 差分注意力（V2）

> Softmax 注意力将少量概率分散到每个不匹配的 token 上。在 100k token 下，这些噪声累积并淹没了信号。差分 Transformer（Ye 等，ICLR 2025）通过将注意力计算为两个 softmax 的差来修复此问题，减去共享的噪声基底。DIFF V2（Microsoft，2026 年 1 月）是生产栈重写：匹配基线 Transformer 的解码延迟，无需自定义 kernel，兼容 FlashAttention。本课程从 V1 到 V2 端到端讲解，并提供一个可在标准库 Python 中运行的差分运算玩具实现。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 7 阶段 · 02（自注意力），第 7 阶段 · 15（注意力变体），第 10 阶段 · 14（架构走读）
**所需时间：** 约60分钟

## 学习目标

- 精确说明 softmax 注意力为什么有噪声基底，以及它为什么随上下文长度增长。
- 推导差分注意力公式，解释为什么减法能抵消共享噪声分量同时保留信号。
- 梳理 V1 到 V2 的 diff：什么变快了、什么变简单了、什么变稳定了，以及每个改变对生产预训练为什么是必要的。
- 用纯 Python 从零实现差分注意力，并在合成信号加噪声查询上实证验证噪声抵消特性。

## 问题所在

标准 softmax 注意力有一个数学特性，在规模化时变成操作上的麻烦。对于查询 `q`，注意力权重是 `softmax(qK^T / sqrt(d))`。Softmax 永远不会产生精确的零——每个不匹配的 token 都获得一些正的质量。那些残余质量就是噪声，它随上下文长度增长。在 128k token 时，即使每个不匹配的 token 只获得 0.001% 的概率，127,999 个合起来贡献了大约 12% 的总和。模型必须学会绕过一个随上下文增长的噪声基底。

经验上，这表现为注意力头干扰：长上下文 RAG 中的幻觉引用、100k token 检索任务中的中间丢失失败、以及 32k 以上针干草堆基准的微妙精度退化。差分 Transformer 论文（arXiv:2410.05258，ICLR 2025）测量了差距：DIFF Transformer 达到了更低的困惑度、更高的长上下文精度和比同等规模基线更少的幻觉。

DIFF V1 有三个问题阻碍其进入前沿预训练流水线。它的值缓存每步解码需要加载两次，需要自定义 CUDA kernel 破坏了 FlashAttention 兼容性，以及它的逐头 RMSNorm 在 70B 以上规模的长时间训练中不稳定。DIFF V2（Microsoft unilm 博客，2026 年 1 月 20 日）修复了所有三个问题。本课程讲解两个版本，构建差分算子，并在玩具查询上基准测试噪声抵消。

## 概念说明

### Softmax 的噪声基底

对于查询 `q` 和键 `K = [k_1, ..., k_N]`，注意力权重为：

```
w_i = exp(q . k_i / sqrt(d)) / sum_j exp(q . k_j / sqrt(d))
```

没有任何 `w_i` 是零。如果 `k_i` 与 `q` 完全无关，分数 `q . k_i` 不是 0——它围绕零波动，方差为 `||q||^2 / d`。经过 softmax 归一化后，每个无关 token 仍然贡献 `O(1/N)` 的加权和。无关 token 的总贡献是 `O((N-1)/N) = O(1)`——不是小数量。

模型想要的是类似硬 top-k 的东西：匹配 token 上高权重，其余地方接近零。Softmax 太平滑，无法直接做到。

### 差分思想

将每个头的 Q 和 K 投影分成两半：Q = (Q_1, Q_2) 和 K = (K_1, K_2)。计算两个注意力图：

```
A_1 = softmax(Q_1 K_1^T / sqrt(d))
A_2 = softmax(Q_2 K_2^T / sqrt(d))
```

输出：

```
DiffAttn = (A_1 - lambda * A_2) V
```

减法抵消了两个图共享的噪声分布。如果两个图在 127k 无关 token 上都有大致均匀的权重（随机初始化时确实如此），这些就会抵消。信号——在少数真正相关 token 上的尖峰权重——只有在两个图中以相同幅度出现时才会抵消，模型训练后就不会这样了。

`lambda` 是每头的可学习标量，参数化为 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是负数。`lambda_init` 默认为一个小正数如 0.8。

### 为什么这与有头噪声消除匹配

想象两个嘈杂的麦克风录制同一个声音。两者都拾取说话者加上相关的背景噪声。将一个减去另一个，共享噪声就消失了。声音存活是因为两个信号在相位或幅度上差异足够大，阻止了完全抵消。每头的 `lambda` 正是学习这种平衡。

### V1 vs V2：diff

V1 保持参数量与基线 Transformer 相等。为了每个头获得两个查询，它将头维度减半。这损失了头的表达能力——更痛苦的是——每头的值缓存减半。解码每步需要加载两次值缓存（每个 softmax 分支一次）。结果：尽管参数量匹配，解码比基线慢。

V2 将查询头数量加倍，保持 KV 头不变（从上投影借参数）。头维度与基线相同。减法之后，额外维度被投影回与基线 Transformer 的 O_W 投影匹配。三件事同时发生：

1. 解码速度匹配基线（KV 缓存只加载一次）。
2. FlashAttention 原样运行（无需自定义 kernel）。
3. 解码时的算术强度提升（从 HBM 每加载一字节有更多计算）。

V2 还去掉了 V1 用来稳定减法的逐头 RMSNorm。在 70B 级预训练规模下，那个 RMSNorm 在训练后期造成了不稳定。V2 用更简单的初始化方案替代，在没有额外模块的情况下保持训练稳定。

### 何时使用

| 工作负载 | 收益 |
|---------|------|
| 长上下文 RAG（64k+） | 更干净的注意力图，更少的幻觉引用 |
| 针干草堆基准 | 32k 以上显著的精度提升 |
| 多文档 QA | 更少的跨文档干扰 |
| 8k 代码补全 | 边际收益，不值得改变架构 |
| 短聊天（< 4k） | 与基线基本无法区分 |

价值随上下文长度增长。在 4k token 时噪声基底足够小，标准注意力没问题。在 128k 时它在伤害你。

### 与其他 2026 旋钮的兼容性

| 特性 | 与 DIFF V2 兼容？ |
|------|-----------------|
| GQA | 是（V2 增加 Q 头，不增加 KV 头） |
| MLA（DeepSeek） | 原则上是，但没有发表的论文将两者结合 |
| MoE | 是（注意力独立于 MLP 块） |
| RoPE | 是（不变） |
| YaRN / 长上下文缩放 | 是（正是 DIFF 最有帮助的地方） |
| FlashAttention | V2 中是（V1 中否） |
| 推测解码 | 是（注意力改变对推测解码循环不可见） |

```figure
differential-attention
```

## 开始构建

`code/main.py` 用纯 Python 实现差分注意力。一个已知信号加噪声结构的玩具查询让你可以直接度量噪声抵消比。

### 第 1 步：标准 softmax 注意力

标准库矩阵运算：列表的列表、手动矩阵乘法、带数值稳定性最大值减法的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 第 2 步：将 Q、K 分成两半

V1 风格：头维度减半。V2 风格：保持头维度，头数量加倍。玩具实现使用 V1 以获得教学清晰度——数学完全相同，只有记账不同。

### 第 3 步：两个 softmax 分支 + 减法

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：输出权重可以是负数。这没问题——值缓存仍然处理带符号的贡献。后续的 V 投影吸收符号。

### 第 4 步：噪声抵消度量

构建一个长度 1024 的合成序列。在已知位置放置信号 token，其余填充噪声。计算（a）标准 softmax 注意力在信号位置的权重和（b）差分注意力权重。度量两者的信噪比。DIFF 注意力可靠地产出更高的信噪比，因子为 3x-10x，取决于两个分支被训练得有多不同。

### 第 5 步：V1 vs V2 参数核算

给定配置（hidden=4096, heads=32, d_head=128），打印：

- 基线 Transformer：Q、K、V 各大小 `hidden * hidden`，MLP 为 4 * hidden。
- DIFF V1：Q、K 各大小 `hidden * hidden`，V 大小 `hidden * hidden`（不变），内部头维度减半。增加逐头 `lambda` 参数（O(heads * d_head)）。
- DIFF V2：Q 大小 `2 * hidden * hidden`，K 大小 `hidden * hidden`，V 大小 `hidden * hidden`。额外维度在 O_W 之前投影回来。增加相同的 `lambda` 参数。

玩具度量 V2 的额外参数成本（每个注意力块大约 `hidden * hidden`）并打印。

## 使用它

截至 2026 年 4 月，DIFF V2 尚未在每个生产推理服务器中发布，但 vLLM 和 SGLang 的集成正在进行中。同时，该模式出现在：

- Microsoft 内部长上下文生产模型。
- 针对 256k 以上上下文的多个开放模型训练运行的研究复现。
- 在交替层上结合 DIFF 注意力和滑动窗口注意力的混合架构。

2026 年你会使用它的情况：

- 从头训练新模型，目标 64k 以上有效上下文。从一开始就加入差分注意力；之后再重新训练代价很大。
- 微调长上下文模型，其中中间丢失失败主导了你的评估。Q 投影上的 LoRA 可以近似 DIFF 结构。

2026 年你不会使用它的情况：

- 你正在服务一个预训练的稠密模型，长上下文性能稳定。在现有权重上重新训练很少能回本。
- 你的上下文总是在 16k 以下。噪声基底可以忽略。

## 交付它

本课程产出 `outputs/skill-diff-attention-integrator.md`。给定模型架构、目标上下文长度、幻觉画像和训练预算，它产出一份将差分注意力添加到新预训练运行或 LoRA 微调中的集成计划。

## 练习

1. 运行 `code/main.py`。验证差分注意力报告的信噪比在合成查询上高于标准 softmax 注意力。改变噪声幅度，展示标准注意力变得不可用的交叉点。

2. 计算 7B 级模型（hidden=4096, heads=32, d_head=128, 32 层）从基线到 DIFF V1 和从基线到 DIFF V2 的参数量差异。展示哪些组件增加了参数，哪些保持不变。

3. 阅读 DIFF V1 论文（arXiv:2410.05258）的第 3 节和 DIFF V2 Hugging Face 博客的第 2 节。用两句话解释 V1 的逐头 RMSNorm 为什么是必要的，以及 V2 为什么能在不导致训练发散的情况下去掉它。

4. 实现消融实验：用 `lambda = 0`（纯第一个 softmax）和 `lambda = 1`（完全减法）计算差分注意力。在合成查询上度量信噪比如何随扫描变化。找出使信噪比最大化的 `lambda`。

5. 将玩具扩展到 GQA + DIFF V2。选择 8 个 KV 头和 32 个 Q 头。展示 KV 缓存大小与相同 (8, 32) 配置的基线 GQA 模型匹配。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 差分注意力 | "两个 softmax 相减" | 将 Q、K 分成两半，计算两个 softmax 图，从第一个中减去第二个（乘以 lambda），然后乘以 V |
| 噪声基底 | "softmax 的非零尾部" | softmax 在每个无关 token 上分配的 O(1/N) 权重，在长上下文中累加为 O(1) |
| lambda | "减法缩放" | 每头可学习标量，参数化为 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`；可以是负数 |
| DIFF V1 | "ICLR 2025 版本" | 原始差分 Transformer；头维度减半以保持参数量，需要自定义 kernel，解码较慢 |
| DIFF V2 | "2026 年 1 月修复" | 加倍 Q 头保持 KV 头不变；匹配基线解码速度并兼容 FlashAttention |
| 逐头 RMSNorm | "V1 的稳定器" | V1 在差分之后应用的额外归一化；V2 移除它以防止训练后期不稳定 |
| 信噪比 | "注意力浪费了多少" | 真实信号位置的权重与无关位置平均权重的比率 |
| 中间丢失 | "长上下文失败模式" | 长上下文中文档位于中间时检索精度下降的经验现象——差分注意力减少此现象 |
| 算术强度 | "每加载一字节的 FLOPs" | V2 通过在每次 KV 加载中加倍查询来提升的比率；对内存受限的解码很重要 |

## 延伸阅读

- [Ye et al. -- Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258) -- 含噪声抵消理论和长上下文消融的原始论文
- [Microsoft unilm -- Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2) -- 生产栈重写，匹配基线解码，兼容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333) -- 为什么减法能恢复预训练注意力结构的理论分析
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900) -- 参数共享变体
- [Vaswani et al. -- Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762) -- DIFF 从中减去的基线 Transformer
- [Liu et al. -- Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172) -- 差分注意力针对的长上下文基准
