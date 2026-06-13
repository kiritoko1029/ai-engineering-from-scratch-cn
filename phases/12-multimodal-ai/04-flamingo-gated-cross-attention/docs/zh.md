# Flamingo 与门控交叉注意力实现少样本 VLM

> DeepMind 的 Flamingo（2022）率先做了两件事。它证明了单一模型可以处理任意交错的图像、视频和文本序列。它还证明了 VLM 可以做上下文学习——给一个包含三对（图像，描述）的少样本提示，模型无需任何梯度更新就能为新图像生成描述。其机制是：门控交叉注意力层，插入冻结 LLM 的现有层之间，带有学习的 tanh 门控，初始化为零以保留 LLM 的文本能力。本课将讲解 Flamingo 的 Perceiver resampler 和门控交叉注意力架构——Gemini 交错输入和 Idefics2 视觉 token 的祖先。

**类型：** 学习
**语言：** Python（标准库，门控交叉注意力 + Perceiver resampler 演示）
**前置要求：** Phase 12 · 03（BLIP-2 Q-Former）
**所需时间：** 约120分钟

## 学习目标

- 解释门控交叉注意力如何通过 `tanh(gate) = 0` 在初始化时保留冻结 LLM 的文本能力。
- 走一遍 Perceiver resampler：N 个图像 patch → 通过交叉注意力得到 K 个固定的"潜在"query。
- 描述 Flamingo 如何处理交错的图文序列，使用尊重图像放置的因果掩码。
- 复现一个少样本多模态提示结构（3 个图文示例 + 一个查询图像）。

## 问题所在

BLIP-2 将 32 个视觉 token 送入冻结 LLM 的输入层。适用于每个提示一张图像。但如果你想送入*多张*图像与文本交错，比如"这是图像 A，为它生成描述；这是图像 B，为它生成描述；现在这是图像 C，为它生成描述"呢？LLM 的自注意力需要在单个流中处理图像 token 和文本 token，哪些位置可以关注哪些图像的问题变得棘手。

Flamingo 的答案：完全不改变 LLM 的输入流。在现有 LLM 块之间插入额外的交叉注意力层。文本 token 仍然像往常一样流经 LLM 的因果自注意力。每隔几个 LLM 块，文本 token 也通过一个新的门控层交叉关注图像特征。门控（初始化为零）意味着在第 0 步新层是无操作——模型行为与预训练 LLM 完全一致。随着训练的进行，门控打开，视觉信息开始流动。

Flamingo 回答的第二个问题：如何处理每个提示中可变数量的图像（0、1 或多张）？Perceiver resampler——一个小型交叉注意力模块，接受任意数量的 patch 并产生固定数量的视觉潜在 token。LLM 交叉注意力层看到的形状始终相同，无论提示中有多少图像。

## 概念说明

### 冻结的 LLM

Flamingo 从一个冻结的 Chinchilla 70B LLM 开始。所有 700 亿权重不动。现有的文本自注意力和 FFN 正常运行。

### Perceiver resampler

对于提示中的每张图像，ViT 产生 N 个 patch token。Perceiver resampler 有 K 个固定的可学习潜在向量（Flamingo 使用 K=64）。每个 resampler 块有两个子步骤：

1. 交叉注意力：K 个潜在向量关注 N 个 patch token（Q 来自潜在向量，K/V 来自 patch）。
2. 潜在向量之间的自注意力 + FFN。

经过 6 个 resampler 块后，输出是 K=64 个维度为 1024 的视觉 token，无论 ViT 产生了多少 patch。224x224 图像（196 个 patch）和 480x480 图像（900 个 patch）都输出为 64 个 resampler token。

对于视频，resampler 在时间维度上应用：每帧的 patch 产生 64 个潜在向量，时间位置编码让模型区分 t=0 和 t=N。完整视频变成 T * 64 个视觉 token。

### 门控交叉注意力

在冻结 LLM 的每 M 层之间（Flamingo 使用 M=4），插入一个新的门控交叉注意力块：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是可学习的标量，初始化为零。
- `tanh(0) = 0`，因此初始化时门控分支贡献为零。
- 随着 `alpha` 偏离零，交叉注意力的贡献平滑增长。
- 残差连接意味着即使完全打开的门控也不会覆盖 LLM 的文本表示；只是在其上叠加视觉信息。

这是 Flamingo 最重要的设计选择：视觉条件是加性的、门控的、初始化为零的。第 0 步的 Flamingo 在纯文本输入上是一个完美的 Chinchilla 70B。

### 交错输入的掩码交叉注意力

在类似"<图像 A> 描述 A <图像 B> 描述 B <图像 C> ?"的提示中，每个文本 token 只应看到序列中在它之前的图像。交叉注意力掩码强制：位置 `t` 的文本 token 只关注图像索引 `i < i_t` 的图像 resampler token，其中 `i_t` 是位置 `t` 之前最近的图像。"只看到前一张图像"或"看到所有前面的图像"都是合理的选择；Flamingo 选择了前者。

### 上下文少样本学习

Flamingo 的提示如下：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到补全模式并输出"bird"（或 image3 显示的任何内容）。没有梯度更新。冻结 LLM 的上下文学习能力通过门控交叉注意力延续——这是论文的核心论点，也是其重要性的原因。

### 训练数据

Flamingo 在三个数据集上训练：

1. MultiModal MassiveWeb（M3W）：4300 万个包含交错图像和文本的网页，重建阅读顺序。
2. 图文对（ALIGN + LTIP）：44 亿对。
3. 视频文本对（VTP）：2700 个短视频片段。

OBELICS（2023）是交错网络语料库的开放复现，Idefics、Idefics2 和大多数开放的"类 Flamingo"模型都在其上训练。

### OpenFlamingo 和 Otter

OpenFlamingo（2023）是开放复现。架构相同（Perceiver resampler + 门控交叉注意力，基于冻结的 LLaMA 或 MPT）。检查点有 3B、4B、9B。由于基础 LLM 较小且数据较少，质量落后于 Flamingo。

Otter（2023）基于 OpenFlamingo，在 MIMIC-IT（多模态指令数据集）上进行指令微调，表明门控交叉注意力也适用于指令遵循。

### 后裔

- Idefics / Idefics2 / Idefics3：Hugging Face 的门控交叉注意力谱系，逐步简化（Idefics2 放弃了 resampler，改用带自适应池化的直接 patch token）。
- Flamingo 到 Chameleon 的过渡：到 2024 年许多团队转向早期融合（课程 12.11）；Flamingo 式的门控交叉注意力在需要骨干冻结的生产环境中仍然存在。
- Gemini 的交错输入：概念上继承了 Flamingo 的交错格式灵活性，尽管具体机制是私有的。

### 与 BLIP-2 的对比

| | BLIP-2 | Flamingo |
|---|---|---|
| 视觉桥梁 | 在输入层做一次 Q-Former | 每 M 层做门控交叉注意力 |
| 视觉 token | 每张图像 32 个 | 每张图像每层交叉注意力 64 个 |
| 冻结 LLM | 是 | 是 |
| 少样本上下文 | 弱 | 强——论文的核心 |
| 交错输入 | 无原生支持 | 有，设计目标 |
| 训练数据 | 1.3 亿对 | 13 亿对 + 4300 万交错页面 |
| 参数量 | 1.88 亿训练 | 约 100 亿训练（交叉注意力层） |
| 计算量 | 8 张 A100 上数天 | 数千张 TPUv4 上数周 |

选择 BLIP-2 用于预算内的单图像 VQA。选择 Flamingo/Idefics2 用于交错、少样本或多图像推理。

## 开始构建

`code/main.py` 演示了：

1. 对 36 个假 patch token 使用 8 个可学习潜在向量的 Perceiver resampler（纯 Python 交叉注意力）。
2. `alpha = 0` 时的门控交叉注意力步骤 → 输出等于输入（LLM 不变），然后 `alpha = 2.0` → 视觉贡献混合进来。
3. 交错掩码构建器，为"（图像 1）（文本 1）（图像 2）（文本 2）"序列生成 2D 注意力掩码。

## 交付成果

本课生成 `outputs/skill-gated-bridge-diagnostic.md`。给定开放 VLM 的配置（resampler 有/无、交叉注意力频率、门控方案），它识别 Flamingo 谱系元素并解释冻结策略。在调试为什么微调导致文本性能退化时很有用（答案：门控开得太快太大）。

## 练习

1. 计算 Flamingo-9B 的视觉参数量：90 亿 LLM + 14 亿门控交叉注意力层 + 6400 万 resampler。训练参数占总参数的比例是多少？

2. 在 PyTorch 中实现门控残差 `y = tanh(alpha) * cross + x`。实验验证在 `alpha=0` 时，`y==x` 在初始化时精确成立。

3. 阅读 OpenFlamingo 第 3.2 节（arXiv:2308.01390），了解他们如何处理批次中每个提示有不同图像数量的情况。描述填充策略。

4. 为什么 Flamingo 的交叉注意力掩码让文本 token 只关注*最近的*前一张图像，而非所有前面的图像？阅读 Flamingo 论文第 2.4 节并解释权衡。

5. 上下文少样本：为一个新的 Flamingo 变体构建一个包含 4 个"图像 → 主要物体颜色"示例的提示。描述当示例数量从 0 变化到 8 时的预期准确率模式。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Perceiver resampler | "固定潜在交叉注意力" | 从可变数量的输入 patch 产生 K 个固定 token 的模块 |
| 门控交叉注意力 | "Tanh 门控桥梁" | 残差层 `y = tanh(alpha)*cross + x`，可学习 alpha，初始化为 0 |
| 交错输入 | "混合序列" | 图像和文本按阅读顺序自由混合的提示格式 |
| 冻结 LLM | "无 LLM 梯度" | 文本 LLM 的权重不更新；只有 resampler + 交叉注意力层训练 |
| 少样本 | "上下文示例" | 在提示中给出几个（图像，答案）对；模型无需微调即可泛化 |
| OBELICS | "交错网络语料库" | 包含 1.41 亿个网页的开放数据集，图像和文本按阅读顺序排列 |
| Chinchilla | "700 亿冻结基座" | Flamingo 的冻结文本 LLM，来自 DeepMind 的 Chinchilla 论文 |
| 门控调度 | "alpha 如何变化" | 训练期间交叉注意力门控打开的速度 |
| 交叉注意力频率 | "每 M 层" | 门控交叉注意力块插入的频率；Flamingo 使用 M=4 |
| OpenFlamingo | "开放复现" | MosaicML/LAION 的 3-9B 开放检查点；架构与 Flamingo 相同 |

## 延伸阅读

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始论文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 开放复现。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交错网络语料库。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用 Perceiver 架构。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 指令微调的 Flamingo 后裔。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 方法的现代简化。
