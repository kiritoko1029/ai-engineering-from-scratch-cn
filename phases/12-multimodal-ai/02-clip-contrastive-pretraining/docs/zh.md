# CLIP 与对比式视觉-语言预训练

> OpenAI 的 CLIP（2021）证明了一个足够大的想法足以驱动未来五年：仅使用嘈杂的网络图文对和对比损失，将图像编码器和文本编码器对齐到同一向量空间。零监督标签。4 亿对数据。由此产生的嵌入空间可以做零样本分类、图文检索，并作为视觉塔接入每一个 2026 年的 VLM。SigLIP 2（2025）用 sigmoid 替代了 softmax，以更低成本超越了 CLIP。本课将从 InfoNCE 到 sigmoid 逐对损失推导数学原理，并用标准库 Python 实现训练步骤。

**类型：** 构建
**语言：** Python（标准库，InfoNCE + sigmoid 损失实现）
**前置要求：** Phase 12 · 01（ViT patch），Phase 7（Transformer）
**所需时间：** 约180分钟

## 学习目标

- 从互信息推导 InfoNCE 损失，并实现一个数值稳定的向量化版本。
- 解释为什么 sigmoid 逐对损失（SigLIP）可以扩展到 batch 32768+ 而无需 softmax 所需的 all-gather 开销。
- 通过构建文本模板（`a photo of a {class}`）并对余弦相似度取 argmax 来运行零样本 ImageNet 分类。
- 说出 CLIP / SigLIP 预训练提供的四个杠杆：batch 大小、温度、提示模板、数据质量。

## 问题所在

CLIP 之前的视觉是监督式的。收集标注数据集（ImageNet：120 万张图像，1000 个类别），训练 CNN，部署。标注很昂贵，标注偏向标注者能达成一致的内容，而且标注无法在不微调的情况下迁移到新任务。

网络图文对有十亿以上的松散标注数据可以免费使用。一张金毛猎犬的照片配上替代文本"我的狗 Max 在公园里"就携带了监督信号——文本描述了图像。问题在于：你能把这些变成有用的训练吗？

CLIP 的答案：将图文对视为匹配任务。给定一批 N 张图像和 N 条描述，学习将每张图像与其自身描述匹配，对抗 N-1 个干扰项。监督信号是"这两样东西属于一起；这 N-1 个不属于"。没有类别标签。没有人工标注。只有对比损失。

由此产生的嵌入空间所做的远超 CLIP 的训练目标。ImageNet 零样本之所以有效，是因为"一张猫的照片"嵌入在从未被明确标注为猫的猫图片附近。这就是催生了每一个 2026 年 VLM 的赌注。

## 概念说明

### 双编码器

CLIP 有两个塔：

- 图像编码器 `f`：ViT 或 ResNet，每张图像输出一个 D 维向量。
- 文本编码器 `g`：小型 Transformer，每条描述输出一个 D 维向量。

两个塔都将输出归一化到单位长度。相似度为 `cos(f(x), g(y)) = f(x)^T g(y)`，因为两者都是单位范数。

对于一批 N 个（图像，描述）对，构建形状为 `(N, N)` 的相似度矩阵 `S`：

```
S[i, j] = cos(f(x_i), g(y_j)) / tau
```

其中 `tau` 是学习的温度（CLIP 初始化为 0.07；在对数空间中学习）。

### InfoNCE 损失

CLIP 在行和列上使用对称交叉熵：

```
loss_i2t = CE(S, labels=identity)     # 每张图像的正样本是其自身描述
loss_t2i = CE(S^T, labels=identity)   # 每条描述的正样本是其自身图像
loss = (loss_i2t + loss_t2i) / 2
```

这就是 InfoNCE。CE 中的 softmax 迫使每张图像与其自身描述的匹配度高于批次中所有其他描述。"负样本"是所有其他批次项。更大的批次 = 更多负样本 = 更强的信号。CLIP 在 batch 32k 下训练；规模很重要。

### 温度

`tau` 控制 softmax 的锐度。低 tau → 锐利分布，难负样本挖掘效果。高 tau → 柔和，所有样本都有贡献。CLIP 学习 `log(1/tau)`，并裁剪以防止退化。SigLIP 2 固定初始 tau，改用学习的偏置。

### 为什么 sigmoid 更易扩展（SigLIP）

Softmax 需要整个相似度矩阵同步。在分布式训练中，你必须将每个嵌入 all-gather 到每个副本，然后做 softmax。通信量在 world size 上是二次方的。

SigLIP 用逐元素 sigmoid 替代 softmax：对于每对 `(i, j)`，损失是一个"这对是否匹配？"的二分类。正类标签是主对角线，其余都是负类。损失为：

```
L = -1/N sum over (i, j) [ y_ij log sigmoid(S[i,j]) + (1-y_ij) log sigmoid(-S[i,j]) ]
```

`y_ij = 1` 当 `i == j`，否则为 0。每对的损失是独立的。无需 all-gather。每个 GPU 计算其本地块并求和。SigLIP 2 可以廉价地扩展到 batch 32k-512k，而 CLIP 需要成比例更多的通信。

### 零样本分类

给定 N 个类别名称，为每个类别构建文本模板：

```
"a photo of a {class}"
```

用文本编码器嵌入每个模板。用图像编码器嵌入你的图像。余弦相似度的 argmax = 预测类别。无需在目标类别上训练。

提示模板很重要。CLIP 原始论文为每个类别使用 80 个模板（普通、艺术、照片、绘画等）并平均嵌入。+3 个 ImageNet 点。现代用法通常选择一两个模板。

### 线性探测和微调

零样本是基线。线性探测（在冻结的 CLIP 特征之上训练一个线性层用于目标类别）在域内任务上优于零样本。完全微调在域内优于线性探测，但可能损害零样本迁移。三种模式，三种权衡。

### SigLIP 2：NaFlex 和密集特征

SigLIP 2（2025）新增：
- NaFlex：单一模型处理可变宽高比和分辨率。
- 更好的密集特征用于分割和深度估计，目标是在 VLM 中作为冻结骨干使用。
- 多语言：在 100+ 种语言上训练，而 CLIP 仅支持英语。
- 10 亿参数规模，而 CLIP 最高到 4 亿。

在 2026 年的开放 VLM 中，SigLIP 2 SO400m/14 是默认视觉塔。在纯图文检索中，当特定的 LAION-2B 训练分布匹配你的查询模式时，CLIP 仍然是默认选择。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN（Google，2021）：与 CLIP 相同的想法，18 亿对规模，90% 嘈杂数据。证明了嘈杂数据可以扩展。OpenCLIP（LAION）：在 LAION-400M / 2B 上对 CLIP 的开放复现，多种规模，首选开放检查点。EVA-CLIP：从掩码图像建模初始化；VLM 的强大骨干。BASIC：Google 的 CLIP+ALIGN 混合体。同一家族，不同的数据和调优。

### 零样本天花板

CLIP 类模型在 ImageNet 零样本上大约封顶在 76%（CLIP-G、OpenCLIP-G）。超越需要更大的数据量（SigLIP 2 达到 80%+）或架构变化（监督头、更多参数）。基准测试正在饱和；真正的价值是下游 VLM 消费的嵌入空间。

```figure
multimodal-fusion
```

## 开始构建

`code/main.py` 实现了：

1. 一个玩具双编码器（基于哈希的图像特征、文本字符特征），让你无需 numpy 就能看到 InfoNCE 的形状。
2. 纯 Python 的 InfoNCE 损失（通过 log-sum-exp 实现数值稳定性）。
3. Sigmoid 逐对损失用于对比。
4. 零样本分类例程：计算与一组文本提示的余弦相似度，argmax 预测。

运行它并观察损失曲线。绝对数字是玩具级别的；形状与真实 CLIP 训练器输出的匹配。

## 交付成果

本课生成 `outputs/skill-clip-zero-shot.md`。给定一组图像（通过路径）和目标类别列表，它使用 CLIP 模板构建文本提示，用指定检查点（如 `openai/clip-vit-large-patch14`）嵌入两侧，并返回 top-1 / top-5 预测及相似度分数。该技能拒绝对不在提示列表中的类别做出声明。

## 练习

1. 手动计算一个 batch 4 对的 InfoNCE。构建 4x4 相似度矩阵，运行 softmax，取出对角线，计算交叉熵。用 Python 实现验证此手动计算。

2. SigLIP 在温度之外使用一个偏置参数 `b`：`S'[i,j] = S[i,j]/tau + b`。当批次中类别严重不平衡（每行负样本远多于正样本）时，`b` 起什么作用？阅读 SigLIP 第 3 节（arXiv:2303.15343）。

3. 构建一个猫 vs 狗的零样本分类器。尝试两种提示模板：`a photo of a {class}` 和 `a picture of a {class}`。在 100 张测试图像上测量准确率。模板集成是否优于单个模板？

4. 计算 512 GPU、batch 32k 下 softmax InfoNCE 与 sigmoid 逐对损失的通信成本。哪个是 O(N)，哪个是 O(N^2)？引用 SigLIP 第 4 节。

5. 阅读 OpenCLIP 缩放定律论文（arXiv:2212.07143，Cherti 等人）。从图中复现他们的数据缩放结论：在固定模型大小下，ImageNet 零样本准确率与训练数据量之间的对数线性关系是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| InfoNCE | "对比损失" | 对批次相似度矩阵的交叉熵；每个项的正样本是其配对项，负样本是其余所有 |
| Sigmoid 损失 | "SigLIP 损失" | 逐对二元交叉熵；无 softmax，无 all-gather，在分布式训练中廉价扩展 |
| 温度 | "tau" | 在 softmax/sigmoid 之前缩放 logits 的标量；控制分布的锐度 |
| 零样本 | "无微调分类" | 使用文本提示构建类别嵌入，通过余弦相似度分类；无需在目标类别上训练 |
| 提示模板 | "a photo of a ..." | 围绕类别名称的文本脚手架；影响零样本准确率 1-5 个百分点 |
| 双编码器 | "双塔" | 一个图像编码器 + 一个文本编码器，输出在共享的 D 维空间中 |
| 难负样本 | "棘手的干扰项" | 与正样本足够相似的负样本，模型必须努力将它们分开 |
| 线性探测 | "冻结 + 一层" | 仅在冻结特征之上训练线性分类器；衡量特征质量 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的能力，可在任意宽高比和分辨率下输入图像而无需调整大小 |
| 温度缩放 | "对数参数化 tau" | CLIP 参数化 `log(1/tau)` 以使梯度行为良好；裁剪以防止退化到接近零的 tau |

## 延伸阅读

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 论文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多语言 + NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用嘈杂网络数据扩展。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP 缩放定律。
