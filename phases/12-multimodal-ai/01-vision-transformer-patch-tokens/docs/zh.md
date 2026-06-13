# Vision Transformer 与 Patch Token 原语

> 在进入多模态之前，图像必须先变成 Transformer 能处理的 token 序列。2020 年的 ViT 论文用 16x16 像素 patch、线性投影和位置嵌入给出了答案。五年后的今天，所有 2026 年前沿模型（Claude Opus 4.7 原生支持 2576px、Gemini 3.1 Pro、Qwen3.5-Omni）仍然以此为起点——编码器从 ViT 演进到 DINOv2 再到 SigLIP 2，加入了 register token，位置编码方案变成了 2D-RoPE，但这一原语始终未变。本课将端到端地阅读 patch token 流水线，并用标准库 Python 实现它，为 Phase 12 的后续课程建立"视觉 token"的具体心智模型。

**类型：** 学习
**语言：** Python（标准库，patch tokenizer + 几何计算器）
**前置要求：** Phase 7（Transformer），Phase 4（计算机视觉）
**所需时间：** 约120分钟

## 学习目标

- 将 HxWx3 图像转换为带有正确位置编码的 patch token 序列。
- 计算给定（patch 大小、分辨率、隐藏维度、深度）的 ViT 的序列长度、参数量和 FLOPs。
- 说出将 ViT 从 2020 年研究带到 2026 年生产的三项升级：自监督预训练（DINO / MAE）、register token 和原生分辨率打包。
- 为下游任务在 CLS 池化、均值池化和 register token 之间做出选择。

## 问题所在

Transformer 处理的是向量序列。文本本身就是序列（字节或 token）。图像是一个具有三个颜色通道的二维像素网格——不是序列。如果将每个像素展平，一张 224x224 的 RGB 图像将变成 150,528 个 token，在这个长度上做自注意力是不可行的（序列长度的二次方复杂度）。

2020 年之前的做法是在前面接一个 CNN 特征提取器：ResNet 生成一个 7x7 的 2048 维向量特征图，将这 49 个 token 送给 Transformer。这能用，但继承了 CNN 的偏置（平移等变性、局部感受野），也失去了 Transformer 对规模的追求。

Dosovitskiy 等人（2020）提出了一个直截了当的问题：如果我们跳过 CNN 呢？将图像分割成固定大小的 patch（比如 16x16 像素），将每个 patch 线性投影为一个向量，加上位置嵌入，然后将序列送入一个标准 Transformer。这在当时是异端——没有卷积的视觉。但有了足够的数据（JFT-300M，然后是 LAION），它在 ImageNet 上击败了 ResNet，并持续改进。

到 2026 年，ViT 原语已成为无可争议的基础。每个开放权重 VLM 的视觉塔都是它的某种后裔（DINOv2、SigLIP 2、CLIP、EVA、InternViT）。问题不再是"是否使用 patch？"而是"用什么 patch 大小、什么分辨率策略、什么预训练目标、什么位置编码"。

## 概念说明

### Patch 即 token

给定形状为 `(H, W, 3)` 的图像 `x` 和 patch 大小 `P`，你将图像切成 `(H/P) x (W/P)` 个不重叠的 patch。每个 patch 是一个 `P x P x 3` 的像素立方体。将每个立方体展平为 `3 P^2` 维向量。应用一个共享的线性投影 `W_E`，形状为 `(3 P^2, D)`，将每个 patch 映射到模型的隐藏维度 `D`。

以 ViT-B/16 标准配置为例：
- 分辨率 224，patch 大小 16 → 网格 14x14 → 196 个 patch token。
- 每个 patch 包含 `16 x 16 x 3 = 768` 个像素值，投影到 `D = 768`。
- 添加一个可学习的 `[CLS]` token → 序列长度 197。

Patch 投影在数学上等价于卷积核大小为 `P`、步长为 `P`、输出通道数为 `D` 的 2D 卷积。生产代码正是这样实现的——`nn.Conv2d(3, D, kernel_size=P, stride=P)`。"线性投影"是概念性的表述；"卷积核"是高效的实现。

### 位置嵌入

Patch 没有内在顺序——Transformer 把它们当作一个集合。早期 ViT 添加了可学习的 1D 位置嵌入（每个位置一个 768 维向量，共 197 个）。能用，但将模型绑定到训练分辨率：推理时如果改变网格大小，必须对位置表进行插值。

现代视觉骨干使用 2D-RoPE（Qwen2-VL 的 M-RoPE、SigLIP 2 的默认方案）或分解式 2D 位置。2D-RoPE 根据 patch 的（行，列）索引旋转 query 和 key 向量，因此模型从旋转角度推断相对 2D 位置。不需要位置表。模型在推理时可以处理任意网格大小。

### CLS token、池化输出和 register token

图像级表示是什么？三种方式共存：

1. `[CLS]` token。在 patch 序列前添加一个可学习向量。经过所有 Transformer 块后，CLS token 的隐藏状态就是图像表示。继承自 BERT。原始 ViT、CLIP 使用。
2. 均值池化。对 patch token 的输出隐藏状态取平均。SigLIP、DINOv2、大多数现代 VLM 使用。
3. Register token。Darcet 等人（2023）观察到，没有显式 sink token 训练的 ViT 会发展出高范数的"伪影"patch，劫持自注意力。添加 4-16 个可学习的 register token 可以吸收这些负担，并改善密集预测质量（分割、深度估计）。DINOv2 和 SigLIP 2 都自带 register token。

这个选择对下游任务很重要。CLS 适合分类。对于将 patch token 送入 LLM 的 VLM，你完全跳过池化——每个 patch 都成为 LLM 的输入 token。Register token 在交接前被丢弃（它们是脚手架，不是内容）。

### 预训练：监督式、对比式、掩码式、自蒸馏式

2020 年的 ViT 在 JFT-300M 上用监督分类预训练。很快被以下方法取代：

- CLIP（2021）：在 4 亿对图文对上进行对比学习。见课程 12.02。
- MAE（2021，He 等人）：掩码 75% 的 patch，重建像素。自监督，纯图像数据。
- DINO（2021）/ DINOv2（2023）：学生-教师自蒸馏，无标签，无描述。2023 年的 DINOv2 ViT-g/14 是最强的纯视觉骨干，"密集特征"用例的默认选择。
- SigLIP / SigLIP 2（2023，2025）：使用 sigmoid 损失的 CLIP 加上 NaFlex 支持原生宽高比。2026 年开放 VLM 中主导的视觉塔（Qwen、Idefics2、LLaVA-OneVision）。

预训练的选择决定了骨干擅长什么：CLIP/SigLIP 适合与文本的语义匹配，DINOv2 适合密集视觉特征，MAE 适合作为下游微调的起点。

### 缩放定律

ViT 缩放（Zhai 等人 2022）确立了 ViT 的质量在模型大小、数据量和计算量方面遵循可预测的规律。在固定计算量下：
- 更大的模型 + 更多数据 → 更好的质量。
- Patch 大小是序列长度与保真度之间的杠杆。Patch 14（DINOv2/SigLIP SO400m 的典型配置）比 patch 16 每张图像产生更多 token；对 OCR 和密集任务更好，对速度更差。
- 分辨率是另一个大杠杆。从 224 到 384 到 512 几乎总是有帮助的，但 FLOPs 呈二次方增长。

ViT-g/14（10 亿参数，patch 14，分辨率 224 → 256 个 token）和 SigLIP SO400m/14（4 亿参数，patch 14）是 2026 年开放 VLM 的两个主力编码器。

### ViT 参数量计算

完整计算在 `code/main.py` 中。以 224 分辨率下的 ViT-B/16 为例：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

在加载检查点之前，用这种方式估算每个 ViT 的参数量。骨干的大小决定了你在任何下游 VLM 中的显存下限。

### 2026 年生产配置

2026 年大多数开放 VLM 搭载的编码器是原生分辨率（NaFlex）下的 SigLIP 2 SO400m/14。它具有：
- 4 亿参数。
- Patch 大小 14，默认分辨率 384 → 每张图像 729 个 patch token。
- 图像级任务使用均值池化；所有 729 个 patch 流入 LLM 用于 VQA。
- 4 个 register token，在 LLM 交接前丢弃。
- 2D-RoPE 配合图像级缩放支持原生宽高比。

该配置中的每个决定都可以追溯到一篇你可以阅读的论文。

```figure
image-patch-tokens
```

## 开始构建

`code/main.py` 是一个 patch tokenizer 和几何计算器。它接收（图像 H、W、patch P、隐藏维度 D、深度 L）并报告：

- Patching 后的网格形状和序列长度。
- 合成 8x8 像素玩具图像的 token 序列（走一遍展平 + 投影路径）。
- 按 patch 嵌入、位置嵌入、Transformer 块和头部细分的参数量。
- 目标分辨率下每次前向传播的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的对比表。

运行它。将参数量与公开数字对齐。尝试不同的 patch 大小和分辨率，感受 token 数量的代价。

## 交付成果

本课生成 `outputs/skill-patch-geometry-reader.md`。给定一个 ViT 配置（patch 大小、分辨率、隐藏维度、深度），它会输出 token 数量、参数量和显存估算及依据。每当你为 VLM 选择视觉骨干时使用此技能——它可以避免"token 爆炸导致 LLM 上下文被填满"的意外。

## 练习

1. 计算 Qwen2.5-VL 在原生 1280x720 输入、patch 大小 14 下的 patch token 序列长度。与仅 CLS 表示相比如何？

2. 1080p 帧（1920x1080）在 patch 14 下产生多少个 token？以 30 FPS 播放 5 分钟视频，总共多少个视觉 token？哪种节省最多：池化、帧采样还是 token 合并？

3. 用纯 Python 实现对 patch token 的均值池化。验证对 DINOv2 输出的 196 个 token 做均值池化后，与模型 `forward` 返回的池化嵌入一致。

4. 阅读"Vision Transformers Need Registers"（arXiv:2309.16588）第 3 节。用两句话描述 register token 吸收了什么伪影，以及为什么这对下游密集预测很重要。

5. 修改 `code/main.py` 以支持 patch-n'-pack：给定一组不同分辨率的图像，生成单个打包序列和块对角注意力掩码。完成课程 12.06 后进行验证。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Patch | "16x16 像素方块" | 输入图像的一个固定大小不重叠区域；成为一个 token |
| Patch embedding | "线性投影" | 一个共享的学习矩阵（或步长=P 的 Conv2d），将展平的 patch 像素映射到 D 维向量 |
| CLS token | "类别 token" | 前置的可学习向量，其最终隐藏状态代表整张图像；2026 年已非必需 |
| Register token | "Sink token" | 额外的可学习 token，吸收 ViT 在预训练期间产生的高范数注意力伪影 |
| 位置嵌入 | "位置信息" | 使序列具有位置感知的逐位置向量或旋转；2D-RoPE 是现代默认方案 |
| 网格 | "Patch 网格" | 给定分辨率和 patch 大小下的 (H/P) x (W/P) 二维 patch 数组 |
| NaFlex | "原生灵活分辨率" | SigLIP 2 的特性：单一模型支持多种宽高比和分辨率，无需重新训练 |
| 骨干 | "视觉塔" | 预训练的图像编码器，其 patch token 输出在 VLM 中送入 LLM |
| 池化 | "图像级摘要" | 将 patch token 转为一个向量的策略：CLS、均值、注意力池化或基于 register 的 |
| Patch 14 vs 16 | "更细 vs 更粗的网格" | Patch 14 每张图像产生更多 token，OCR 保真度更高，速度更慢；patch 16 是经典默认值 |

## 延伸阅读

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始 ViT 论文。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，自监督预训练。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大规模自蒸馏，无标签。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 与伪影分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年默认视觉塔。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 经验缩放定律。
