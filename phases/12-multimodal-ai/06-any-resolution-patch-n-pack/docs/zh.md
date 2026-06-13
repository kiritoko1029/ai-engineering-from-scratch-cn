# 任意分辨率视觉：Patch-n'-Pack 与 NaFlex

> 真实图像不是 224x224 的正方形。收据是 9:16，图表是 16:9，医学扫描可能是 4096x4096，手机截图是 9:19.5。2024 年之前的 VLM 答案——将所有东西缩放到固定正方形——丢掉了让 OCR、文档理解和高分辨率场景解析工作的信号。NaViT（Google，2023）证明了你可以将可变分辨率的 patch 打包到单个 Transformer 批次中，配合块对角掩码。Qwen2-VL 的 M-RoPE（2024）完全抛弃了绝对位置表。LLaVA-NeXT 的 AnyRes 将高分辨率图像拼贴为基础 + 子图像。SigLIP 2 的 NaFlex 变体（2025）现在是希望用单一检查点服务每种宽高比的开放 VLM 的默认编码器。本课将端到端实现 patch-n'-pack。

**类型：** 构建
**语言：** Python（标准库，patch 打包器 + 块对角掩码）
**前置要求：** Phase 12 · 01（ViT patch），Phase 12 · 05（LLaVA）
**所需时间：** 约120分钟

## 学习目标

- 将一批可变分辨率图像的 patch 打包成一个序列，并构建块对角注意力掩码。
- 为给定任务在 AnyRes 拼贴（LLaVA-NeXT）、NaFlex（SigLIP 2）和 M-RoPE（Qwen2-VL）之间做出选择。
- 计算 OCR、图表和摄影在不缩放下的 token 预算。
- 说出正方形缩放的三种失败模式：文字变形、内容裁剪、token 浪费在填充上。

## 问题所在

Transformer 需要序列。批次是相同长度序列的堆叠。如果你的图像是 224x224，每次得到 196 个 patch token，不需要填充，任务完成。在 224 上训练，在 224 上推理，再也不用考虑分辨率。

世界不配合。文档是纵向的（8.5x11 英寸，约 2:3）。图表截屏是横向的（16:9）。收据又高又窄（1:3）。医学成像以 2048x2048 或更大尺寸交付。移动设备截屏是 1170x2532（0.46:1）。

2024 年之前的三种选择及其失败原因：

1. 缩放到固定正方形（224x224 或 336x336）。拉伸扭曲了文字和人脸。缩小破坏了图表标签和 OCR 内容。直到 LLaVA-1.5 之前的标准做法。
2. 裁剪到固定宽高比。你丢掉了大部分图像，选择裁剪位置本身就是一个视觉问题。
3. 填充到最长边。解决了扭曲问题，但纵向图像浪费了 50%+ 的 token 在填充上。所有这些填充 token 上的二次方注意力成本。

2024-2025 年的答案：让 Transformer 以图像原生分辨率吃入 patch，并想办法将异构批次打包成一个序列而不浪费计算。

## 概念说明

### NaViT 和 patch-n'-pack

NaViT（Dehghani 等人，2023）是证明这在规模上可行的论文。思路很机械：

1. 对批次中的每张图像，在选定的 patch 大小（比如 14）下计算其原生 patch 网格。
2. 将每张图像的 patch 展平为自己的可变长度序列。
3. 将所有图像的 patch 拼接成批次的一个长序列。
4. 构建块对角注意力掩码，使图像 A 的 patch 只在图像 A 内关注。
5. 携带每 patch 的位置信息（2D RoPE 或分数位置嵌入）。

一个包含三张图像的批次：336x336（576 个 token）、224x224（256 个 token）和 448x336（768 个 token）变成一个 1600 个 token 的序列和 1600x1600 的块对角掩码。没有填充。没有浪费计算。Transformer 处理任意宽高比。

NaViT 还引入了训练期间的分数 patch 丢弃——随机丢弃批次中 50% 的 patch——既正则化又加速训练。SigLIP 2 继承了这一点。

### AnyRes（LLaVA-NeXT）

LLaVA-NeXT 的 AnyRes 是务实的替代方案。给定一张高分辨率图像和固定编码器（336 分辨率的 CLIP 或 SigLIP），拼贴图像：

1. 从预定义集合中选择最匹配图像宽高比的网格布局——（1x1）、（1x2）、（2x1）、（1x3）、（3x1）、（2x2）等。
2. 将整张图像拼贴到网格中；每个拼贴变成一个 336x336 裁剪。
3. 同时生成缩略图：整张图像缩放到 336x336 作为全局上下文 token。
4. 通过冻结的 336 编码器编码每个拼贴。拼接拼贴 token + 缩略图 token。

对于 672x672 图像，2x2 网格加缩略图：4 * 576 + 576 = 2880 个视觉 token。昂贵但有效——LLM 同时看到局部细节和全局上下文。

当你的编码器冻结且只支持一种分辨率时，AnyRes 是首选路径。对于大图像，它会爆炸式增长 token 数量（1344x1344 图像在 4x4 网格下是 9216 + 576 ≈ 9800 个 token，填满大部分 8k LLM 上下文）。

### M-RoPE（Qwen2-VL）

Qwen2-VL 引入了多模态旋转位置嵌入。不同于 NaViT 的分数位置或 AnyRes 的拼贴加缩略图，每个 patch 携带一个 3D 位置（时间、高度、宽度）。query/key 旋转处理任意的 H、W 和时间长度。

M-RoPE 原生支持动态分辨率，无需重新训练。推理时你输入任意 HxW 图像，patch 嵌入器产生 H/14 x W/14 个 token，每个 token 获得其（t=0, r=行, c=列）位置，RoPE 用正确的频率旋转注意力，完成。Qwen2.5-VL 和 Qwen3-VL 延续了这一点。InternVL3 的 V2PE 是相同的想法，但每种模态有可变编码。

不同于 AnyRes，M-RoPE 在原生分辨率下是 O(H x W / P^2) 个 token——没有乘法拼贴开销。不同于 NaViT，它仍然期望每次前向传播处理单张图像。跨分辨率的批次仍然需要在上面加 patch-n'-pack。

### NaFlex（SigLIP 2）

NaFlex 是 SigLIP 2 检查点的原生灵活模式。单一模型在推理时服务多种序列长度（256、729、1024 个 token）。内部在训练期间使用 NaViT 式的 patch-n'-pack 和每 patch 的绝对分数位置。卖点：一个检查点，根据任务在推理时选择你的 token 预算。

对于语义任务（分类、检索），256 个 token。对于 OCR 或图表理解，1024 个 token。无需重新训练。

### 打包掩码

块对角掩码是大多数实现出错的地方。对于覆盖图像 `i=0..B-1`（长度为 `n_i`）的打包序列，总长度 `N_total`，掩码 `M` 形状为 `(N_total, N_total)`，当两个索引落在同一图像的块内时为 1，否则为 0。你可以从累积长度列表构建：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中用 `torch.block_diag` 或显式 gather 一行搞定。FlashAttention 的可变长度路径（`cu_seqlens`）完全跳过掩码，直接使用累积长度张量在序列内关注——对于典型批次比密集掩码快约 10 倍。

### Token 预算

根据任务选择策略：

- OCR / 文档：1024-4096 个 token。SigLIP 2 NaFlex 1024，或 AnyRes 3x3 + 缩略图。
- 图表和 UI：384-448 原生分辨率下 729-1024 个 token。Qwen2.5-VL 动态分辨率配合最大像素限制。
- 自然照片：256-576 个 token 就够了。下游 LLM 看到足够信息。在内容密度高的地方花 token。
- 视频：空间池化后每帧 64-128 个 token，2-8 FPS。课程 12.17 涵盖此内容。

2026 年的生产规则：为每个任务选择最大像素限制，以原生宽高比编码到该限制，打包批次，跳过填充。Qwen2.5-VL 暴露了 `min_pixels` 和 `max_pixels` 作为这个旋钮。

## 开始构建

`code/main.py` 为异构图像批次实现 patch-n'-pack，使用整数像素坐标。它：

- 接收一组（H，W）图像大小。
- 计算每张图像在 patch 大小 14 下的 patch 序列长度。
- 将它们打包成总长度为 `sum(n_i)` 的单个序列。
- 构建块对角注意力掩码（密集型，便于理解）。
- 比较打包成本与正方形缩放和 AnyRes 拼贴。
- 打印混合批次（收据、图表、截屏、照片）的 token 预算表。

运行它。输出的数字就是每一个 2026 年开放 VLM 使用 patch-n'-pack 的原因。

## 交付成果

本课生成 `outputs/skill-resolution-budget-planner.md`。给定混合宽高比工作负载（OCR、图表、照片、视频帧）和总 token 预算，它选择正确的策略（NaFlex、AnyRes、M-RoPE 或固定正方形）并输出每个请求的配置。在为产品规划 VLM 规模时使用此技能——它可以防止默默的 10 倍 token 爆炸摧毁延迟预算。

## 练习

1. 一张收据是 600x1500（1:2.5）。在 patch 大小 14 下，原生分辨率有多少个 token？正方形缩放到 336 后呢？哪个在实践中丢失更多 OCR 准确率？

2. 为一个包含四张图像（长度分别为 256、576、729、1024）的批次构建块对角掩码。验证注意力矩阵为 2585x2585，恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 个非零元素。

3. 对于 1792x896 图像在 patch 14 下，比较：（a）正方形缩放到 336 然后编码，（b）AnyRes 2x1 + 缩略图，（c）M-RoPE 原生。哪个用最少的 token？哪个保留最多的细节？

4. 实现分数 patch 丢弃：给定一个打包序列，均匀随机丢弃 50% 的 token，并相应更新块对角掩码。测量掩码稀疏度的变化。

5. 阅读 Qwen2-VL 论文（arXiv:2409.12191）第 3.2 节。用两句话描述 `min_pixels` 和 `max_pixels` 控制什么以及为什么两个界限都很重要。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Patch-n'-pack | "NaViT 式打包" | 将不同图像的可变长度 patch 序列拼接成一个批次维度 |
| 块对角掩码 | "打包掩码" | 将每张图像的 patch 限制为只关注自身而非打包中邻居的注意力掩码 |
| AnyRes | "LLaVA-NeXT 拼贴" | 将高分辨率图像分割成固定大小拼贴网格加全局缩略图；用固定编码器编码每个拼贴 |
| NaFlex | "SigLIP 2 原生灵活" | 单一 SigLIP 2 检查点在推理时服务 256/729/1024 token 预算，无需重新训练 |
| M-RoPE | "多模态 RoPE" | 3D 旋转位置编码（时间、行、列），处理任意 H、W、T，无需位置表 |
| cu_seqlens | "FlashAttention 打包" | FlashAttention varlen 路径使用的累积长度张量，替代密集块对角掩码 |
| min_pixels / max_pixels | "分辨率界限" | Qwen2.5-VL 每请求旋钮，限制非常小或非常大输入的 token 数量 |
| 视觉 token 预算 | "每张图像多少 token" | 每张图像输出的 patch token 大致数量；设定 LLM 的提示预算和注意力成本 |

## 延伸阅读

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
