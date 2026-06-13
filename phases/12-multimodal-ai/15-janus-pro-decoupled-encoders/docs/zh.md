# Janus-Pro：解耦编码器的统一多模态模型

> 统一多模态模型存在一个不可避免的矛盾。理解任务需要语义特征——SigLIP 或 DINOv2 输出的富含概念级信息的向量。生成任务需要对重建友好的编码——能重新组合为清晰像素的 VQ token。这两个目标在单一编码器中无法兼容。Janus（DeepSeek，2024 年 10 月）和 Janus-Pro（DeepSeek，2025 年 1  月）认为解决方案是停止试图兼顾：解耦两个编码器。在任务之间共享 Transformer 主体，但理解任务走 SigLIP 路径，生成任务走 VQ tokenizer 路径。在 7B 规模下，Janus-Pro 在 GenEval 上超越 DALL-E 3，同时在 MMMU 上匹配 LLaVA。本课将分析为什么双编码器方案在单编码器失败的地方取得了成功。

**类型：** 构建
**语言：** Python（标准库，双编码器路由 + 共享主体信号）
**前置要求：** 第 12 阶段 · 第 13 课（Transfusion），第 12 阶段 · 第 14 课（Show-o）
**所需时间：** 约120分钟

## 学习目标

- 解释为什么单一共享编码器会在理解或生成质量上做出妥协。
- 描述 Janus-Pro 的路由机制：理解任务在输入端使用 SigLIP 特征，生成任务在输入和输出端都使用 VQ token。
- 追踪使 Janus-Pro 在 Janus 未能成功的地方取得突破的数据混合扩展策略。
- 比较解耦式（Janus-Pro）、耦合连续式（Transfusion）和耦合离散式（Show-o）架构。

## 问题所在

统一模型在理解和生成之间共享一个 Transformer 主体。之前的尝试（Chameleon、Show-o、Transfusion）都使用一个视觉 tokenizer 同时服务两个方向。这种 tokenizer 是一种妥协：

- 为重建优化（生成）：VQ-VAE 捕获精细的像素细节，但产生的 token 语义连贯性较弱。
- 为语义优化（理解）：SigLIP 嵌入将"猫"的图像聚集在"猫"的 token 附近，但无法支持良好的重建。

Show-o 和 Transfusion 在某个方向上为这种妥协付出了可见的质量代价。Janus-Pro 的问题是：当任务有不同需求时，为什么还要坚持使用一个 tokenizer？

## 概念说明

### 解耦视觉编码

Janus-Pro 的架构将两个编码器分离开来：

- 理解路径。输入图像 → SigLIP-SO400m → 2 层 MLP → Transformer 主体。
- 生成路径。输入图像（如果以现有图像为条件）→ VQ tokenizer → token ID → Transformer 主体。
- 输出生成。Transformer 预测的图像 token → VQ 解码器 → 像素。

Transformer 主体是共享的。主体上游和下游的所有组件都是任务特定的。

通过提示格式来区分输入：`<understand>` 标签路由到 SigLIP；`<generate>` 路由到 VQ。或者根据任务隐式路由。

### 为什么有效

理解损失获得 SigLIP 特征，这是 CLIP 风格预训练调优过的语义相似度特征。模型的感知基准测试优于 Show-o / Transfusion，因为输入特征更适合该任务。

生成损失获得 VQ token，这是 tokenizer 为重建调优过的编码。图像质量优于 Show-o，因为 VQ 编码能干净地组合回像素。

共享的 Transformer 主体看到两种输入分布（SigLIP 和 VQ），并学会同时处理两者。核心主张是：只要有足够的数据和足够的参数，主体就能吸收这种切换。

### 数据扩展——Janus vs Janus-Pro

Janus（原版，arXiv 2410.13848）引入了解耦机制但规模较小（1.3B 参数，有限数据）。Janus-Pro（arXiv 2501.17811）进行了扩展：

- 7B 参数（vs 1.3B）。
- 第一阶段（对齐）使用 9000 万图文对，从 7200 万提升。
- 第二阶段（统一）使用 7200 万，从 2600 万提升。
- 第三阶段新增 20 万图像生成指令样本。

结果：Janus-Pro-7B 在 MMMU 上匹配 LLaVA（60.3 vs 约 58），在 GenEval 上超越 DALL-E 3（0.80 vs 0.67）。一个开放模型，在统一光谱的两侧都具有竞争力。

### JanusFlow——整流流变体

JanusFlow（arXiv 2411.07975）将 VQ 生成路径替换为整流流生成路径（连续式）。拆分变为 SigLIP 负责理解 + 整流流负责生成。质量上限进一步提升。架构仍然是解耦编码器 + 共享主体。

### 共享主体的职责

Transformer 主体处理统一序列，但面对两种输入分布。其职责是：

- 对于理解：消费 SigLIP 特征 + 文本 token → 自回归输出文本。
- 对于生成：消费文本 token + （可选的图像 VQ token）→ 自回归输出图像 VQ token。

主体在每个块中没有模态特定的权重。它就是你期望在 Qwen 或 Llama 内部找到的那种文本风格 Transformer，加上两个输入适配器。

有趣的是，这意味着 Janus-Pro 的主体可以用预训练 LLM 来初始化。Janus-Pro 确实从 DeepSeek-MoE-7B 初始化。这个选择很重要：LLM 贡献的推理能力是纯从零训练的统一模型难以达到的。

### 与 InternVL-U 的比较

InternVL-U（第 12.10 课）是 2026 年的后续方案。它结合了：

- 原生多模态预训练（InternVL3 骨干）。
- 解耦编码器路由（SigLIP 输入，VQ + 扩散头输出）。
- 统一的理解 + 生成 + 编辑。

InternVL-U 将 Janus-Pro 的架构选择纳入了更大的框架。解耦编码器的理念现在已成为大规模统一模型的默认方案。

### 局限性

解耦编码器增加了架构复杂度。两个 tokenizer 需要训练，两条输入路径需要维护，两套故障模式需要处理。对于不需要生成能力的产品，Janus-Pro 属于过度设计——选择 LLaVA 系列的理解模型即可。

对于不需要理解能力的产品，Janus-Pro 属于大材小用——选择 Stable Diffusion 3 / Flux 模型即可。

对于同时需要两者的产品，Janus-Pro 现在是参考性的开放架构。

## 开始构建

`code/main.py` 模拟 Janus-Pro 的路由机制：

- 两个模拟编码器：SigLIP 风格（生成 256 维语义向量）和 VQ 风格（生成整数码）。
- 一个提示路由器，根据任务标签选择编码器。
- 一个共享主体（占位实现），无论哪个编码器产生的 token 序列都能处理。
- 从第一阶段（对齐）到第三阶段（指令调优）的加权样本调度切换。

打印 3 个示例的路由路径：图像问答、T2I、图像编辑。

## 交付成果

本课生成 `outputs/skill-decoupled-encoder-picker.md`。给定一个需要在前沿质量水平上实现统一生成 + 理解的产品，在 Janus-Pro、JanusFlow 或 InternVL-U 之间做出选择，并给出具体的数据规模建议。

## 练习

1. Janus-Pro-7B 在 GenEval 上超越 DALL-E 3。解释为什么一个 7B 的开放模型能在生成上匹配前沿闭源模型，却在理解上做不到。

2. 实现一个路由函数：给定提示文本，分类为 `understand` 或 `generate`。如何处理"描述一下然后画个草图"这类模糊提示？

3. JanusFlow 用整流流替换了 VQ 路径。Transformer 主体现在输出什么？损失函数有什么变化？

4. 提出第四种任务，Janus-Pro 架构可以通过增加一个解耦编码器来处理。例如：图像分割（DINO 风格）、深度估计（MiDaS 风格）。

5. 阅读 Janus-Pro 第 4.2 节关于数据扩展的内容。哪个数据阶段对 T2I 质量提升贡献最大（相比 Janus）？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 解耦编码 | "两个视觉编码器" | 每个方向使用独立的 tokenizer 或编码器：语义用于理解，重建用于生成 |
| 共享主体 | "一个 Transformer" | 单个 Transformer 处理任一编码器的输出；没有模态特定的权重 |
| SigLIP 用于理解 | "语义特征" | CLIP 系列视觉塔，提供丰富的概念特征但重建能力较差 |
| VQ 用于生成 | "重建编码" | 可以干净地解码回像素的向量量化 token |
| JanusFlow | "整流流变体" | 使用连续流匹配生成头替代 VQ 的 Janus-Pro |
| 路由标签 | "任务标签" | 提示标记（`<understand>` / `<generate>`），用于选择输入编码器 |

## 延伸阅读

- [Wu 等人 — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen 等人 — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma 等人 — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong 等人 — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
