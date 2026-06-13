# LLaVA 与视觉指令微调

> LLaVA（2023 年 4 月）是地球上被复制最多的多模态架构。它用 2 层 MLP 替代了 BLIP-2 的 Q-Former，用朴素的 token 拼接替代了 Flamingo 的门控交叉注意力，并在 GPT-4 从纯文本描述生成的 15.8 万条视觉指令轮次上训练。2023 到 2026 年间任何构建 VLM 的从业者都构建了 LLaVA 的某种变体。LLaVA-1.5 加入了 AnyRes。LLaVA-NeXT 提升了分辨率。LLaVA-OneVision 在一个方案中统一了图像、多图像和视频。本课将解读方案，实现投影器，并解释"为什么更简单反而赢了"。

**类型：** 构建
**语言：** Python（标准库，投影器 + 指令模板构建器）
**前置要求：** Phase 12 · 02（CLIP），Phase 11（LLM 工程——指令微调）
**所需时间：** 约180分钟

## 学习目标

- 构建一个 2 层 MLP 投影器，将 ViT patch 嵌入（维度 1024）映射到 LLM 的嵌入维度（维度 4096）。
- 走一遍 LLaVA 两阶段方案：（1）在 55.8 万对图文对上对齐投影器，（2）在 15.8 万条 GPT-4 生成的轮次上进行视觉指令微调。
- 构建一个 LLaVA 格式的提示，包含图像 token 占位符、系统提示和用户/助手轮次。
- 解释为什么社区从 Q-Former 转向 MLP，尽管 Q-Former 在 token 预算上有优势。

## 问题所在

BLIP-2 的 Q-Former（课程 12.03）将图像压缩为 32 个 token。干净、高效、适合基准测试。但它有两个问题。

首先，Q-Former 可训练但其损失不是最终任务。阶段 1 训练 ITC+ITM+ITG。阶段 2 训练 LM 损失。query 学习某种中间表示，然后 LLM 必须解码它。信息在瓶颈中丢失。

其次，Q-Former 需要 1.88 亿参数，在 LLaVA 的 2023 年规模下你必须与目标 LLM 协同设计。换 LLM，重新训练 Q-Former。换视觉编码器，重新训练。每种组合都是一个独立的研发项目。

LLaVA 的答案简单得令人尴尬：取 ViT 的 576 个 patch token，每个通过 2 层 MLP（`1024 → 4096 → 4096`），然后将全部 576 个倒入 LLM 的输入序列。没有瓶颈。没有在奇怪目标上的阶段 1 预训练。只是在直接的 LM 损失上训练 MLP。

数据从哪来？LLaVA 的第二个洞见：用 GPT-4（纯文本）生成指令数据。将 COCO 描述和边界框数据喂给 GPT-4，让它产生对话、描述和复杂推理问题。免费获得 15.8 万条指令-回复轮次。无需人工标注。

结果：一个在 8 张 A100 上运行一天的 VLM，在 MMMU 上击败了 Flamingo，并发布了社区可以扩展的开放检查点。到 2023 年底，它已衍生出 50+ 个分支。

## 概念说明

### 架构

LLaVA-1.5 13B 版本：
- 视觉编码器：CLIP ViT-L/14 @ 336（阶段 1 冻结，阶段 2 可选解冻）。
- 投影器：带 GELU 激活的 2 层 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-13B（后来是 Llama-3.1-8B）。

对一张图像 + 文本提示的前向传播：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

图像占用 LLM 上下文的 576 个 token。在 2048 上下文下，留给文本 1472 个 token。在 32k 上下文下，这只是舍入误差。

### 阶段 1：投影器对齐

冻结 ViT。冻结 LLM。仅训练 2 层 MLP。数据集：55.8 万对图文对（LAION-CC-SBU）。损失：在投影的图像 token 条件下的描述语言建模。

在 batch 128 下一个 epoch 几个小时就能完成。投影器学习将 ViT 空间映射到 LLM 空间。无任务特定监督。

### 阶段 2：视觉指令微调

解冻投影器（仍可训练）。解冻 LLM（通常完全解冻，有时用 LoRA）。在 15.8 万条视觉指令轮次上训练。

指令数据是关键。Liu 等人通过以下方式生成：
1. 取一张 COCO 图像。
2. 提取文本描述（5 条人工描述 + 边界框列表）。
3. 用三个提示模板发送给 GPT-4：
   - 对话："生成用户和助手之间关于这张图像的来回对话。"
   - 详细描述："给出图像的丰富、详细描述。"
   - 复杂推理："提出一个需要推理图像的问题，然后回答它。"
4. 将 GPT-4 的输出解析为（指令，回复）对。

这些都不直接接触图像——只有文本描述。GPT-4 幻觉出合理的图像内容。有一些噪声，但有效：15.8 万条轮次足以解锁对话能力。

### 为什么社区复制了这个

- 没有阶段 1 特定的损失需要调优。全程 LM 损失。
- 投影器几小时训练完成，不是几天。
- LLM 可以替换（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3），只需重新训练投影器。
- 视觉指令数据流水线使用 GPT-4，为新领域重新生成成本低廉。

### LLaVA-1.5 和 LLaVA-NeXT

LLaVA-1.5（2023 年 10 月）新增：
- 学术任务数据（VQA、OKVQA、RefCOCO）混入指令微调。
- 更好的系统提示。
- 2048 → 32k 上下文。

LLaVA-NeXT（2024 年 1 月）新增：
- AnyRes：将高分辨率图像分割成 2x2 或 1x3 的 336x336 裁剪网格，加一个全局低缩略图。每个裁剪变成 576 个 token；每张图像总计约 2880 个视觉 token。OCR 和图表任务大幅跃升。
- 更好的指令数据混合，加入 ShareGPT4V（高质量 GPT-4V 描述）。
- 更强的基础 LLM（Mistral-7B、Yi-34B）。

### LLaVA-OneVision

课程 12.08 深入讲解 OneVision。简短版本：相同的投影器，但用覆盖单图像、多图像和视频的课程训练，共享视觉 token 预算。

### 与 Q-Former 的对比

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每张图像视觉 token | 32 | 576（基础）或 2880（AnyRes） |
| 可训练参数 | 1.88 亿 + LM | 4000 万 + LM |
| 阶段 1 损失 | ITC+ITM+ITG | 仅 LM |
| LLM 替换 | 需要重新训练 | 换个 LLM 只需少量重新训练 |
| 多图像 | 棘手 | 自然（拼接） |
| 视频 | 棘手 | 自然（逐帧拼接） |
| Token 预算 | 小 | 大 |

MLP 在简洁性和 token 灵活性上胜出。Q-Former 在 token 预算上胜出。到 2023 年底，token 预算不再是约束（LLM 上下文增长到 32k-128k+），简洁性占据主导。

### 提示格式

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是占位符 token。在分词之前，它被替换为 576 个视觉 token（AnyRes 下为 2880 个）。分词器看到的序列比训练时略长，但 LLM 能处理这种新输入，因为阶段 1 已经教会它了。

### 参数经济

LLaVA-1.5-7B 分解：
- CLIP ViT-L/14 @ 336：3.03 亿（阶段 1 冻结，阶段 2 常解冻）。
- 投影器（2 层线性）：约 2200 万可训练。
- Llama-7B：70 亿。
- 总计：73 亿参数。阶段 2 可训练：全部 70 亿 + 2200 万投影器。

阶段 2 训练成本：8xA100 上约 20 小时。这是关键数字——一天，一个节点，可复现。这就是 LLaVA 传播的原因。

## 开始构建

`code/main.py` 实现了：

1. 纯 Python 的 2 层 MLP 投影器（维度 16 → 32 → 32，玩具规模）。
2. 提示构建流水线：系统提示 + `<image>` 替换为 N 个投影 token + 用户轮次 + 助手生成占位符。
3. 可视化 576 个视觉 token 块在 LLM 上下文中的样子（2k / 32k / 128k 上下文消耗的百分比）。

## 交付成果

本课生成 `outputs/skill-llava-vibes-eval.md`。给定一个 LLaVA 系列检查点，它运行一个 10 提示的直觉评估套件（3 个描述、3 个 VQA、2 个推理、2 个拒绝），并输出人类可读的评分卡。不是基准测试；而是确认投影器和 LLM 连接良好的冒烟测试。

## 练习

1. 计算 `1024 → 4096 → 4096` 的 2 层 MLP 投影器的可训练参数量。包含 GELU 和偏置，它占 LLaVA-13B 的比例是多少？

2. 为一个"拒绝"场景构建 LLaVA 提示——图像包含一个私人个体。写出预期的助手回复。为什么 LLaVA 应该零样本拒绝这个，需要什么训练数据来强化拒绝？

3. 阅读 LLaVA-NeXT 博客的 AnyRes 部分。计算 1344x672 图像在 AnyRes 下的视觉 token 数。与 336x336 下基础 576 个 token 比较。

4. LLaVA 阶段 1 投影器在描述上用 LM 损失训练。如果跳过阶段 1 直接进入阶段 2（视觉指令微调）会怎样？引用 Prismatic VLMs 消融实验（arXiv:2402.07865）的答案。

5. LLaVA-Instruct-150k 使用 GPT-4 配合 COCO 描述生成指令。对于新领域（医学 X 光、卫星图像），描述生成领域指令的四步数据流水线。每一步可能出什么问题？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 投影器 | "MLP 桥梁" | 带 GELU 的 2 层 MLP，将 ViT 维度映射到 LLM 维度 |
| 图像 token | "<image> 占位符" | 推理前被 N 个投影视觉 token 替换的提示标记 |
| 视觉指令微调 | "LLaVA 阶段 2" | 在 GPT-4 生成的（图像，指令，回复）三元组上训练 |
| 阶段 1 对齐 | "投影器预训练" | 冻结 ViT 和 LLM，用 LM 损失在描述上训练投影器 |
| AnyRes | "多裁剪拼贴" | 将高分辨率图像分割成裁剪网格，拼接每个裁剪的视觉 token |
| LLaVA-Instruct | "GPT-4 生成的" | 从 COCO 描述 + GPT-4 合成的 15.8 万条指令-回复对 |
| 视觉编码器冻结 | "骨干锁定" | CLIP 权重在阶段 1 不更新，阶段 2 有时也不更新 |
| ShareGPT4V | "更好的描述" | GPT-4V 生成的 100 万条密集描述，用于更高质量的对齐 |
| VQA | "视觉问答" | 回答关于图像的自由形式问题的任务 |
| Prismatic VLMs | "设计空间论文" | Karamcheti 2024 消融实验，系统测试投影器和数据选择 |

## 延伸阅读

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 论文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集描述数据集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 设计空间消融实验。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 统一单图像、多图像、视频。
