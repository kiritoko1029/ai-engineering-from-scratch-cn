# 从 CLIP 到 BLIP-2——Q-Former 作为模态桥梁

> CLIP 对齐了图像和文本，但无法生成描述、回答问题或进行对话。BLIP-2（Salesforce，2023）用一个小型可训练桥梁解决了这个问题：32 个可学习的 query 向量通过交叉注意力关注冻结 ViT 的特征，然后直接插入冻结 LLM 的输入流中。1.88 亿参数的桥梁将 110 亿参数的 LLM 连接到 ViT-g/14。2026 年之前所有基于适配器的 VLM——MiniGPT-4、InstructBLIP、LLaVA 的变体——都是它的后裔。本课将解读 Q-Former 的架构，解释其两阶段训练，并构建一个玩具版本，将视觉 token 送入冻结的文本解码器。

**类型：** 构建
**语言：** Python（标准库，交叉注意力 + 可学习 query 演示）
**前置要求：** Phase 12 · 02（CLIP），Phase 7（Transformer）
**所需时间：** 约180分钟

## 学习目标

- 解释为什么在冻结的视觉编码器和冻结的 LLM 之间放置可训练瓶颈，在成本和稳定性上优于端到端微调。
- 实现一个交叉注意力块，其中一组固定的可学习 query 关注外部图像特征。
- 走一遍 BLIP-2 的两阶段预训练：表征阶段（ITC + ITM + ITG）然后是生成阶段（使用冻结解码器的 LM 损失）。
- 将 Q-Former 与 LLaVA 中使用的更简单的 MLP 投影器进行比较，并论证各自何时胜出。

## 问题所在

你有一个冻结的 ViT，每张图像产生 256 个维度为 1408 的 patch token。你有一个冻结的 70 亿参数 LLM，期望维度为 4096 的 token 嵌入。显而易见的桥梁——一个从 1408 到 4096 的线性层——能用，但将所有 256 个 patch token 送入 LLM 的上下文，每张图像要消耗 256 个额外 token。在一个 32 张图像的批次中，仅视觉模态就占用了 8192 个 token。

BLIP-2 的问题：你能否将 256 个 token 的图像表示压缩到更少的 token（比如 32 个），同时保留足够的信息让 LLM 能够生成描述、回答问题和推理图像？而且你能否在不触及冻结骨干的情况下训练这个桥梁，将训练成本控制在仅桥梁参数的范围内？

答案是 Q-Former。32 个可学习的"query"向量交叉关注 ViT 的 patch token，产生 32 个 token 的视觉摘要供 LLM 消费。总共 1.88 亿参数。在首次接触 LLM 之前，用对比、匹配和生成目标进行训练。

## 概念说明

### 可学习 query

Q-Former 的核心技巧：不让 LLM 的文本 token 关注图像 patch，而是引入一组新的 32 个可学习 query 向量 `Q`，让*它们*关注图像 patch。这些 query 是模型的参数——在训练期间学习，对每张图像使用相同的 32 个 query。

交叉注意力之后，每个 query 持有图像的压缩摘要——"描述主要物体"、"描述背景"、"数物体"等。query 并不真正专门化于语义标签；它们学习使下游损失下降的任何编码。

### 架构

Q-Former 是一个小型 Transformer（12 层，约 1 亿参数），有两条路径：

1. Query 路径：32 个 query 向量流经自注意力（彼此之间），然后对冻结 ViT 的 patch token 做交叉注意力，然后 FFN。
2. 文本路径：类似 BERT 的文本编码器与 query 路径共享自注意力和 FFN 权重。文本路径的交叉注意力被禁用。

训练时两条路径都运行。query 和文本通过共享的自注意力交互，这意味着 query 可以根据文本条件执行需要文本的任务（ITM、ITG）。VLM 交接推理时，只有 query 流过，产出 32 个视觉 token。

### 两阶段训练

BLIP-2 分两阶段预训练：

阶段 1：表征学习（无 LLM）。三个损失：
- ITC（图文对比）：池化 query token 与文本 CLS token 之间的 CLIP 式对比损失。
- ITM（图文匹配）：二分类器——这对图文是否匹配？难负样本挖掘。
- ITG（基于图像的文本生成）：文本上的因果 LM 头，以 query 为条件。迫使 query 编码可生成文本的内容。

只有 Q-Former 训练。ViT 冻结。不涉及 LLM。

阶段 2：生成学习。接入冻结的 LLM（OPT-2.7B 或 Flan-T5-XL 等）。通过一个小型线性层将 32 个 query 输出投影到 LLM 的嵌入维度。将它们前置到文本提示。仅训练线性投影和 Q-Former，对拼接的提示 + 图像 + 描述序列施加 LM 损失。

阶段 2 之后，Q-Former + 投影就是完整的视觉适配器。推理时：图像 → ViT → Q-Former → 线性投影 → 前置到文本 → 冻结的 LLM 生成输出。

### 参数经济学

BLIP-2 使用 ViT-g/14（11 亿，冻结）+ OPT-6.7B（67 亿，冻结）+ Q-Former（1.88 亿，训练）= 总计 80 亿，训练 1.88 亿。仅 Q-Former 约占全栈参数的 2.4%。训练成本反映这一点：在几张 A100 上几天，而端到端需要几周。

质量：BLIP-2 在零样本 VQA 上匹配或击败 Flamingo-80B，同时小 50 倍。桥梁起作用了。

### InstructBLIP 和指令感知 Q-Former

InstructBLIP（2023）扩展了 Q-Former，增加了一个额外输入：指令文本本身。在交叉注意力时，query 现在可以访问图像 patch 和指令。query 可以按指令专门化（"数汽车"、"描述情绪"），而不是学习单一的固定摘要。在保留任务上有基准增益。

### MiniGPT-4 和仅投影器方法

MiniGPT-4 保留了 Q-Former 但只训练输出线性投影，同时冻结其他一切。便宜，但质量有代价——query 是 BLIP-2 的，不是你的。适合快速迭代，不是最佳架构。

### 为什么 LLaVA 选择了更简单的方式

LLaVA（2023，课程 12.05）用一个普通的 2 层 MLP 替代了 Q-Former，将每个 ViT patch token 投影到 LLM 空间——对于 24x24 网格是每张图像 576 个 token，全部送入 LLM。压缩率更差，但让 LLM 可以关注原始 patch。这在当时有争议；到 2023 年底它成为主流，因为视觉指令数据（LLaVA-Instruct-150k）证明了 MLP 可以被训练以保留足够的信号。权衡是：LLaVA 的上下文填满更快，但它天然适合多图像和视频。

到 2026 年，领域分裂了：Q-Former 在 token 预算重要的场景（长视频、多图像）中存活；MLP 投影器在每 token 原始质量优先的场景中占主导。

### 门控交叉注意力：Flamingo，祖先

Flamingo（课程 12.04）早于 BLIP-2，使用相同的交叉注意力思想，但是在每个冻结的 LLM 层中，而非作为单一桥梁。BLIP-2 证明了你可以压缩到仅输入层并仍然有效。Gemini 和 Idefics 结合了两者：交错输入 token 加可选的门控交叉注意力用于上下文少样本。

### 2026 年的后裔

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及出于 token 预算考虑的大多数视频-语言模型。
- Perceiver resampler：Flamingo 的变体（课程 12.04）；Idefics 系列、Eagle、OmniMAE。
- MLP 投影器：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- 注意力池化：VILA、PaliGemma。

四种方案都有效。决定性的问题是你受限于 token 预算还是每 token 质量。

## 开始构建

`code/main.py` 构建了一个标准库的 Q-Former 风格交叉注意力：

1. 模拟 256 个图像 patch token（维度 128）。
2. 实例化 32 个可学习 query（维度 128）。
3. 运行缩放点积交叉注意力（Q 来自 query，K/V 来自 patch）。
4. 通过线性层投影到 LLM 维度（512）。
5. 输出 32 个 LLM 就绪的视觉 token。

所有数学用纯 Python（向量的嵌套循环）。玩具级别但形状正确。注意力权重矩阵被打印出来，让你看到每个 query 从哪些 patch 中提取信息。

## 交付成果

本课生成 `outputs/skill-modality-bridge-picker.md`。给定目标 VLM 配置（视觉编码器 token 数、LLM 上下文预算、部署约束、质量目标），它推荐 Q-Former vs MLP vs Perceiver resampler，并附上简要论证和每种桥梁的参数量估算。

## 练习

1. 在 PyTorch 中实现交叉注意力块。验证当有 32 个 query 和 256 个 key/value 时，注意力权重矩阵为 32 x 256，每行 softmax 后和为 1。

2. 在 BLIP-2 阶段 1 中，Q-Former 同时运行三个损失：ITC、ITM、ITG。用伪代码写出每个的前向签名。哪个需要文本编码器路径处于激活状态？

3. 比较参数量：Q-Former（12 层，768 隐藏维度）vs 2 层 MLP 投影器（1408 → 4096，两层）。在什么 LLM 规模下，1.88 亿 Q-Former 的成本能在训练效率上收回？

4. 阅读 BLIP-2 论文（arXiv:2301.12597）第 3.2 节关于 Q-Former 的初始化。解释为什么从 BERT-base 初始化（而非随机）能加速收敛。

5. 对于以 1 FPS 采样到 60 帧的 10 分钟视频，计算每帧 token 成本：（Q-Former → 32 token/帧）vs（MLP 投影器 → 576 token/帧）。哪个能放入 128k token 的 LLM 上下文窗口？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Q-Former | "查询 Transformer" | 带有 32 个可学习 query 向量的小型 Transformer，交叉关注冻结的 ViT 特征 |
| 可学习 query | "视觉的软提示" | 一组固定参数，作为交叉注意力的 query 侧；每个模型学习一次，所有输入共享 |
| 交叉注意力 | "Q 来自这边，K/V 来自那边" | query、key 和 value 来自不同来源的注意力；query 从 ViT patch 中提取信息的方式 |
| ITC | "图文对比" | 应用于 Q-Former 池化 query 与文本 CLS 的 CLIP 式损失 |
| ITM | "图文匹配" | 基于难负样本挖掘对的二分类器；迫使 query 区分细粒度不匹配 |
| ITG | "基于图像的文本生成" | 以 query 为条件生成文本的因果 LM 损失；迫使 query 编码可解码为文本的内容 |
| 两阶段预训练 | "先表征后生成" | 阶段 1 单独训练 Q-Former（ITC/ITM/ITG）；阶段 2 接入冻结 LLM，仅训练投影 + Q-Former |
| 冻结骨干 | "不微调" | 视觉编码器和 LLM 权重固定；只有桥梁训练 |
| 投影头 | "线性到 LLM 维度" | 将 Q-Former 输出映射到 LLM 嵌入维度的最终线性层 |
| Perceiver resampler | "Flamingo 的版本" | 类似的可学习 query 交叉注意力，Flamingo 在每层使用而非作为单一桥梁 |

## 延伸阅读

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心论文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 前身，包含 ITC/ITM/ITG 三件套。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — "先对齐后融合"——阶段 1 训练的概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — 指令感知 Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — 仅投影器方法。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 可学习 query 交叉注意力的通用架构。
