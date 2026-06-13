# 开放权重 VLM 方案：什么真正重要

> 2024-2026 年的开放权重 VLM 文献是一片消融实验表格的森林。Apple 的 MM1 测试了图像编码器、连接器和数据混合的 13 种组合。Allen AI 的 Molmo 证明了详细的人工描述优于 GPT-4V 蒸馏。Cambrian-1 运行了 20+ 种编码器对比。Idefics2 正式化了五维设计空间。Prismatic VLMs 在受控基准上比较了 27 种训练方案。从所有这些噪音中，一小组结果跨论文成立：图像编码器比连接器架构重要，数据混合比两者都重要，详细的人工描述优于蒸馏的合成数据。本课解读这些表格，省去你自己阅读的麻烦。

**类型：** 学习 + 实验
**语言：** Python（标准库，消融表解析器 + 方案选择器）
**前置要求：** Phase 12 · 05（LLaVA 基线）
**所需时间：** 约180分钟

## 学习目标

- 说出五维 VLM 设计空间：图像编码器、连接器、LLM、数据混合、分辨率策略。
- 阅读 MM1 / Idefics2 / Cambrian-1 消融表并预测哪个旋钮能改善给定基准。
- 给定计算预算和任务组合，为新 VLM 选择方案（编码器、连接器、数据、分辨率）。
- 解释为什么在相同 token 数量下，详细的人工描述优于 GPT-4V 蒸馏。

## 问题所在

数百个开放权重 VLM 存在。"好"和"最先进"之间的差距大多不在架构。而在于数据、分辨率策略和编码器选择。当你的模型表现不佳时，知道先转动哪个旋钮，可以省去一个 500 万 GPU 小时的错误。

2023 年浪潮（LLaVA-1.5、InstructBLIP、MiniGPT-4）使用描述对预训练 + LLaVA-Instruct-150k。良好的基线。MMMU 大约封顶在 35%。

2024 年浪潮（MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs）进行了详尽的消融实验。结果令人惊讶且实用。

## 概念说明

### 五维设计空间

Idefics2（Laurençon 等人，2024）命名了这些维度：

1. 图像编码器。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。编码器在 patch 大小、分辨率和预训练目标上各有不同。
2. 连接器。MLP（2-4 层）、Q-Former（32 个 query + 交叉注意力）、Perceiver Resampler（64 个 query）、C-Abstractor（卷积 + 双线性池化）。
3. 语言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是主要的参数成本。
4. 训练数据。描述对（CC3M、LAION）、交错（OBELICS、MMC4）、指令（LLaVA-Instruct、ShareGPT4V、PixMo、Cauldron）。
5. 分辨率策略。固定 224/336/448、AnyRes、原生动态。训练中递增或恒定。

每个生产 VLM 在每个维度上都做了选择。MMMU 分数的大部分方差由维度 1、4 和 5 解释——而非你选择的连接器。

### 维度 1：编码器 > 连接器

MM1 第 3.2 节表明：从 CLIP ViT-L/14 换到 SigLIP SO400m/14 增加了 3+ 个 MMMU 点。将连接器从 MLP 换到 Perceiver Resampler 增加不到 1 个点。Idefics2 复现了：SigLIP > CLIP，在相同 token 数下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的"Cambrian 视觉编码器对决"（Tong 等人，2024）在以视觉为中心的基准（CV-Bench）上运行了 20+ 种编码器。排行榜顶部是 DINOv2 和 SigLIP 的混合；CLIP 处于中游；ImageBind 和 ViT-MAE 更低。CLIP ViT-L 到 DINOv2 ViT-g/14 在 CV-Bench 上的差距约 5-7 个点。

2026 年开放 VLM 的默认编码器是 SigLIP 2 SO400m/14 用于语义 + 密集特征，有时与 DINOv2 ViT-g/14 特征拼接（Cambrian 的"空间视觉聚合器"这样做）。

### 维度 2：连接器设计无关紧要

MM1、Idefics2、Prismatic 和 MM-Interleaved 都得出了相同结论：在固定视觉 token 数下，连接器架构几乎不重要。在均值池化 patch 上的 2 层 MLP 在相同 token 预算下与 32 个 query 的 Q-Former 相差不到 1 个点。

真正重要的是 token 数量。更多视觉 token = 更多 LLM 计算 = 更好性能，到一定程度后收益递减。每张图像 64 个 token 对 OCR 来说太少。576-1024 个 token 是大多数开放 VLM 的最佳点。2048+ 仅对文档和图表有帮助。

Q-Former vs MLP 是成本问题，不是质量问题：Q-Former 将 token 封顶在 32-64 个，无论图像分辨率；MLP 输出所有 patch token。对于高分辨率输入，Q-Former 节省 LLM 上下文；对于低分辨率，差异是噪音。

### 维度 3：LLM 大小设定天花板

将 LLM 从 7B 翻倍到 13B 在每篇 VLM 论文中都可靠地增加 2-4 个 MMMU 点。到 70B 时大多数基准饱和。VLM 的多模态推理天花板就是 LLM 的文本推理天花板——视觉编码器只能喂给它，不能替它推理。

这就是为什么 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上碾压：语言大脑巨大。7B VLM 无法通过巧妙的连接器设计替代 70B VLM。

### 维度 4：数据——详细人工描述优于蒸馏

Molmo + PixMo（Deitke 等人，2024）是每个人都应该读的 2024 年结果。Allen AI 让人工标注员用 1-3 分钟的密集语音转文字描述图像，产生了 71.2 万张密集描述的图像。训练数据中完全没有 GPT-4V 蒸馏。

Molmo-72B 在 11 个基准中的 11 个上击败了 Llama-3.2-90B-Vision。差距不在架构——而在描述质量。详细的人工描述每张图像包含的信息量是短网络描述的 5-10 倍，并且在 GPT-4V 蒸馏产生幻觉的地方保持事实准确。

ShareGPT4V（Chen 等人，2023）和 Cauldron（Idefics2）用混合人工 + GPT-4V 描述遵循了相同的策略。趋势清晰：对于 2026 年前沿，描述密度 > 描述数量 > 蒸馏便利性。

### 维度 5：分辨率及其策略

Idefics2 的消融实验：384 → 448 增加 1-2 个点。448 → 980 配合图像分割（AnyRes）在 OCR 基准上再增加 3-5 个点。固定分辨率训练在中等准确率上趋于平稳；分辨率递增（从 224 开始，到 448 或原生结束）训练更快且最终更高。

Cambrian-1 运行了分辨率 vs token 的权衡：在固定计算量下，你可以用更低分辨率获得更多 token，或用更高分辨率获得更少 token。更高分辨率对 OCR 胜出；更低分辨率更多 token 对通用场景理解胜出。

2026 年生产方案：阶段 1 在 384 固定训练，阶段 2 动态分辨率最高到 1280 用于 OCR 密集任务。

### Prismatic 受控比较

Prismatic VLMs（Karamcheti 等人，2024）是控制了所有维度的论文。相同的 13B LLM，相同的指令数据，相同的评估——每次只有一个维度变化。结果：

- 每张图像视觉 token 数解释了约 60% 的方差。
- 编码器选择解释了约 20%。
- 连接器架构解释了约 5%。
- 其余（数据混合、调度器、学习率）约 15%。

这是粗略的分解，但它是文献中"我应该先消融什么"的最清晰答案。

### 2026 年选择器

根据证据，2026 年新项目的默认开放 VLM 方案：

- 编码器：原生分辨率配合 NaFlex 的 SigLIP 2 SO400m/14，如果需要分割/定位则拼接 DINOv2 ViT-g/14 密集特征。
- 连接器：patch token 上的 2 层 MLP。除非受 token 限制，否则跳过 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，7B 用于成本考量，70B 用于质量，根据目标延迟选择。
- 数据：PixMo + ShareGPT4V + Cauldron，补充任务特定指令数据。
- 分辨率：动态（长边最小 256，最大 1280 像素）。
- 调度：阶段 1 对齐（仅投影器），阶段 2 完全微调，阶段 3 任务特定微调。

这些默认值中的每一个都可以追溯到本课末尾引用论文中的实测消融实验。

## 开始构建

`code/main.py` 是一个消融表解析器和方案选择器。它编码了 MM1 和 Idefics2 的消融表（精简版），让你查询：

- "给定预算 X 和任务 Y，什么方案胜出？"
- "如果我在 7B Llama 上将 SigLIP 换成 CLIP，预期的 MMMU 变化是多少？"
- "为了 80% 置信度的答案，我应该先消融哪个维度？"

输出是一个排序的方案列表，附带预期基准变化和"先消融"推荐。

## 交付成果

本课生成 `outputs/skill-vlm-recipe-picker.md`。给定目标任务组合、计算预算和延迟目标，它输出完整方案（编码器、连接器、LLM、数据混合、分辨率策略），并引用支持每个选择的消融实验。阻止工程师在每次新 VLM 项目启动时重新发明 Idefics2 消融表。

## 练习

1. 阅读 MM1 第 3.2 节。对于固定 2B LLM、5000 万图像预算，哪种编码器胜出？在 13B LLM 下答案会翻转吗？为什么？

2. Cambrian-1 发现拼接 DINOv2 + SigLIP 在以视觉为中心的基准上优于单独使用，但在 MMMU 上没有增益。预测哪些基准提升，哪些持平。

3. 你的目标是在 2B LLM 上的移动 UI 代理。选择编码器、连接器、分辨率和数据混合。用具体的消融表论证每个选择。

4. Molmo 发布了 4B 和 72B 模型。4B 与封闭 7B VLM 竞争力相当；72B 在 11/11 基准上击败 Llama-3.2-90B-Vision。这告诉你关于 LLM 大小平台期假说的什么？

5. 设计一个消融表来隔离 7B VLM 上数据混合质量与编码器质量。最少需要多少次训练运行？提出四个维度的设置。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 消融 | "转动一个旋钮" | 运行多次训练，恰好在一个设计空间维度上不同，其余保持不变 |
| 连接器 | "桥梁" / "投影器" | 将视觉编码器输出映射到 LLM token 空间的可训练模块（MLP、Q-Former、Perceiver） |
| 详细人工描述 | "密集描述" | 多句人工撰写的描述（通常 80-300 个 token），比网络替代文本更丰富 |
| 蒸馏 | "GPT-4V 描述" | 由更强的私有 VLM 生成的训练数据；便利但容易继承幻觉 |
| AnyRes / 动态分辨率 | "高分辨率路径" | 通过拼贴或 M-RoPE 输入大于编码器原生分辨率图像的策略 |
| 分辨率递增 | "课程" | 从低分辨率开始并逐步增加的训练策略，加速对齐学习 |
| 以视觉为中心的基准 | "CV-Bench / BLINK" | 强调细粒度视觉感知而非语言密集推理的评估 |
| PixMo | "Molmo 的数据" | Allen AI 的 71.2 万张密集描述图像数据集；人工语音转录为密集描述 |

## 延伸阅读

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
