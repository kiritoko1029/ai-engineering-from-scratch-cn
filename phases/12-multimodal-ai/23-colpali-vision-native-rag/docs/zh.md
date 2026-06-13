# ColPali 与视觉原生文档 RAG

> 传统 RAG 将 PDF 解析为文本，切分成块，嵌入块，存储向量。每一步都丢失信号：OCR 丢弃图表数据，分块破坏表格行，文本嵌入忽略图表。ColPali（Faysse 等人，2024 年 7 月）提出了更简单的问题：为什么要提取文本？直接通过 PaliGemma 嵌入页面图像，使用 ColBERT 风格的晚期交互进行检索，保留文档携带的所有布局、图表、字体和格式信号。发布基准测试：在视觉丰富文档上，端到端精度比文本 RAG 高 20-40%。ColQwen2、ColSmol 和 VisRAG 扩展了这一模式。本课将解读视觉原生 RAG 的理念，并构建一个小型 ColPali 风格的索引器。

**类型：** 构建
**语言：** Python（标准库，多向量索引器 + MaxSim 评分器）
**前置要求：** 第 11 阶段（LLM 工程——RAG 基础），第 12 阶段 · 第 05 课（LLaVA）
**所需时间：** 约180分钟

## 学习目标

- 解释双编码器检索（每个文档一个向量）与晚期交互检索（每个文档多个向量）的区别。
- 描述 ColBERT 的 MaxSim 操作以及 ColPali 如何将其从文本 token 泛化到图像 patch。
- 构建一个小型 ColPali 风格的索引器：页面 → patch 嵌入 → 查询词嵌入上的 MaxSim → top-k 页面。
- 在发票 / 财务报告用例上比较 ColPali + Qwen2.5-VL 生成器 vs 文本 RAG + GPT-4。

## 问题所在

文本 RAG 处理 PDF 时丢弃了文档的大部分信息。财务报告的 Q3 营收增长通常在图表中；医学报告的发现在注释图像中；法律合同的签名块是布局事实，而非文本事实。

文本 RAG 管道：

1. PDF → 通过 OCR / pdftotext 转为文本。
2. 文本 → 300-500 token 的块。
3. 块 → 双编码器嵌入（一个向量）。
4. 用户查询 → 嵌入 → 余弦相似度 → top-k 块。
5. 块 + 查询 → LLM。

五个有损步骤。图表未被捕获。表格跨块断裂。多栏布局被展平。图表注释消失。

ColPali 的修复：跳过 OCR，直接嵌入页面图像。使用 ColBERT 风格的晚期交互进行检索，使模型在查询时可以关注细粒度 patch。

## 概念说明

### ColBERT（2020）

ColBERT（Khattab & Zaharia，arXiv:2004.12832）是一种文本检索方法。不是每个文档一个向量，而是每个 token 一个向量。在查询时：

- 查询 token 获得自己的嵌入（N_q 个向量）。
- 文档 token 获得嵌入（N_d 个向量，通常缓存）。
- 分数 = 对查询 token 求和，对文档 token 取最大余弦相似度：Σ_i max_j cos(q_i, d_j)。

这就是 MaxSim 操作。每个查询 token"挑选"其最佳匹配的文档 token。最终分数是求和。

优点：召回率强，处理词级语义。缺点：每个文档 N_d 个向量，存储开销大。

### ColPali

ColPali（Faysse 等人，arXiv:2407.01449）将 ColBERT 模式应用于图像。

- 每个页面由 PaliGemma（ViT + 语言）编码为 patch 嵌入：每页 N_p 个向量。
- 每个用户查询（文本）编码为查询 token 嵌入：N_q 个向量。
- 分数 = Σ_i max_j cos(q_i, p_j)，即查询文本 token 与页面图像 patch 上的 MaxSim。
- 按总分检索 top-k 页面。

文档摄入时：用 PaliGemma 嵌入每个页面，存储所有 patch 嵌入。查询时：嵌入查询 token，对所有存储的页面嵌入计算 MaxSim，返回 top-k 页面。

优点：在视觉丰富文档上端到端胜过文本 RAG 20-40%。每个 patch 向量捕捉局部布局和内容。

缺点：N_p 个 patch × 4 字节浮点数 × D 维向量/页 = 存储增长快。通过 PQ / OPQ 量化缓解。

### ColQwen2 和 ColSmol

ColQwen2（illuin-tech，2024-2025）将 PaliGemma 替换为 Qwen2-VL。更好的基础编码器，更好的检索。

ColSmol 是适用于本地 / 边缘的小规模变体。约 1B 参数的 ColSmol 检索器可在消费级 GPU 上运行。

### VisRAG

VisRAG（Yu 等人，arXiv:2410.10594）是另一种变体：不在 patch 上做 MaxSim，而是用 VLM 将每个页面池化为单一向量，然后用双编码器检索。更快的索引 + 更小的存储，但召回率更弱。

质量 vs 成本的权衡：ColPali 追求质量，VisRAG 追求规模。

### M3DocRAG

M3DocRAG（Cho 等人，arXiv:2411.04952）将多模态检索扩展到多页多文档推理。跨文档检索页面，为 VLM 组合多页上下文。

### ViDoRe——基准测试

ColPali 的配套基准测试。视觉文档检索评估。任务包括财务报告、科学论文、行政文档、医疗记录、手册。指标：nDCG@5。

ColPali-v1 在 ViDoRe 上得分约 80% nDCG@5；同一文档上的文本 RAG 得分约 50-60%。

### 端到端 RAG 管道

视觉原生 RAG 的流程：

1. 摄入：PDF → 页面图像 → PaliGemma 编码 → 存储所有 patch 嵌入。
2. 查询：用户文本 → 查询 token 嵌入 → 对所有索引页面计算 MaxSim → top-k 页面。
3. 生成：top-k 页面图像 + 查询 → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程无 OCR。图表、字体、布局全部流入答案。

### 存储计算

一份 50 页的财务报告，每页 729 个 patch，128 维嵌入：

- ColPali：50 * 729 * 128 * 4 字节 = 约 18 MB 原始，PQ 后约 4 MB。
- 文本 RAG：50 个块 * 768 维 * 4 字节 = 约 150 kB。

ColPali 每文档存储约 30 倍。规模化后，OPQ / PQ 降至约 5-10 倍，通常可以接受。

### 文本 RAG 仍然胜出的场景

- 没有布局信号的纯文本文档（维基文章、聊天记录）。文本 RAG 更简单且存储更便宜。
- 存储成本主导的数百万页面档案。
- 严格的监管要求，需要可提取的 OCR 文本配合检索。

2026 年其他所有场景——财务报告、科学论文、法律合同、医疗记录、UX 文档——视觉原生 RAG 胜出。

## 开始构建

`code/main.py`：

- 玩具 patch 编码器：将"页面"（小型特征向量网格）映射为 patch 嵌入数组。
- MaxSim 评分器：计算查询 token 嵌入集与页面 patch 集之间的 ColBERT 风格分数。
- 索引 5 个玩具页面，运行 3 个查询，返回带分数的 top-k。

## 交付成果

本课生成 `outputs/skill-vision-rag-designer.md`。给定一个文档 RAG 项目，在 ColPali / ColQwen2 / VisRAG / 文本 RAG 之间做出选择并确定存储规模。

## 练习

1. 一份 200 页的年度报告，每页 729 个 patch，128 维嵌入，4 字节浮点数。计算原始存储和 PQ 压缩（8 倍）后的存储。

2. MaxSim 是 Σ_i max_j cos(q_i, p_j)。这个求和捕捉了什么简单平均相似度没有捕捉到的？

3. ColPali 以 patch 集索引页面。如果改为词级索引（如 ColBERT 所做的），会有什么变化？权衡是什么？

4. 为 100 万页语料库设计端到端管道，延迟预算为每查询 500ms。选择 ColQwen2 / VisRAG 并论证。

5. 阅读 M3DocRAG（arXiv:2411.04952）。描述多页注意力模式以及它与单页 ColPali 检索的区别。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 晚期交互 | "ColBERT 风格" | 使用每 token 或每 patch 嵌入 + MaxSim 的检索，非单一文档向量 |
| MaxSim | "patch 上取最大" | 对每个查询 token，选择最高相似度的文档 token；跨查询求和 |
| 双编码器 | "单向量" | 每个文档一个向量；更快但丢失粒度 |
| 多向量 | "每文档多向量" | 每文档/页面存储 N_p 个向量；存储成本增长但召回率提升 |
| patch 嵌入 | "页面特征" | VLM 编码器每个图像 patch 一个向量，按页面缓存 |
| ViDoRe | "视觉文档基准" | ColPali 的视觉文档检索基准套件 |
| PQ 量化 | "乘积量化" | 在保持向量相似度的同时将存储缩小约 8 倍的压缩方法 |

## 延伸阅读

- [Faysse 等人 — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu 等人 — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho 等人 — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
