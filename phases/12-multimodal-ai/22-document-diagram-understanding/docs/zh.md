# 文档与图表理解

> 文档不是照片。PDF、科学论文、发票或手写表格具有布局、表格、图表、脚注、标题和语义结构，这些是普通图像理解无法捕捉的。VLM 之前的方案是流水线式的：Tesseract OCR + LayoutLMv3 + 表格提取启发式规则。VLM 浪潮用无 OCR 模型替代了它——Donut（2022）、Nougat（2023）、DocLLM（2023）——直接输出结构化标记。到 2026 年，前沿方案就是"把页面图像以 2576px 原生分辨率喂给 Claude Opus 4.7"，结构化标记输出随之免费获得。本课将走完文档 AI 的三个时代。

**类型：** 构建
**语言：** Python（标准库，布局感知文档解析器骨架）
**前置要求：** 第 12 阶段 · 第 05 课（LLaVA），第 5 阶段（NLP）
**所需时间：** 约180分钟

## 学习目标

- 解释文档 AI 的三个时代：OCR 流水线、无 OCR、VLM 原生。
- 描述 LayoutLMv3 的三个输入流：文本、布局（边界框）、图像 patch，以及统一掩码。
- 比较 Donut（无 OCR，图像 → 标记）、Nougat（科学论文 → LaTeX）、DocLLM（布局感知生成式）、PaliGemma 2（VLM 原生）。
- 为新任务（发票、科学论文、手写表格、中文小票）选择文档模型。

## 问题所在

"理解这个 PDF"看似简单实则很难。信息分布在：

- 文本内容（90% 的信号）。
- 布局（标题、脚注、侧边栏、双栏格式）。
- 表格（行、列、合并单元格）。
- 图表和示意图。
- 手写批注。
- 字体和排版（标题 vs 正文）。

原始 OCR 只提取文本，丢失其余一切。一个关注发票的系统需要知道"总计：$1,245"来自右下角，而不是脚注。

## 概念说明

### 第一代——OCR 流水线（2021 年之前）

经典方案栈：

1. PDF → 每页一张图像。
2. Tesseract（或商业 OCR）提取文本及每词的边界框。
3. 布局分析器识别块（标题、表格、段落）。
4. 表格结构识别器解析表格。
5. 领域规则 + 正则表达式提取字段。

对干净的打印文本有效。在手写、倾斜扫描、复杂表格、非英文脚本上失败。每种失败模式都需要自定义异常路径。

### TrOCR（2021）

TrOCR（Li 等人，arXiv:2109.10282）用在合成 + 真实文本图像上训练的 Transformer 编码器-解码器替代了 Tesseract 的经典 CNN-CTC。在手写和多语言文本上完胜。仍然是流水线（检测器 → TrOCR → 布局），但 OCR 步骤大幅提升。

### 第二代——无 OCR（2022-2023）

首批无 OCR 模型说：完全跳过检测，直接将图像像素映射到结构化输出。

Donut（Kim 等人，arXiv:2111.15664）：
- 编码器-解码器 Transformer，编码器为 Swin-B。
- 输出为表格理解的 JSON、摘要的 markdown 或任何任务特定 schema。
- 无 OCR、无布局、无检测。

Nougat（Blecher 等人，arXiv:2308.13418）：
- 专门在科学论文上训练。
- 输出为 LaTeX / markdown。
- 处理公式、多栏布局、图表。
- 每个 arXiv 解析器都调用的模型。

这些是专家，不是通才。Donut 处理科学论文失败；Nougat 处理发票失败。

### LayoutLMv3（2022）

另一条路线。LayoutLMv3（Huang 等人，arXiv:2204.08387）保留 OCR 但增加了布局理解：

- 三个输入流：OCR 文本 token、每 token 的 2D 边界框、图像 patch。
- 跨所有三种模态的掩码训练目标（掩码文本、掩码 patch、掩码布局）。
- 下游任务：分类、实体提取、表格 QA。

LayoutLMv3 是基于 OCR 的文档理解的巅峰。在表格和发票上表现出色。需要上游 OCR。在标准化文档基准上达到预 VLM 时代的最佳精度。

### DocLLM（2023）

DocLLM（Wang 等人，arXiv:2401.00908）是 LayoutLM 的生成式兄弟。以布局 token 为条件生成自由形式的答案。更适合文档上的 QA；仍然依赖 OCR 输入。

### 第三代——VLM 原生（2024+）

2024 年的 VLM 足够好，可以完全替代流水线。以高分辨率将完整页面图像输入 VLM，提出问题，获得答案。

- LLaVA-NeXT 336-tile AnyRes 适用于小型文档。
- Qwen2.5-VL 动态分辨率原生处理 2048+ 像素。
- Claude Opus 4.7 支持 2576px 文档。
- PaliGemma 2（2025 年 4 月）专门为文档 + 手写训练。

VLM 原生与 OCR 流水线之间的差距迅速缩小。到 2026 年，VLM 原生在以下场景胜出：

- 场景文本（手写 + 打印，混合脚本）。
- 带合并单元格的复杂表格。
- 嵌入文本中的数学公式。
- 带文本注释的图表。

OCR 流水线在以下场景仍然胜出：

- 大规模纯扫描工作负载，每页延迟很重要。
- 流水线可靠性（确定性失败 vs VLM 幻觉）。
- 需要可审计 OCR 输出的受监管环境。

### Claude 4.7 / GPT-5 前沿

在 2576 像素原生输入下，前沿 VLM 以接近人类精度进行文档理解。2026 年初的基准测试数据：

- DocVQA：Claude 4.7 约 95.1，PaliGemma 2 约 88.4，Nougat 约 77.3，流水线 LayoutLMv3 约 83。
- ChartQA：Claude 4.7 约 92.2，GPT-4V 约 78。
- VisualMRC：Claude 4.7 约 94。

闭源模型的差距主要在于分辨率和基础 LLM 规模。7B 的开放模型落后几个百分点但在追赶。

### 数学公式和 LaTeX 输出

科学论文需要精确的 LaTeX 公式输出。Nougat 专门为此训练。使用 LaTeX 目标训练的 VLM（Qwen2.5-VL-Math、Nougat 衍生版本）能产生可用的 LaTeX。没有显式 LaTeX 训练的 VLM 产生可读但不精确的转录。

2026 年的科学论文管道：先在 PDF 上跑 Nougat，然后在困难页面上用 VLM。

### 手写

仍然是最困难的子任务。混合打印 + 手写（医生笔记、填写的表格）是 OCR 流水线在成本上仍然胜过 VLM 的场景。纯手写的 VLM 正在改善（Claude 4.7、PaliGemma 2）。

### 2026 年方案

对于新的文档 AI 项目：

- 大规模纯打印发票：LayoutLMv3 + 规则，成本高效。
- 混合文档（科学 + 手写 + 表格）：VLM 原生（PaliGemma 2 或 Qwen2.5-VL）。
- 完整 arXiv 收录：Nougat 处理数学，VLM 处理图表。
- 受监管场景：OCR 流水线 + VLM 验证器交叉检查。

## 开始构建

`code/main.py`：

- 玩具布局感知 tokenizer：给定（文本，边界框）对，产生 LayoutLMv3 风格的输入。
- Donut 风格的任务 schema 生成器：表格的 JSON 模板。
- 跨 OCR 流水线、Donut、Nougat 和 VLM 原生的每页 token 预算比较。

## 交付成果

本课生成 `outputs/skill-document-ai-stack-picker.md`。给定一个文档 AI 项目（领域、规模、质量、受监管），在 OCR 流水线、无 OCR 专家模型和 VLM 原生之间做出选择。

## 练习

1. 你的项目是每天处理 1000 万张发票。哪种方案栈在不损失精度的情况下最小化每页成本？

2. 为什么 LayoutLMv3 在表格 QA 上优于纯 CLIP VLM，但在场景文本上不如？边界框流放弃了什么？

3. Nougat 生成 LaTeX。提出一个 VLM 原生输出在 LaTeX 保真度上胜过 Nougat 的测试用例，以及一个 Nougat 胜出的用例。

4. 阅读 PaliGemma 2 论文（Google，2024）。相比 PaliGemma 1，关键的训练数据添加是什么，它提升了文档精度？

5. 设计一个受监管安全的混合方案：OCR 流水线为主，VLM 为辅交叉检查。如何解决分歧？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| OCR 流水线 | "Tesseract 风格" | 分阶段栈：检测 → OCR → 布局 → 规则；确定性，脆弱 |
| 无 OCR | "Donut 风格" | 图像到输出的 Transformer，跳过显式 OCR；单一模型 |
| 布局感知 | "LayoutLM" | 输入包含每 token 的边界框坐标；跨模态的统一掩码 |
| VLM 原生 | "前沿 VLM" | 直接以高分辨率将页面图像输入 Claude/GPT/Qwen VLM；无流水线 |
| DocVQA | "文档基准" | 文档 VQA 标准；被引用最多的分数 |
| 标记输出 | "LaTeX / MD" | 结构化输出格式，而非自由文本；支持下游自动化 |

## 延伸阅读

- [Li 等人 — TrOCR (arXiv:2109.10282)](https://arxiv.org/abs/2109.10282)
- [Blecher 等人 — Nougat (arXiv:2308.13418)](https://arxiv.org/abs/2308.13418)
- [Huang 等人 — LayoutLMv3 (arXiv:2204.08387)](https://arxiv.org/abs/2204.08387)
- [Kim 等人 — Donut (arXiv:2111.15664)](https://arxiv.org/abs/2111.15664)
- [Wang 等人 — DocLLM (arXiv:2401.00908)](https://arxiv.org/abs/2401.00908)
