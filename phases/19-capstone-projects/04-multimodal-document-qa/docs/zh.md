# 毕业项目 04 — 多模态文档问答（视觉优先 PDF、表格、图表）

> 2026 年的文档问答前沿已从"先 OCR 再文本"转向视觉优先的后期交互。ColPali、ColQwen2.5 和 ColQwen3-omni 将每个 PDF 页面作为图像处理，使用多向量后期交互进行嵌入，让查询直接关注图像块。在财务 10-K、科学论文和手写笔记上，这种模式大幅超越了先 OCR 再文本的方法。在 1 万页数据上端到端构建该流水线，并发布与先 OCR 再文本方案的对比结果。

**类型：** 毕业项目
**语言：** Python（流水线）、TypeScript（查看器 UI）
**前置要求：** 阶段 4（计算机视觉）、阶段 5（NLP）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 12（多模态）、阶段 17（基础设施）
**涉及阶段：** P4 · P5 · P7 · P11 · P12 · P17
**所需时间：** 30 小时

## 问题所在

企业积累了大量 PDF，OCR 流水线却处理得很糟糕：扫描的 10-K 包含旋转的表格、满是公式的科学论文、只有作为图像才有意义的图表、手写批注。将这些内容当作纯文本处理意味着丢失一半的信号。2026 年的解决方案是对原始页面图像进行后期交互多向量检索。ColPali（Illuin Tech）开创了这一方法；ColQwen2.5-v0.2 和 ColQwen3-omni 进一步提升了准确率。在 ViDoRe v3 上，视觉优先的检索得分以显著幅度超过先 OCR 再文本的方法——在图表、表格和手写内容上差距更大。

代价是存储和延迟。一个 ColQwen 嵌入每页约 2048 个图像块向量，而非单个 1024 维向量。原始存储空间大幅膨胀。DocPruner（2026）在可忽略的准确率损失下实现 50% 的压缩。你将索引 1 万页，衡量 ViDoRe v3 的 nDCG@5，在 2 秒内提供答案，并与先 OCR 再文本的基线进行直接对比。

## 概念说明

后期交互意味着每个查询 Token 与每个图像块 Token 独立打分，然后对每个查询 Token 的最高分求和。你获得了细粒度匹配能力，而无需单一池化向量。多向量索引（Vespa、Qdrant multi-vector 或 AstraDB）存储每个图像块的嵌入，并在检索时运行 MaxSim。

回答器是一个视觉语言模型，接收查询和 top-k 检索到的页面作为图像，写出带有证据区域（边界框或页面引用）的答案。Qwen3-VL-30B、Gemini 2.5 Pro 和 InternVL3 是 2026 年的前沿选择。对于公式和科学记法，OCR 后备方案（Nougat、dots.ocr）作为可选文本通道被接入。

评估是二维矩阵。一个维度：内容类型（纯文字段落、密集表格、柱状/折线图、手写笔记、公式）。另一个维度：检索方式（视觉优先后期交互 vs 先 OCR 再文本 vs 混合）。每个单元格有 nDCG@5 和答案准确率。报告就是交付成果。

## 架构

```
PDFs -> page renderer (PyMuPDF, 180 DPI)
           |
           v
  ColQwen2.5-v0.2 embed (multi-vector per page, ~2048 patches)
           |
           +------> DocPruner 50% compression
           |
           v
   multi-vector index (Vespa or Qdrant multi-vector)
           |
query ----+----> retrieve top-k pages (MaxSim)
           |
           v
  VLM answerer: Qwen3-VL-30B | Gemini 2.5 Pro | InternVL3
    inputs: query + top-k page images + optional OCR text
           |
           v
  answer with cited page numbers + evidence regions
           |
           v
  Streamlit / Next.js viewer: highlighted boxes on source page
```

## 技术栈

- 页面渲染：PyMuPDF（fitz），180 DPI，纵向归一化
- 后期交互模型：ColQwen2.5-v0.2 或 ColQwen3-omni（Hugging Face 上的 vidore 团队）
- 索引：Vespa 多向量字段，或 Qdrant multi-vector，或 AstraDB + MaxSim
- 压缩：DocPruner 2026 策略（保留高方差图像块，50% 压缩率，准确率损失 < 0.5%）
- OCR 后备（公式/密集表格）：dots.ocr 或 Nougat
- VLM 回答器：自托管 Qwen3-VL-30B 或托管 Gemini 2.5 Pro；InternVL3 作为备选
- 评估：ViDoRe v3 基准，M3DocVQA 用于多页推理
- 查看器 UI：Next.js 15，带 Canvas 叠加的证据区域

## 开始构建

1. **摄入。** 遍历包含 10-K、科学论文和扫描文档的 1 万页 PDF 语料库。将每页渲染为 1536x2048 PNG。持久化 `{doc_id, page_num, image_path}`。

2. **嵌入。** 对每个页面图像运行 ColQwen2.5-v0.2。输出形状约 2048 个维度为 128 的图像块嵌入。应用 DocPruner 保留信号最强的一半。写入 Vespa 多向量字段或 Qdrant multi-vector。

3. **查询。** 对每个传入的查询，使用查询塔进行嵌入（Token 级嵌入）。对索引运行 MaxSim：对每个查询 Token，取页面图像块嵌入上的最大点积，求和。返回 top-k 页面。

4. **综合。** 使用查询和 top-5 页面图像调用 Qwen3-VL-30B。提示词："仅使用提供的页面回答。对每个声明引用 (doc_id, page) 并指明区域（图表、表格、段落）。"

5. **证据区域。** 后处理答案以提取引用区域。如果 VLM 输出边界框（Qwen3-VL 支持），在查看器中将它们渲染为叠加层。

6. **OCR 后备。** 对于被识别为公式密集的页面（基于图像方差的启发式方法），运行 Nougat 或 dots.ocr，将 OCR 文本作为与图像并行的额外通道传递。

7. **评估。** 运行 ViDoRe v3（检索 nDCG@5）和 M3DocVQA（多页问答准确率）。同时在同一语料库上使用同一综合器运行先 OCR 再文本的流水线。生成内容类型 x 方式的矩阵。

8. **UI。** 先用 Streamlit 原型；再用 Next.js 15 构建生产级查看器，支持逐页证据区域叠加。

## 使用示例

```
$ doc-qa ask "what was the 2024 operating margin change for segment EMEA?"
[retrieve]   top-5 pages in 320ms (ColQwen2.5, MaxSim, Vespa)
[synth]      qwen3-vl-30b, 1.4s, cited (form-10k-2024, p. 88) + (..., p. 92)
answer:
  EMEA operating margin moved from 18.2% to 16.8%, a 140bp decline.
  cited: 10-K-2024.pdf p.88 (Table 4, Segment Operating Margin)
         10-K-2024.pdf p.92 (MD&A, Operating Performance)
[viewer]     open with highlighted bounding boxes overlaid on p.88 Table 4
```

## 交付成果

`outputs/skill-doc-qa.md` 描述了交付成果：一个视觉优先的多模态文档问答系统，针对特定语料库调优，并在 ViDoRe v3 上与先 OCR 再文本的基线进行评估。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | ViDoRe v3 / M3DocVQA 准确率 | 与 OCR 文本基线和已发布排行榜的基准数字对比 |
| 20 | 证据区域锚定 | 引用区域实际包含答案跨度的比例 |
| 20 | 存储与延迟工程 | DocPruner 压缩率、索引 p95、答案 p95 |
| 20 | 多页推理 | 在人工标注的 100 个问题多页集上的准确率 |
| 15 | 源文件检查体验 | 查看器清晰度、叠加保真度、并排对比工具 |
| **100** | | |

## 练习

1. 在同一语料库上衡量 ColQwen2.5-v0.2 与 ColQwen3-omni。哪个模型在哪些页面上正确而另一个遗漏？在索引中添加"内容类别"标签以按类型路由。

2. 激进地压缩嵌入（75%、90%）。找到压缩临界点：ViDoRe nDCG@5 降到 OCR 基线以下的位置。

3. 构建混合方案：并行运行先 OCR 再文本和 ColQwen，用 RRF 融合，用交叉编码器重排序。混合方案是否优于单独方案？在哪些场景帮助最大？

4. 将 Qwen3-VL-30B 替换为更小的 VLM（Qwen2.5-VL-7B）。衡量准确率与成本的曲线。

5. 添加手写笔记支持。渲染手写语料库，用 ColQwen 嵌入，衡量检索效果。与手写 OCR 流水线进行对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 后期交互 | "ColPali 风格检索" | 查询 Token 与页面图像块独立打分；MaxSim 聚合 |
| 多向量 | "逐图像块嵌入" | 每个文档有多个向量，而非一个池化向量 |
| MaxSim | "后期交互打分" | 对每个查询 Token，取文档向量上的最大相似度；求和 |
| DocPruner | "图像块压缩" | 2026 年的压缩方案，保留 50% 图像块，准确率损失可忽略 |
| ViDoRe v3 | "文档检索基准" | 2026 年衡量视觉文档检索的标准 |
| 证据区域 | "引用的边界框" | 源页面上定位答案跨度的边界框 |
| OCR 后备 | "公式通道" | 用于公式或表格密集页面的、与视觉并行的文本流水线 |

## 延伸阅读

- [ColPali（Illuin Tech）仓库](https://github.com/illuin-tech/colpali) — 后期交互文档检索参考
- [ColPali 论文（arXiv:2407.01449）](https://arxiv.org/abs/2407.01449) — 基础方法论文
- [ColQwen 系列（Hugging Face）](https://huggingface.co/vidore) — 生产就绪的检查点
- [M3DocRAG（Adobe）](https://arxiv.org/abs/2411.04952) — 多页多模态 RAG 基线
- [Vespa 多向量教程](https://docs.vespa.ai/en/colpali.html) — 参考服务技术栈
- [Qdrant 多向量支持](https://qdrant.tech/documentation/concepts/vectors/#multivectors) — 备选索引
- [AstraDB 多向量](https://docs.datastax.com/en/astra-db-serverless/databases/vector-search.html) — 备选托管索引
- [Nougat OCR](https://github.com/facebookresearch/nougat) — 支持公式的 OCR 后备方案
