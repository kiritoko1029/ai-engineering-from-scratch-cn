# 嵌入模型——2026深度解析

> Word2Vec 为每个单词生成一个向量。而现代嵌入模型则能为整段文本生成跨语言的向量，并提供稀疏、密集及多向量等多种表示形式，其大小可根据索引需求进行调整。若选择不当，RAG 系统将检索到错误的内容。

**类型：** 学习
**编程语言：** Python
**先修课程：** 第 5 阶段 · 03（Word2Vec）、第 5 阶段 · 14（信息检索）
**时长：** 约 60 分钟

## 问题所在

您的 RAG 系统有 40% 的概率检索到错误的段落。问题的根源很少在于向量数据库或提示词，而在于嵌入模型。

在 2026 年选择嵌入模型时，需要从五个维度进行考量：

1. **密集型、稀疏型与多向量型。** 每个段落一个向量，每个标记一个向量，或是使用稀疏加权词袋模型。
2. **语言覆盖范围。** 在仅处理英语任务时，单语英语模型依然更具优势；而在语料混合的情况下，多语言模型表现更佳。
3. **上下文长度。** 512 个标记、8,192 个标记或 32,768 个标记——而实际有效容量通常仅为标称最大值的 60-70%。
4. **维度预算。** 全精度下，每个向量需要 3,072 个浮点数，即每向量占用 12 KB 存储空间。若需存储 1 亿个向量，每月的存储成本约为 1,300 美元；而“俄罗斯套娃式截断”技术可将该成本降低 4 倍。
5. **开源型与托管型。** 开源权重模型让您能够掌控整个技术栈和数据；而托管服务则意味着您需放弃控制权以换取始终最新的功能。

## 概念概述

![密集嵌入、稀疏嵌入与多向量嵌入](../assets/embedding-modes.svg)

**密集嵌入。** 每段文本对应一个向量（通常维度为 384-3,072）。通过余弦相似度根据语义相近程度对文本进行排序。常用模型包括 OpenAI 的 `text-embedding-3-large`、BGE-M3 的密集模式以及 Voyage-3。为默认选择。

**稀疏嵌入。** 采用 SPLADE 风格。Transformer 模型会为每个词汇表中的词预测一个权重，随后将大部分权重设为零。最终得到一个大小为 |vocab| 的稀疏向量。该方式能够实现类似 BM25 的词汇匹配功能，但使用了学习得到的词权重，对关键词占比较高的查询表现优异。

**多向量嵌入（后期交互型）。** 代表模型有 ColBERTv2 和 Jina-ColBERT。每个词对应一个向量。评分采用 MaxSim 方法：对于查询中的每个词，找到与之最相似的文档词，然后将所有得分相加。虽然存储和评分的成本较高，但在处理长文本查询及领域专用语料库时表现更佳。

**BGE-M3：同时整合三种嵌入方式。** 单一模型可同时输出密集嵌入、稀疏嵌入以及多向量嵌入表示。这三种表示可以独立被查询，其得分通过加权求和的方式融合在一起。当需要从单个检查点中获得最大灵活性时，它是 2026 年的默认选择。

**马特罗什卡表示学习。** 这种训练方式旨在让向量的前 N 维构成一个独立的、有用的嵌入向量。例如，将一个 1,536 维的向量截断为 256 维，虽然精度会下降约 1%，但存储空间可节省 6 倍。该技术被 OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2 以及 Nomic v1.5+ 等模型支持。

### MTEB排行榜仅能反映部分情况。

Massive Text Embedding Benchmark——在2022年首次发布时包含8种任务类型下的56项测试任务，MTEB v2版本则扩展至100多项任务。2026年初，Gemini Embedding 2在检索性能指标上表现最佳，得分达67.71 MTEB-R；Cohere embed-v4在通用任务领域的得分则为65.2 MTEB；BGE-M3则在开源多语言任务中领先，得分为63.0。虽然排行榜具有参考价值，但并不能作为唯一标准——务必针对自身应用领域进行专项测试。

### 三层架构模式

| 使用场景 | 模式 |
|----------|---------|
| 快速初筛 | 高密度双编码器（BGE-M3、text-3-small） |
| 提升召回率 | 稀疏型模型（SPLADE、BGE-M3 sparse）+ RRF融合机制 |
| 保证前50个结果的精确度 | 多向量模型（ColBERTv2）或交叉编码器重排器 |

大多数生产环境会同时使用这三种方案。

## 构建它

### 步骤 1：基准方案——使用 Sentence-BERT 生成密集嵌入向量

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 会将点积转换为余弦相似度。请始终将其设置为该值。

### 步骤 2：马特罗什卡截断法

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

在截断后进行重新归一化处理。Nomic v1.5、OpenAI text-3以及Voyage-4经过专门训练，因此在前几层处理中可实现无损失转换。而非套娃式模型（即原始的Sentence-BERT）在遭遇截断时性能会急剧下降。

### 步骤 3：BGE-M3 的多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三个索引，一次推理调用。分数融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在您的领域数据上调整权重。

### 步骤 4：对自定义任务进行 MTEB 评估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*具有代表性的*子数据集上运行候选模型。切勿仅凭排行榜排名来判断——具体应用领域至关重要。

### 步骤 5：从零开始手动实现余弦函数

参见 `code/main.py`。平均哈希技巧嵌入（仅使用标准库实现）。虽然无法与Transformer嵌入相媲美，但展示了其整体流程：分词 → 向量化 → 归一化 → 点积运算。

## 常见陷阱

- **查询与文档使用相同模型。** 某些模型（如 Voyage、Jina-ColBERT）采用非对称编码方式——查询和文档会经过不同的处理路径。务必查阅模型卡片中的相关说明。
- **缺少前缀。** `bge-*` 系列模型要求在查询语句前添加 `"Represent this sentence for searching relevant passages: "` 这一前缀。若忘记添加，召回率将出现 3-5 个百分点的差距。
- **Matryoshka 方法过度截断。** 通常将输入长度从 1,536 缩减到 256 是安全的，但直接缩减到 64 则不可行。需在自己的评估集上进行验证。
- **上下文被截断。** 大多数模型会在达到最大长度限制时自动截断输入内容。对于较长的文档，需要将其分块处理（参见第 23 课）。
- **忽略延迟尾部分。** MTEB 分数会掩盖 p99 延迟指标。一个参数量为 600M 的模型虽然得分可能仅比参数量为 335M 的模型高 2 分，但其每次查询的耗时可能是后者的 3 倍。

## 使用它

2026年推荐技术栈：

| 场景 | 推荐模型 |
|-----------|----------|
| 仅支持英文、需快速响应且依赖API | `text-embedding-3-large` 或 `voyage-3-large` |
| 开源权重、支持英文 | `BAAI/bge-large-en-v1.5` |
| 开源权重、支持多语言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 长上下文需求（32k字以上） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 仅限CPU部署 | Nomic Embed v2（1.37亿参数，MoE架构） |
| 存储空间受限 | Matryoshka-truncated模型结合int8量化技术 |
| 包含大量关键词的查询 | 结合SPLADE稀疏向量及RRF融合技术与密集向量处理 |

2026年应用策略：首先选用BGE-M3或text-3-large，通过MTEB指标在目标领域进行评估；若某个领域专用模型能带来超过3分的性能提升，则替换为该专用模型。

## 发布它

保存为 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 练习题

1. **简单级。** 使用 `bge-small-en-v1.5` 在全维度（384）下对100个句子进行编码，然后在Matryoshka 128维度下再次编码。针对10次查询，测量MRR的下降幅度。
2. **中等级。** 在来自您所在领域的500篇文本中，比较BGE-M3的密集型、稀疏型以及Colbert模型在召回率@10指标上的表现。哪种模型的表现最佳？RRF融合方法是否优于最优的单一模型？
3. **高级别。** 对您所在领域中排名前二的三个候选模型，在三项任务上运行MTEB评估。需报告MTEB得分、100次查询批处理的p99延迟时间，以及每百万次查询的成本。从中选出帕累托最优的模型。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 密集嵌入 | 向量 | 每条文本对应一个固定大小的向量，通过余弦相似度进行排序。 |
| 稀疏嵌入 | 学习型 BM25 | 每个词汇标记对应一个权重；大部分为零值；采用端到端训练方式。 |
| 多向量嵌入 | ColBERT 风格 | 每个标记对应一个向量；使用 MaxSim 进行评分；索引规模越大，召回率越高。 |
| 级联嵌套嵌入 | 俄罗斯套娃技巧 | 前 N 维本身即构成一个有效的较小尺寸嵌入。 |
| MTEB | 测评标准 | 大规模文本嵌入评测基准——初始版本包含 56 项任务，v2 版本超过 100 项。 |
| BEIR | 检索评测标准 | 包含 18 项零样本检索任务；常被用作衡量跨领域鲁棒性的指标。 |
| 非对称编码 | 查询路径 ≠ 文档路径 | 模型对查询和文档使用不同的投影方式。 |

## 延伸阅读

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — 双向编码器相关论文。  
- [Muennighoff 等人 (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排名榜相关论文。  
- [Chen 等人 (2024). BGE-M3: 多语言、多功能、多粒度模型](https://arxiv.org/abs/2402.03216) — 统一的三模态模型。  
- [Kusupati 等人 (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 维度阶梯训练目标。  
- [Santhanam 等人 (2022). ColBERTv2: 通过轻量级后期交互实现高效检索](https://arxiv.org/abs/2112.01488) — 生产环境中的后期交互应用。  
- [Hugging Face 上的 MTEB 排名榜](https://huggingface.co/spaces/mteb/leaderboard) — 实时排名信息。
