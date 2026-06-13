# 毕业项目 02 — 基于代码库的 RAG（跨仓库语义搜索）

> 2026 年，每一个严肃的工程团队都在运行内部代码搜索，它能理解语义，而不仅仅是匹配字符串。Sourcegraph Amp、Cursor 的代码库问答、Augment 的企业级知识图谱、Aider 的仓库地图、Pinterest 的内部 MCP——形态如出一辙。摄入多个仓库、用 tree-sitter 解析、对函数和类级别的代码块进行嵌入、混合搜索、重排序、带引用地回答。本毕业项目要求你构建一个能处理 10 个仓库共 200 万行代码，并能在每次 git push 时增量重建索引的系统。

**类型：** 毕业项目
**语言：** Python（数据摄入）、TypeScript（API + UI）
**前置要求：** 阶段 5（NLP 基础）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 17（基础设施）
**涉及阶段：** P5 · P7 · P11 · P13 · P17
**所需时间：** 30 小时

## 问题所在

到 2026 年，每个前沿编码智能体都配备了代码库检索层，因为仅靠上下文窗口无法解决跨仓库的问题。Claude 的 100 万 Token 上下文有所帮助，但并不能消除对排序检索的需求。对原始代码块进行朴素的余弦相似度搜索，会在生成代码、monorepo 重复代码以及罕见导入符号的长尾场景中污染结果。生产环境的解决方案是基于 AST 感知的代码块进行混合搜索（Dense + BM25），配合重排序器，并以符号引用图谱为支撑。

你需要通过索引一个真实的仓库集群来学习这些——而非一个教程仓库——并衡量 MRR@10、引用可信度和增量更新的新鲜度。失败模式都来自基础设施层面：一个 10 万文件的 monorepo、一次改动了半数文件的推送、一个需要跨越四个仓库才能正确回答的查询。

## 概念说明

AST 感知的摄入流水线使用 tree-sitter 解析每个文件，提取函数和类节点，并在节点边界而非固定 Token 窗口处进行分块。每个代码块有三种表示：稠密嵌入（Voyage-code-3 或 nomic-embed-code）、稀疏 BM25 词项，以及一段简短的自然语言摘要。摘要提供了第三种可检索的模态——当用户问"X 是如何被授权的"时，摘要中提到了"authz"，即使代码中只有 `check_permission`。

检索是混合式的。一次查询同时触发稠密和 BM25 搜索，合并 top-k 结果后交给交叉编码器重排序器（Cohere rerank-3 或 bge-reranker-v2-gemma-2b）。重排序后的列表传递给长上下文综合器（使用提示缓存的 Claude Sonnet 4.7 或自托管的 Llama 3.3 70B），并要求它对每个声明引用文件和行范围。没有引用的答案会被后置过滤器拒绝。

增量更新是基础设施问题。Git push 触发一次 diff：哪些文件变了，哪些符号变了。只有受影响的代码块会重新嵌入。受影响的跨文件符号边（导入、方法调用）会被重新计算。索引保持一致，无需每次提交都重新处理 200 万行代码。

## 架构

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技术栈

- 解析：tree-sitter，支持 17 种语言语法（Python、TS、Rust、Go、Java、C++ 等）
- 稠密嵌入：Voyage-code-3（托管）或 nomic-embed-code-v1.5（自托管），bge-code-v1 作为备选
- 稀疏索引：Tantivy（Rust），BM25F，对符号名称和正文进行字段加权
- 向量数据库：Qdrant 1.12，支持混合搜索；或 pgvector + pgvectorscale（适用于 5000 万向量以下的团队）
- 代码块摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，启用提示缓存
- 重排序器：Cohere rerank-3 或自托管的 bge-reranker-v2-gemma-2b
- 编排：LlamaIndex Workflows 用于摄入，LangGraph 用于查询智能体
- 综合器：Claude Sonnet 4.7（100 万上下文），启用提示缓存
- 符号图谱：Neo4j（托管）或 kuzu（嵌入式），用于导入和调用边
- 可观测性：Langfuse，每个检索+综合步骤都有 Span

## 开始构建

1. **摄入遍历器。** 在每次 push hook 时遍历 Git 历史。收集变更文件。对每个文件，使用 tree-sitter 解析，提取函数和类节点及其完整源码范围。输出代码块记录 `{repo, path, start_line, end_line, symbol, body}`。

2. **代码块摘要器。** 将代码块批量发送给 Haiku 4.5，在系统前言上启用提示缓存。提示词："用一句话概括这个函数，说明其公开接口和副作用。"将摘要与代码块一起存储。

3. **嵌入池。** 两个并行队列：稠密（Voyage-code-3，批量 128）和摘要（同一模型，但处理摘要字符串）。将向量写入 Qdrant，负载为 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 字段加权的 Tantivy 索引：符号名称权重 4，符号正文权重 1，摘要权重 2。支持"查找名为 X 的函数"和"查找执行 X 功能的函数"两类查询。

5. **符号图谱。** 对每个代码块记录边：导入（此文件使用了仓库 Z 中的符号 Y）、调用（此函数调用了类 C 的方法 M）、继承。存储在 kuzu 中。在查询时用于跨仓库边界扩展检索。

6. **查询智能体。** LangGraph 包含三个节点。`retrieve` 并行触发稠密和 BM25 搜索，按 (repo, path, symbol) 去重。`rerank` 在 top-50 上运行交叉编码器，保留 top-10。`synth` 将重排序后的代码块作为上下文调用 Claude Sonnet 4.7，缓存系统提示，要求 file:line 引用。

7. **引用强制。** 解析模型输出；任何没有 `(repo/path:start-end)` 锚点的声明都会被标记为需要重新询问或丢弃。仅返回有引用的答案给用户。

8. **增量重建索引。** 每次 webhook 触发时，计算符号级别的 diff。仅重新嵌入文本变更的代码块。重新计算导入变更的代码块的符号边。衡量标准：一个 50 文件的推送在 200 万行代码的仓库集群上应在 60 秒内完成重建索引。

9. **评估。** 为 100 个跨仓库问题标注标准的 file:line 答案。衡量 MRR@10、nDCG@10、引用可信度（具有可验证锚点的声明比例）以及 p50/p99 延迟。

## 使用示例

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付成果

交付的技能文件 `outputs/skill-codebase-rag.md`。给定一个仓库语料库，它会搭建摄入流水线、混合索引和查询智能体，并为任何跨仓库问题返回带引用的答案。评分标准：

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 检索质量 | 在 100 个问题的保留集上衡量 MRR@10 和 nDCG@10 |
| 20 | 引用可信度 | 具有可验证 file:line 锚点的答案声明比例 |
| 20 | 延迟与规模 | 在已索引语料库规模下，10k QPS 的 p95 查询延迟 |
| 20 | 增量索引正确性 | 从 git push 到可搜索的时间（50 文件提交场景） |
| 15 | 用户体验与答案格式 | 引用可点击性、代码片段预览、追问入口 |
| **100** | | |

## 练习

1. 将 Voyage-code-3 替换为自托管的 nomic-embed-code。衡量 MRR@10 的差异。报告启用重排序后差距是否缩小。

2. 向语料库注入 20% 的生成代码（LLM 生成的模板代码）并重新评估。观察检索污染现象。在负载中添加"generated"标志并降低这些命中的权重。

3. 在你的语料库规模下，对 Qdrant 混合搜索与 pgvector + pgvectorscale 进行基准测试。报告批量大小为 1 时的 p99 延迟。

4. 添加基于采样的漂移检测：每周重新运行 100 个问题的评估。当 MRR@10 下降超过 5% 时告警。

5. 扩展到跨语言符号解析：一个 Python 函数通过 gRPC 调用 Go 服务。使用符号图谱将它们关联起来。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| AST 感知分块 | "函数级拆分" | 在 tree-sitter 节点边界而非固定 Token 窗口处切割代码 |
| 混合搜索 | "稠密 + 稀疏" | 并行运行 BM25 和向量搜索，合并 top-k，重排序 |
| 交叉编码器重排序 | "二阶段排序" | 对每个 (query, candidate) 对一起打分的模型，比余弦相似度更准确 |
| 提示缓存 | "缓存的系统提示" | 2026 年 Claude / OpenAI 功能，对重复的前缀 Token 最多打 9 折 |
| 符号图谱 | "代码图谱" | 跨文件和仓库的导入、调用、继承边 |
| 引用可信度 | "基于事实的回答率" | 用户可以通过点击锚点并阅读引用范围来验证的声明比例 |
| 增量重建索引 | "推送至可搜索时间" | 从 git push 到变更符号可查询的墙钟时间 |

## 延伸阅读

- [Sourcegraph Amp](https://ampcode.com) — 生产级跨仓库代码智能
- [Sourcegraph Cody RAG 架构](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 本毕业项目的参考深入解读
- [Aider 仓库地图](https://aider.chat/docs/repomap.html) — tree-sitter 排序的仓库视图
- [Augment Code 企业知识图谱](https://www.augmentcode.com) — 商用符号图谱 RAG
- [Qdrant 混合搜索文档](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 参考实现
- [Voyage AI 代码嵌入](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 详情
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — 交叉编码器参考
- [Pinterest MCP 内部搜索](https://medium.com/pinterest-engineering) — 内部平台参考
