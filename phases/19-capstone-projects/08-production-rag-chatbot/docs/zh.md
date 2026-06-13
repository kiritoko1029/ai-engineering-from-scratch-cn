# 毕业项目 08 — 受监管垂直领域的生产级 RAG 聊天机器人

> Harvey、Glean、Mendable 和 LlamaCloud 在 2026 年都运行着相同的生产形态。使用 docling 或 Unstructured 和 ColPali 处理视觉内容进行摄入。混合搜索。使用 bge-reranker-v2-gemma 重排序。使用 Claude Sonnet 4.7 进行综合，提示缓存命中率达 60-80%。使用 Llama Guard 4 和 NeMo Guardrails 进行安全防护。使用 Langfuse 和 Phoenix 进行监控。使用 RAGAS 在 200 个问题的黄金集上进行评分。在一个受监管领域（法律、临床、保险）构建一个这样的系统，毕业项目的核心是通过黄金集测试、红队测试和漂移仪表盘。

**类型：** 毕业项目
**语言：** Python（流水线 + API）、TypeScript（聊天 UI）
**前置要求：** 阶段 5（NLP）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 12（多模态）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P5 · P7 · P11 · P12 · P17 · P18
**所需时间：** 30 小时

## 问题所在

受监管领域的 RAG（法律合同、临床试验方案、保险政策）是 2026 年部署最多的生产形态，因为 ROI 明确且风险具体。Harvey（Allen & Overy）为法律领域构建了它。Mendable 提供开发者文档版本。Glean 覆盖企业搜索。模式是：高保真摄入、混合检索加重排序、带引用强制和提示缓存的综合、多层安全防护，以及持续的漂移监控。

难点不在于模型。而在于辖区感知的合规性（HIPAA、GDPR、SOC2）、引用级别的可审计性、成本控制（提示缓存在命中率高时可获得 60-90% 的折扣）、通过 RAGAS 忠实度进行幻觉检测，以及当源文档更新而索引未跟上时的漂移检测。本毕业项目要求你在 200 个问题的黄金集上部署所有这些功能，同时配备红队套件。

## 概念说明

流水线有两面。**摄入**：docling 或 Unstructured 解析结构化文档；ColPali 处理视觉丰富的文档；代码块获得摘要、标签和基于角色的访问标签。向量存入 pgvector + pgvectorscale（5000 万向量以下）或 Qdrant Cloud；稀疏 BM25 并行运行。**对话**：LangGraph 处理记忆和多轮对话；每个查询运行混合检索，使用 bge-reranker-v2-gemma-2b 重排序，使用 Claude Sonnet 4.7（启用提示缓存）进行综合，输出通过 Llama Guard 4 和 NeMo Guardrails，发出带引用锚定的响应。

评估技术栈有四层。**黄金集**（200 个标注的带引用的问答对）用于正确性。**红队**（越狱、PII 提取尝试、领域外问题）用于安全性。**RAGAS** 用于每轮自动评估忠实度/答案相关性/上下文精度。**漂移仪表盘**（Arize Phoenix）每周监控检索质量和幻觉分数。

提示缓存是成本杠杆。Claude 4.5+ 和 GPT-5+ 支持缓存系统提示和检索上下文。在 60-80% 的命中率下，每次查询成本降低 3-5 倍。流水线必须设计为稳定的前缀（系统提示 + 重排序上下文在前）以实现高缓存命中率。

## 架构

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## 技术栈

- 摄入：Unstructured.io 或 docling 用于结构化文档；ColPali 用于视觉丰富的 PDF
- 向量数据库：pgvector + pgvectorscale（5000 万向量以下）；否则用 Qdrant Cloud
- 稀疏：Tantivy BM25，带字段权重
- 编排：LlamaIndex Workflows（摄入）+ LangGraph（对话）
- 重排序器：自托管 bge-reranker-v2-gemma-2b 或托管 Voyage rerank-2
- LLM：Claude Sonnet 4.7，启用提示缓存；备选自托管 Llama 3.3 70B
- 评估：RAGAS 0.2 在线，DeepEval 用于幻觉和越狱套件
- 可观测性：自托管 Langfuse，带注释队列；Arize Phoenix 用于漂移
- 防护：Llama Guard 4 输入/输出分类器，NeMo Guardrails v0.12 策略，Presidio PII 清洗
- 合规：代码块上的基于角色的访问标签；GDPR/HIPAA 的辖区标签

```
canary-rollout
```

## 开始构建

1. **摄入。** 使用 Unstructured 或 docling 解析你的语料库（严肃构建需要 1000-10000 个文档）。对于扫描/视觉密集的页面，通过 ColPali 路由。生成带摘要、角色标签、辖区标签的代码块。

2. **索引。** 稠密嵌入（Voyage-3 或 Nomic-embed-v2）存入 pgvector + pgvectorscale。BM25 通过 Tantivy 作为侧索引。角色和辖区过滤器作为负载。

3. **混合检索。** 先按角色+辖区过滤；然后并行稠密 + BM25；用倒数排名融合合并；top-20 给重排序器；top-5 给综合器。

4. **带提示缓存的综合。** 系统提示 + 静态策略放在缓存头中；重排序上下文作为缓存扩展；用户问题作为未缓存后缀。目标稳态 60-80% 缓存命中率。

5. **防护。** Llama Guard 4 用于输入；NeMo Guardrails 规则阻止领域外问题或策略禁止的主题；Presidio 清洗输出中的意外 PII；引用强制后置过滤器。

6. **黄金集。** 200 个由领域专家标注的问答对，包含（答案、引用）。按精确引用匹配、答案正确性、忠实度（RAGAS）对智能体评分。

7. **红队。** 50 个对抗性提示：越狱（PAIR、TAP）、PII 外泄尝试、领域外、跨辖区泄露。按通过/失败和严重性评分。

8. **漂移仪表盘。** Arize Phoenix 每周追踪检索质量（nDCG、引用忠实度）。下降 5% 时告警。

9. **成本报告。** Langfuse：提示缓存命中率、每次查询 Token、按阶段的 $/查询分解。

## 使用示例

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## 交付成果

`outputs/skill-production-rag.md` 描述了交付成果。一个带有合规标签的受监管领域聊天机器人，通过评分标准评估，通过实时漂移监控观测。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | RAGAS 忠实度 + 答案相关性 | 在黄金集（200 个问答）上的在线分数 |
| 20 | 引用正确性 | 具有可验证源锚点的答案比例 |
| 20 | 防护覆盖率 | Llama Guard 4 通过率 + 越狱套件结果 |
| 20 | 成本/延迟工程 | 提示缓存命中率、p95 延迟、$/查询 |
| 15 | 漂移监控仪表盘 | Phoenix 实时仪表盘，含每周检索质量趋势 |
| **100** | | |

## 练习

1. 在不同辖区（例如 HIPAA 与 GDPR 并行）下构建第二个语料库切片。在 20 个问题的跨辖区探针上展示角色+辖区过滤防止交叉泄露。

2. 在一周的生产流量中衡量提示缓存命中率。识别哪些查询破坏了缓存前缀。重新组织。

3. 添加 10k Token 摘要缓冲区的多轮记忆。衡量随着对话增长忠实度是否下降。

4. 将 Claude Sonnet 4.7 替换为自托管的 Llama 3.3 70B。衡量 $/查询和忠实度差异。

5. 添加"不确定"模式：如果 top 重排序分数低于阈值，智能体说"我没有可靠的引用"而不是回答。衡量虚假置信度的降低。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 提示缓存 | "缓存的系统 + 上下文" | Claude/OpenAI 功能：缓存的前缀 Token 在命中时打 6-9 折 |
| RAGAS | "RAG 评估器" | 忠实度、答案相关性、上下文精度的自动评分 |
| 黄金集 | "标注评估" | 200+ 个专家标注的带引用问答对；真实标准 |
| 辖区标签 | "合规标签" | 附加在代码块上的 GDPR/HIPAA/SOC2 范围；由检索过滤器强制执行 |
| 引用忠实度 | "基于事实的回答率" | 由可检索源范围支持的声明比例 |
| 漂移 | "检索质量衰减" | nDCG 或引用分数的每周变化；告警阈值 5% |
| 红队 | "对抗性评估" | 发布前的越狱、PII 提取、领域外探针 |

## 延伸阅读

- [Harvey AI](https://www.harvey.ai) — 法律生产技术栈参考
- [Glean 企业搜索](https://www.glean.com) — 企业级 RAG 参考
- [Mendable 文档](https://mendable.ai) — 开发者文档 RAG 参考
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — 托管摄入
- [Anthropic 提示缓存](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — 成本杠杆参考
- [RAGAS 0.2 文档](https://docs.ragas.io/) — 标准 RAG 评估框架
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移可观测性参考
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 安全分类器
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — 策略规则框架
