# 毕业项目 11 — LLM 可观测性与评估仪表盘

> Langfuse 转为开放核心。Arize Phoenix 发布了 2026 年 GenAI 语义规范映射。Helicone 和 Braintrust 都加倍投入每用户成本归属。Traceloop 的 OpenLLMetry 成为事实上的 SDK 插桩标准。生产形态是 ClickHouse 存储追踪、Postgres 存储元数据、Next.js 做 UI，以及一队评估任务（DeepEval、RAGAS、LLM-judge）在采样追踪上运行。构建一个自托管版本，从至少四个 SDK 系列摄入，并演示在五分钟内捕获注入的回归。

**类型：** 毕业项目
**语言：** TypeScript（UI）、Python / TypeScript（摄入 + 评估）、SQL（ClickHouse）
**前置要求：** 阶段 11（LLM 工程）、阶段 13（工具）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P11 · P13 · P17 · P18
**所需时间：** 25 小时

## 问题所在

2026 年，每个运行生产流量的 AI 团队都在模型旁边保持一个可观测性平面。成本归属。幻觉检测。漂移监控。越狱信号。SLO 仪表盘。PII 泄露告警。开源参考——Langfuse、Phoenix、OpenLLMetry——收敛到 OpenTelemetry GenAI 语义规范作为摄入 schema。你现在可以用一个 SDK 插桩 OpenAI、Anthropic、Google、LangChain、LlamaIndex 和 vLLM，并发送兼容的 Span。

你将构建一个自托管仪表盘，从至少四个 SDK 系列摄入，在采样追踪上运行一小组评估任务，检测漂移并告警。衡量标准：给定一个故意注入的回归（一个开始产生 PII 的提示），仪表盘在五分钟内捕获它并触发告警。

## 概念说明

摄入是 OTLP HTTP。SDK 产生 GenAI 语义规范 Span：`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`、`gen_ai.response.id`、`llm.prompts`、`llm.completions`。Span 落入 ClickHouse 进行列式分析；元数据（用户、会话、应用）落入 Postgres。

评估作为批量任务在采样追踪上运行。DeepEval 评分忠实度、毒性和答案相关性。RAGAS 在追踪携带检索上下文时评分检索指标。自定义 LLM-judge 运行领域特定检查（PII 泄露、策略外响应）。评估运行写回同一 ClickHouse，作为链接到父追踪的评估 Span。

漂移检测随时间观察嵌入空间分布（提示嵌入的 PSI 或 KL 散度）加上评估分数趋势。告警推送到 Prometheus Alertmanager，然后到 Slack / PagerDuty。UI 是 Next.js 15 + Recharts。

## 架构

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## 技术栈

- 摄入：OpenTelemetry SDK + GenAI 语义规范；OTLP HTTP 传输
- 采集器：OpenTelemetry Collector，带尾采样处理器（用于成本控制）
- 存储：ClickHouse 存储 Span，Postgres 存储元数据，S3 存储原始事件归档
- 评估：DeepEval、RAGAS 0.2、Arize Phoenix 评估器包、自定义 LLM-judge
- 漂移：每周对池化提示嵌入（sentence-transformers）计算 PSI / KL
- 告警：Prometheus Alertmanager -> Slack / PagerDuty
- UI：Next.js 15 App Router + Recharts + server actions
- 开箱即用支持的 SDK：OpenAI、Anthropic、Google GenAI、LangChain、LlamaIndex、vLLM

## 开始构建

1. **采集器配置。** OpenTelemetry Collector，带 OTLP HTTP 接收器、保留 100% 错误追踪和 10% 成功追踪的尾采样器，以及到 ClickHouse 和 S3 的导出器。

2. **ClickHouse schema。** `spans` 表，列镜像 GenAI 语义规范：`gen_ai_system`、`gen_ai_request_model`、`input_tokens`、`output_tokens`、`latency_ms`、`prompt_hash`、`trace_id`、`parent_span_id`，加 JSON bag 用于长负载。按 user_id 和 app_id 添加二级索引。

3. **SDK 覆盖测试。** 使用每个 SDK（OpenAI、Anthropic、Google、LangChain、LlamaIndex、vLLM）编写小型客户端应用，配合 OpenLLMetry 自动插桩。验证每个都产生落入 ClickHouse 的标准 GenAI Span。

4. **评估任务。** 定时任务读取最近 15 分钟的采样追踪，运行 DeepEval 忠实度、毒性和答案相关性。输出是链接到父追踪的评估 Span。

5. **自定义 LLM-judge。** PII 泄露 judge：给定响应，调用防护 LLM 评分 PII 泄露可能性。高分响应进入分诊队列。

6. **漂移检测。** 每周任务计算本周池化提示嵌入与前 4 周基线之间的 PSI。如果 PSI 超过阈值，告警。

7. **仪表盘。** Next.js 15，页面包括：概览（Span/秒、成本/用户、p95 延迟）、追踪（搜索 + 瀑布图）、评估（忠实度趋势、毒性）、漂移（PSI 随时间变化）、告警。

8. **告警链。** Prometheus 导出器读取评估分数聚合和延迟百分位；Alertmanager 路由到 Slack（警告）和 PagerDuty（严重违规）。

9. **回归探针。** 注入 bug：被评估的聊天机器人 1% 的时间泄露假 SSN。衡量 MTTR：从 bug 部署到 Slack 告警。

## 使用示例

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## 交付成果

`outputs/skill-llm-observability.md` 是交付成果。给定一个 LLM 应用，仪表盘摄入其追踪、运行评估、对漂移告警，并在 Next.js 中展示成本/用户分解。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 追踪 schema 覆盖 | 产生标准 GenAI Span 的 SDK 系列数量（目标：6+） |
| 20 | 评估正确性 | DeepEval / RAGAS 分数 vs 人工标注集 |
| 20 | 仪表盘 UX | 注入回归的 MTTR（目标低于 5 分钟） |
| 20 | 成本/规模 | 持续摄入 1k Span/秒无积压 |
| 15 | 告警 + 漂移检测 | Prometheus/Alertmanager 链端到端验证 |
| **100** | | |

## 练习

1. 为 Haystack 框架添加自定义插桩。验证标准 Span 带着忠实的 `gen_ai.*` 属性落入 ClickHouse。

2. 在同一追踪上将 DeepEval 替换为 Phoenix 评估器。衡量两个评估引擎之间的分数漂移。

3. 增强漂移检测器：按 app-id 而非全局计算 PSI。展示每应用漂移轨迹。

4. 添加"用户影响"页面：每用户成本和每用户失败率，带迷你折线图。

5. 构建尾采样策略：保留 100% 毒性 > 0.5 的追踪，加上其余的 10% 分层采样。衡量引入的采样偏差。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| GenAI 语义规范 | "OTel LLM 属性" | 2025 年 OpenTelemetry 规范，用于 LLM Span 属性（系统、模型、Token） |
| 尾采样 | "追踪后采样" | 采集器在追踪完成后决定保留或丢弃（可查看错误） |
| PSI | "总体稳定性指数" | 比较两个分布的漂移指标；> 0.2 通常表示有意义的漂移 |
| LLM-judge | "评估即模型" | 一个 LLM 按评分标准为另一个 LLM 的输出打分（忠实度、毒性、PII） |
| 尾采样策略 | "保留规则" | 决定哪些追踪持久化 vs 丢弃的规则；错误 + 采样率 |
| 评估 Span | "链接的评估追踪" | 携带评估分数、链接到原始 LLM 调用 Span 的子 Span |
| 每用户成本 | "单位经济学" | 归属到 user_id 的美元成本（按时间窗口）；关键产品指标 |

## 延伸阅读

- [Langfuse](https://github.com/langfuse/langfuse) — 开放核心可观测性平台参考
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — 漂移支持强的备选参考
- [OpenLLMetry（Traceloop）](https://github.com/traceloop/openllmetry) — 自动插桩 SDK 系列
- [OpenTelemetry GenAI 语义规范](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 摄入 schema
- [Helicone](https://www.helicone.ai) — 备选托管可观测性
- [Braintrust](https://www.braintrust.dev) — 备选评估优先平台
- [ClickHouse 文档](https://clickhouse.com/docs) — 列式 Span 存储
- [DeepEval](https://github.com/confident-ai/deepeval) — 评估器库
