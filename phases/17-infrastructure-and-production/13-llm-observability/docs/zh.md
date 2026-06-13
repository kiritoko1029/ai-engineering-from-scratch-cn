# LLM 可观测性技术栈选型

> 2026 年的可观测性市场分为两类。开发平台（LangSmith、Langfuse、Comet Opik）将监控与评估、提示管理、会话回放捆绑在一起。网关/埋点工具（Helicone、SigNoz、OpenLLMetry、Phoenix）专注于遥测。Langfuse 是 MIT 许可证核心，OSS 平衡良好（免费云每月 50K 事件）。Phoenix 是 OpenTelemetry 原生，Elastic License 2.0 — 出色的漂移/RAG 可视化，不是持久化生产后端。Arize AX 使用零拷贝 Iceberg/Parquet 集成，声称比单体可观测性便宜 100 倍。LangSmith 在 LangChain/LangGraph 方面领先，每用户每月 39 美元，仅企业版可自托管。Helicone 基于代理，15-30 分钟设置，免费 100K 请求/月，但在智能体追踪方面深度不足。常见生产模式：网关（Helicone/Portkey）+ 评估平台（Phoenix/TruLens）通过 OpenTelemetry 粘合。

**类型：** 学习
**语言：** Python（标准库，简易追踪采样模拟器）
**前置要求：** Phase 17 · 08（推理指标），Phase 14（智能体工程）
**所需时间：** 约60分钟

## 学习目标

- 区分开发平台（捆绑：评估 + 提示 + 会话）和网关/遥测工具（仅追踪 + 指标）。
- 将六个主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）映射到其许可证、定价和最佳用例。
- 解释让你将网关工具与独立评估平台组合的 OpenTelemetry 粘合模式。
- 说出 2026 年的成本差异化因素（Arize AX 的零拷贝方法 vs 单体摄入），并说明大约 100 倍的乘数。

## 问题所在

你发布了一个 LLM 功能。它能工作。但你对提示失败、工具循环、延迟回退、成本飙升或提示缓存命中率没有可见性。你搜索"LLM 可观测性"，得到八个工具，都在三个不同价格点声称解决同一个问题。

它们解决的不是同一个问题。LangSmith 回答"为什么这次 LangGraph 运行失败了？"Phoenix 回答"我的 RAG 管道是否在漂移？"Helicone 回答"哪个应用在烧 token？"Langfuse 回答"我能自托管整个东西吗？"不同的工具，不同的受众。

选择涉及四个维度：技术栈（LangChain？原生 SDK？多供应商？）、许可证容忍度（仅 MIT？Elastic 可以？商业没问题？）、预算（免费层？100 美元/月？1000 美元/月？）和自托管（必须？最好有？永远不要？）。

## 概念说明

### 两个类别

**开发平台**将可观测性与评估、提示管理、数据集版本控制、会话回放捆绑在一起。你运行实验，查看哪个提示有效，将新提示与旧赢家进行数据集回归。LangSmith、Langfuse、Comet Opik。

**网关/遥测工具**对推理调用进行埋点 — 提示、响应、token、延迟、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。极简。可通过 OpenTelemetry 与独立评估工具组合。

### Langfuse — OSS 平衡

- 核心 Apache / MIT 许可证；通过 Docker 自托管。
- 云免费层：每月 50K 事件。付费：团队每月 29 美元。
- 评估、提示管理、追踪、数据集。合理覆盖所有四个开发平台功能。
- 最佳场景：你想要 LangSmith 级功能但必须自托管或保持 OSS 许可证。

### Phoenix（Arize）— 遥测优先，OpenTelemetry 原生

- Elastic License 2.0；自托管简单。
- 在 RAG 和漂移可视化方面出色。嵌入空间散点图作为一等功能发布。
- 不是为持久化生产后端设计的 — 主要是开发时可观测性。
- 最佳场景：RAG 管道开发、漂移调试，与独立网关配合用于生产。

### Arize AX — 规模化方案

- 商业化。通过 Iceberg/Parquet 的零拷贝数据湖集成。
- 声称在规模化下比单体可观测性（Datadog 级）便宜约 100 倍。数学原理：你在自己的 S3 Parquet 中存储追踪；Arize 直接读取。
- 最佳场景：每天 >1000 万追踪，已有数据湖，想要 LLM 特定仪表板而不想付 Datadog 价格。

### LangSmith — LangChain/LangGraph 优先

- 商业化，每用户每月 39 美元。仅企业版可自托管。
- LangChain 和 LangGraph 技术栈的最佳选择。如果你不使用其中之一，吸引力较小。
- 最佳场景：团队已承诺使用 LangChain，愿意付费。

### Helicone — 基于代理的最小可行方案

- 15-30 分钟设置，只需将 `OPENAI_API_BASE` 替换为 Helicone 代理。
- MIT 许可证；免费每月 100K 请求，付费每月 20 美元+。
- 包含故障转移、缓存、速率限制 — 也充当网关。
- 在智能体/多步骤追踪方面深度不足。
- 最佳场景：快速启动，单技术栈应用，需要网关 + 可观测性合二为一。

### Opik（Comet）— OSS 开发平台

- Apache 2.0，完全开源。
- 与 Langfuse 功能集相似，具有 Comet 背景。
- 最佳场景：ML 团队已使用 Comet，想在同一面板中获得 LLM 可观测性。

### SigNoz — OpenTelemetry 优先的全栈 APM

- Apache 2.0。通过 OpenTelemetry 处理通用 APM 加 LLM。
- 最佳场景：跨服务和 LLM 调用的统一可观测性。

### 粘合层：OpenTelemetry + GenAI 语义约定

OpenTelemetry 在 2025 年末发布了 GenAI 语义约定（`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`）。消费 OTel 的工具可以互操作。正在形成的生产模式：

1. 从每个 LLM 调用发出带 GenAI 约定的 OTel。
2. 路由到网关（Helicone / Portkey）用于日常。
3. 双发到评估平台（Phoenix / Langfuse）用于回归。
4. 归档到数据湖（Iceberg）用于通过 Arize AX 或 DuckDB 进行长期分析。

### 陷阱：在错误的层埋点

在智能体框架内部埋点（例如添加 LangSmith 追踪）将你耦合到该框架。在 HTTP/OpenAI-SDK 层埋点（通过 OpenLLMetry 或你的网关）是可移植的。

### 采样 — 你无法保留一切

在每天 >100 万请求时，全追踪保留成本超过 LLM 调用本身。按规则采样：100% 错误、100% 高成本、5% 成功。始终保留聚合数据；保留长尾的原始数据。

### 你应该记住的数字

- Langfuse 免费云：每月 50K 事件。
- LangSmith：每用户每月 39 美元。
- Helicone 免费：每月 100K 请求。
- Arize AX 声称：在规模化下比单体便宜约 100 倍。
- OpenTelemetry GenAI 约定：2025 年发布，2026 年广泛采纳。

## 开始使用

`code/main.py` 在保留策略（100% 摄入、采样、采样 + 错误）下模拟一天 100 万追踪。报告存储成本以及每种策略下丢失的内容。

## 交付成果

本课生成 `outputs/skill-observability-stack.md`。给定技术栈、规模、预算、许可证立场，选择工具。

## 练习

1. 你的团队使用 LangChain，想要 OSS 自托管可观测性。选择 Langfuse 或 Opik 并论证。
2. 每天 500 万追踪，Datadog 报价每月 15 万美元，计算 Arize AX 的盈亏平衡点。
3. 设计一组你的组织准则应强制要求在每个 LLM 调用上使用的 OpenTelemetry GenAI 属性。
4. 论证 Phoenix 单独是否足以用于生产。何时不够用？
5. Helicone 代理开销 20ms。在 P99 TTFT 300 ms 下，这可以接受吗？如果 SLA 是 100 ms 呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| OpenLLMetry | "LLM 的 OTel" | LLM 的开源 OpenTelemetry 埋点 |
| GenAI 约定 | "OTel 属性" | LLM 调用的标准 OTel 属性名称 |
| LangSmith | "LangChain 可观测性" | 与 LangChain 生态捆绑的商业平台 |
| Langfuse | "开源版 LangSmith" | MIT 开源，功能集相似 |
| Phoenix | "Arize 开发工具" | OpenTelemetry 原生的开发/评估平台 |
| Arize AX | "规模化可观测性" | 商业零拷贝 Iceberg/Parquet 可观测性 |
| Helicone | "代理可观测性" | 收集 LLM 遥测的 HTTP 代理 + 网关功能 |
| Opik | "Comet LLM" | 来自 Comet 的 Apache 2.0 OSS 开发平台 |
| 会话回放 | "追踪重放" | 重放包含工具调用的完整智能体会话 |
| 评估 | "离线测试" | 在标记数据集上运行候选模型/提示 |

## 延伸阅读

- [SigNoz — 2026 年顶级 LLM 可观测性工具](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX 替代方案分析](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — 设置 Langfuse、LangSmith、Helicone、Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI 语义约定](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix 文档](https://docs.arize.com/phoenix)
- [Helicone 文档](https://docs.helicone.ai/)
