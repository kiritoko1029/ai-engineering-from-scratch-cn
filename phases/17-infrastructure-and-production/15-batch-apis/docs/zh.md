# Batch API —— 五折优惠已成为行业标配

> 每家主流供应商都提供了异步 batch API，附带五折优惠和约24小时的处理周期。OpenAI、Anthropic、Google 以及大多数推理平台（Fireworks batch 层级、Together batch）都实现了相同的模式。将 batch 与 prompt caching 叠加使用，离线管线的成本可降至同步未缓存方案的约10%。规则极其简单：如果不是交互式的，就该用 batch。内容生成管线、文档分类、数据提取、报告生成、批量标注、目录打标——任何能容忍24小时延迟的任务，在迁移到 batch 之前都是白白浪费的钱。2026年的生产模式是将每个新的 LLM 工作负载分流到三条通道：交互式（同步 + 缓存）、半交互式（异步队列 + 回退机制）、batch（离线 + 缓存输入叠加）。那些伪装成交互式但实际能容忍分钟级延迟的工作负载，浪费了大部分成本。

**类型：** 学习
**语言：** Python（标准库，简易 batch-vs-sync 成本模拟器）
**前置要求：** 第17阶段 · 第14课（Prompt 与语义缓存）
**所需时间：** 约45分钟

## 学习目标

- 列举三家供应商的 batch API（OpenAI、Anthropic、Google）及其通用的五折优惠 + 24小时处理周期保证。
- 计算在一个离线分类工作负载上叠加 batch + cached-input 的成本，并与同步未缓存基线进行对比。
- 将工作负载分流为交互式 / 半交互式 / batch，并说明理由。
- 识别两个陷阱：部分交互性（用户期望快于24小时）和输出 schema 漂移（batch 文件格式因供应商而异）。

## 问题所在

你的团队部署了一个每夜运行的报告生成管线。50000份文档，逐一摘要，对摘要进行聚类，然后起草一份高管简报。同步运行需要4小时，每晚花费2000美元。你听说了 batch API。

Batch 能给你打五折。你还在系统 prompt（所有5万次调用共享）上启用了 prompt caching。两者叠加后，账单降至每晚180美元——约为基线的9%。同样的管线，只改了三处配置。

Batch 是 LLM 成本工具箱中最便宜的杠杆，却没人去拉。原因大多出在组织层面：团队总想着"实时"，但 SLA 实际上是"明早之前完成"。本课要解决的问题就是：别再把90%的账单白白留在桌面上。

## 概念说明

### 三家 batch API

**OpenAI Batch API**：上传包含请求列表的 JSONL 文件。承诺24小时内完成（实际通常2-8小时）。输入和输出 token 均享五折优惠。端点为 `/v1/batches`。符合缓存条件的输入还能叠加 cached-input 定价。

**Anthropic Message Batches**：JSONL 上传。24小时处理周期。五折优惠。支持 `cache_control`——缓存写入是显式的，读取在 batch 内自动发生。

**Google Vertex AI Batch Prediction**：BigQuery 或 GCS 输入。Gemini 享有类似的五折优惠。与 Vertex 管线集成。

### 语义：异步，而非缓慢

Batch 的含义是"我承诺在24小时内返回结果"——而不是"这需要24小时才能完成"。典型 P50 为2-6小时。供应商会在 GPU 资源利用率较低的非高峰时段调度你的 batch。

### 与缓存叠加

以5万份文档摘要为例，共享相同的4K token 系统 prompt：

- 同步未缓存：50000 ×（$input × 4000 + $output × 200），按全价计算。
- 同步已缓存：系统 prompt 在首次写入后被缓存；剩余49999次调用的输入成本降低10倍。
- Batch + 缓存：在上述基础上，读取和写入均再享五折优惠。

叠加效果：batch + cache = 同步未缓存账单的约10%。任何在离线运行且有共享系统 prompt 的工作负载都应该使用这种方式。

### 工作负载分流

**交互式**——用户等待响应。首 token 延迟（TTFT）很关键。使用同步调用 + prompt caching。不能用 batch。

**半交互式**——用户提交任务，几分钟后回来查看结果。使用异步队列，当 batch 不可用时回退到同步。适用于中等规模的 RAG 索引等场景。

**Batch**——用户期望"明早之前"或"下个小时"拿到结果。内容管线、大规模分类、离线分析。始终用 batch，始终叠加缓存。

常见错误：因为管线是生产环境就把所有任务都归类为交互式。生产环境不是延迟规格——SLA 才是。

### 部分交互性陷阱

某些功能看起来是交互式的，但实际能容忍5-10分钟的延迟。例如：一个带有"刷新"按钮的每夜客户健康报告。用户点击刷新，等10分钟完全可以接受。团队却把它做成了同步的。50个并发刷新请求的成本是通过 batch + 邮件推送方式的10倍。

要问的问题是："24小时对这个用户意味着什么？"如果答案是"他们根本不会注意到"，那就用 batch。

### 输出 schema 陷阱

Batch 文件格式因供应商而异：

- OpenAI：JSONL，每行一个请求。
- Anthropic：JSONL，每行一条消息；响应格式内嵌。
- Vertex：BigQuery 表或 GCS 前缀下的 TFRecord。

要写一个跨供应商的"统一 batch 客户端"，意味着每个供应商都需要适配代码。号称支持多供应商 batch 的网关（Portkey、LiteLLM 部分层级）也只是对原始格式做了薄封装。

### 你应该记住的数字

- 各供应商的 batch 折扣：输入 + 输出统一五折。
- 处理周期 SLA：保证24小时，典型 P50 为2-6小时。
- 叠加 batch + cached-input：约为同步未缓存成本的10%。
- 工作负载分流规则：如果24小时延迟可接受，始终用 batch。

## 开始使用

`code/main.py` 计算5万份文档工作负载在同步、同步+缓存、batch、batch+缓存四种模式下的成本，并以金额和百分比形式报告节省量。

## 部署产出

本课生成 `outputs/skill-batch-triager.md`。根据工作负载特征，将其分流为交互式/半交互式/batch，并估算节省量。

## 练习

1. 运行 `code/main.py`。对于一个10万份文档、3K token 系统 prompt、500 token 输出的管线，计算完整叠加方案（batch + cache）相比同步基线的节省量。
2. 选择你所了解的真实产品中的三个功能，将每个功能分流为交互式/半交互式/batch。
3. 用户抱怨他们的报告花了3小时。这是 batch 分流错误还是合理的交互式场景？写出判断标准。
4. 你的 batch API 返回 SLA 是24小时，但 P99 是20小时。你如何向用户传达这一点——在边界情况下下游系统会如何表现？
5. 计算盈亏平衡点：共享前缀达到多长时，batch + cache 会比在自有预留 GPU 上离线运行更便宜？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Batch API | "异步折扣" | 五折优惠，24小时处理周期 |
| JSONL | "batch 格式" | 每行一个 JSON 请求；OpenAI/Anthropic 标准格式 |
| Message Batches | "Anthropic batch" | Anthropic 的 batch API 产品名称 |
| Batch prediction | "Vertex batch" | Vertex AI 的 batch API 产品 |
| Turnaround SLA | "24小时承诺" | 保证值，非典型值；典型为2-6小时 |
| 工作负载分流 | "交互性决策" | 交互式 / 半交互式 / batch 的路由决策 |
| 输出 schema | "响应格式" | 各供应商的 JSONL 布局；不可移植 |
| 叠加折扣 | "batch + cache" | 两者同时生效时，约为未缓存同步账单的10% |

## 延伸阅读

- [OpenAI Batch API](https://platform.openai.com/docs/guides/batch)——JSONL 格式和 `/v1/batches` 语义。
- [Anthropic Message Batches](https://docs.anthropic.com/en/docs/build-with-claude/batch-processing)——batch 格式和 `cache_control` 的交互。
- [Vertex AI Batch Prediction](https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction)——Gemini batch 语义。
- [Finout — OpenAI vs Anthropic API Pricing 2026](https://www.finout.io/blog/openai-vs-anthropic-api-pricing-comparison)
- [Zen Van Riel — LLM API Cost Comparison 2026](https://zenvanriel.com/ai-engineer-blog/llm-api-cost-comparison-2026/)
