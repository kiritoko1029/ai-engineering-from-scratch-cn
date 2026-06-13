# 托管 LLM 平台 — Bedrock、Vertex AI、Azure OpenAI

> 三大超大规模云厂商，三种截然不同的策略。AWS Bedrock 是一个模型市场 — Claude、Llama、Titan、Stability、Cohere 统一在一个 API 背后。Azure OpenAI 是与 OpenAI 的独家合作，加上预配吞吐量单元（PTU）提供专用容量。Vertex AI 以 Gemini 为核心，拥有最佳的长上下文和多模态体验。2026 年，Artificial Analysis 测得 Azure OpenAI 在 Llama 3.1 405B 等效部署上的中位延迟约为 50 ms，Bedrock 约为 75 ms — PTU 解释了这一差距，因为专用容量胜过共享按需容量。决策规则不是"哪个最快"，而是"哪个模型目录和 FinOps 体系匹配我的产品"。本课教你做出有据可查的权衡选择，而非凭感觉行事。

**类型：** 学习
**语言：** Python（标准库，简易成本与延迟比较器）
**前置要求：** Phase 11（LLM 工程），Phase 13（工具与协议）
**所需时间：** 约60分钟

## 学习目标

- 说出三种平台策略（市场型 vs 独家型 vs Gemini 优先型），并将每种策略匹配到一个产品用例。
- 解释 Azure OpenAI 中预配吞吐量单元（PTU）的作用，以及为什么按需 Bedrock 在 405B 规模下通常慢约 25 ms。
- 画出每个平台的 FinOps 归因体系（Bedrock 应用推理配置文件 vs Vertex 项目按团队 vs Azure 作用域 + PTU 预留）。
- 撰写一份"至少双供应商"策略，并解释为什么单供应商锁定是 2026 年的昂贵错误。

## 问题所在

你为产品选定了 Claude 3.7 Sonnet。现在你需要部署它。你可以直接调用 Anthropic API，也可以通过 AWS Bedrock 调用，或者通过网关调用。直接 API 最简单；Bedrock 增加了 BAA、VPC 端点、IAM 和 CloudWatch 归因。网关增加了跨供应商的故障转移、统一计费和速率限制。

更深层的问题是目录。如果你需要在同一个产品中同时使用 Claude、Llama 和 Gemini，除非那个地方同时是 Bedrock 加 Vertex 加 Azure OpenAI，否则你无法从一个地方买到所有模型。超大规模云厂商并非可以互换 — 它们各自在谁拥有模型层这个问题上下了不同的赌注。

本课映射这三种赌注、延迟差距、FinOps 差距和锁定风险。

## 概念说明

### 三种策略

**AWS Bedrock** — 市场模式。Claude（Anthropic）、Llama（Meta）、Titan（AWS 自研）、Stability（图像）、Cohere（嵌入）、Mistral，外加图像和嵌入子目录。一个 API、一个 IAM 体系、一个 CloudWatch 导出。Bedrock 的赌注是：客户想要的是选择权，而非单一模型。

**Azure OpenAI** — 独家合作。你获得 GPT-4 / 4o / 5 / o 系列、DALL·E、Whisper，以及在 Azure 数据中心中对 OpenAI 模型的微调能力。"Azure OpenAI 服务"目录中没有非 OpenAI 模型 — 那些模型归属 Azure AI Foundry（独立产品）。Azure 的赌注是：OpenAI 始终保持前沿，客户希望在这一特定关系上获得企业级控制。

**Vertex AI** — Gemini 优先，其余其次。Gemini 1.5 / 2.0 / 2.5 Flash 和 Pro，加上 Model Garden（第三方）。Vertex 的赌注是多模态长上下文 — 100 万 token 的 Gemini 上下文是差异化优势。

### 规模下的延迟差距

Artificial Analysis 运行持续基准测试。在等效的 Llama 3.1 405B 部署（共享按需）上，Azure OpenAI 的中位首 token 延迟约为 50 ms；Bedrock 约为 75 ms。这一差距不是 AWS 的失败 — 而是容量模型的差异。Azure 出售 PTU（预配吞吐量单元），为你的租户预留 GPU 容量。Bedrock 的等效方案（预配吞吐量）存在，但每单元约 21 美元/小时起，且大多数客户仍使用共享按需。

按需共享容量与每个其他客户的流量竞争。专用容量则不然。如果你的产品 SLA 是 P99 TTFT < 100 ms，你要么在 Azure 上购买 PTU，要么购买 Bedrock 预配吞吐量，要么接受默认的方差。

### 预配吞吐量经济学

Azure PTU：一块预留的推理计算资源。对于可预测的工作负载，相比按需最高可节省约 70%。按小时固定计费，无论流量多少 — 即使空闲时也要为预留付费。盈亏平衡点通常在 40-60% 的持续利用率。

Bedrock 预配吞吐量：每小时 21-50 美元，取决于模型和区域。类似的数学计算 — 盈亏平衡点约为峰值利用率的一半。需要月度承诺。

Vertex 预配容量按 Gemini SKU 出售；定价因模型和区域而异，公开宣传较少。

### FinOps 体系 — 真正的差异化因素

**Bedrock 应用推理配置文件**是市场中最清晰的归因方式。用 `team`、`product`、`feature` 标记配置文件；通过它路由所有模型调用；CloudWatch 无需后处理即可按配置文件分解成本。2025 年新增，仍是超大规模云厂商原生中最细粒度的方案。

**Vertex** 的归因是按团队设置项目加无处不在的标签。你将每个团队建模为一个 GCP 项目，在每个资源上添加标签，使用 BigQuery Billing Export + DataStudio 进行汇总。工作量更大，但 BigQuery 允许你在成本数据上执行任意 SQL。

**Azure** 依赖订阅/资源组作用域加标签，PTU 预留作为一等成本对象。标签从资源组继承，而非请求，因此按请求归因需要 Application Insights 自定义指标或一个标记头部的网关。

总结：Bedrock 原生最清晰，Vertex 通过 BigQuery 最灵活，Azure 除非你进行埋点否则最不透明。

### 锁定是 2026 年的风险

当单一模型占主导时，单一超大规模云厂商的承诺是可以接受的。2026 年，前沿每月都在移动 — 一个季度是 Claude 3.7，下一个是 Gemini 2.5，再下一个是 GPT-5。锁定一个平台意味着你被三分之二的前沿拒之门外。

团队采用的模式：任何产品关键 LLM 调用至少使用两个供应商。Bedrock 加 Azure OpenAI 是常见的组合 — 一个用 Claude，另一个用 GPT，在它们之间做故障转移，使用同一个网关。成本上升可以忽略不计，因为网关路由最优；在中断期间可用性提升（如 Azure OpenAI 2025 年 1 月事件、AWS us-east-1 中断）是决定性的。

### 数据驻留、BAA 和受监管行业

Bedrock：大多数区域提供 BAA；VPC 端点；护栏。常见的金融科技默认选择。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；欧盟数据驻留；企业受监管默认选择。
Vertex：HIPAA、GDPR、按区域数据驻留；Google Cloud 的合规体系。

三者都满足基本要求。差异在于数据保留策略、日志处理方式，以及滥用监控是否读取你的流量（大多数默认启用；企业版可选择退出）。

### 你应该记住的数字

- Azure OpenAI 在 Llama 3.1 405B 等效部署上的中位 TTFT：约 50 ms（使用 PTU）。
- Bedrock 按需中位 TTFT：约 75 ms。
- Bedrock 预配吞吐量：每单元 21-50 美元/小时。
- Azure PTU 盈亏平衡点：约 40-60% 持续利用率。
- 高利用率下 PTU 相比按需的节省：最高可达 70%。

## 开始使用

`code/main.py` 在合成工作负载上比较三个平台 — 它建模按需 vs PTU 经济学、TTFT 方差和成本归因保真度。运行它以查看 PTU 在何时划算，以及市场模式的模型广度何时超过 TTFT 差距。

## 交付成果

本课生成 `outputs/skill-managed-platform-picker.md`。给定工作负载配置（所需模型、TTFT SLA、日流量、合规要求），它推荐一个主平台、一个备选平台和一个 FinOps 埋点方案。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下，Azure PTU 在 70B 类模型上比按需更划算？计算盈亏平衡点并与宣传的 40-60% 区间进行比较。
2. 你的产品需要 Claude 3.7 Sonnet 和 GPT-4o。设计一个双供应商部署 — 哪个模型放哪个超大规模云厂商，前面用什么网关，故障转移策略是什么？
3. 一个受监管的医疗客户要求 BAA、美国东部数据驻留和低于 100ms 的 P99 TTFT。选择一个平台并用三个具体功能来论证。
4. 你发现本月的 Bedrock 账单涨了 4 倍，但流量没有变化。没有应用推理配置文件的情况下，你如何找到罪魁祸首？有配置文件的话，需要多长时间？
5. 阅读 Azure OpenAI 和 Bedrock 的定价页面。对于每月 1 亿 token 的 Claude 工作负载，哪个更便宜 — 直接 Anthropic API、Bedrock 按需还是 Bedrock 预配吞吐量？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Bedrock | "AWS LLM 服务" | 跨 Claude、Llama、Titan、Mistral、Cohere 的模型市场 |
| Azure OpenAI | "Azure 的 ChatGPT" | 在 Azure 数据中心中独家提供 OpenAI 模型并附带企业控制 |
| Vertex AI | "Google 的 LLM" | 以 Gemini 为核心的平台，附带 Model Garden 提供第三方模型 |
| PTU | "专用容量" | 预配吞吐量单元 — 预留的推理 GPU，按小时计费 |
| 应用推理配置文件 | "Bedrock 标签" | 按产品的成本/使用量配置文件，支持标签，CloudWatch 原生 |
| Model Garden | "Vertex 目录" | Vertex AI 的第三方模型部分，与 Gemini 独立 |
| 至少双供应商 | "LLM 冗余" | 每条关键 LLM 路径跨至少 2 个超大规模云厂商运行的策略 |
| BAA | "HIPAA 文书" | 业务关联协议；处理 PHI 所需；三者均提供 |
| 滥用监控 | "日志监视器" | 供应商端对提示/输出的安全扫描；企业版可选择退出 |

## 延伸阅读

- [AWS Bedrock 定价](https://aws.amazon.com/bedrock/pricing/) — 权威费率卡和预配吞吐量定价。
- [Azure OpenAI 服务定价](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — PTU 经济学和费率卡。
- [Vertex AI 生成式 AI 定价](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini 层级和 Model Garden 附加费。
- [Artificial Analysis LLM 排行榜](https://artificialanalysis.ai/) — 跨供应商的持续延迟和吞吐量基准测试。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO 指南 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — 企业决策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — 归因机制并排比较。
