# 推理平台经济学 — Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> **定价快照日期 2026-04。** 以下数字声明反映本课发布时各供应商的费率卡；在下游引用前请核实所链接的文档。

> 2026 年的推理市场已不再是 GPU 时间租赁。它分化为三个方向：定制芯片（Groq、Cerebras、SambaNova）、GPU 平台（Baseten、Together、Fireworks、Modal）和 API 优先市场（Replicate、DeepInfra）。Fireworks 在 2026 年 5 月 1 日将每 GPU 每小时价格上涨了 1 美元，40 亿美元估值加上每天 10 万亿+ token 的处理量说明了以量驱动的模式是可行的。Baseten 在 2026 年 1 月以 50 亿美元估值完成 3 亿美元 E 轮融资。竞争定位规则很简单：Fireworks 优化延迟，Together 优化目录广度，Baseten 优化企业级打磨，Modal 优化 Python 原生开发体验，Replicate 优化多模态覆盖，Anyscale 优化分布式 Python。本课为你提供一个可以交给创始人的矩阵。

**类型：** 学习
**语言：** Python（标准库，简易单次调用经济学比较器）
**前置要求：** Phase 17 · 01（托管 LLM 平台），Phase 17 · 04（vLLM 服务内部机制）
**所需时间：** 约60分钟

## 学习目标

- 说出三个细分市场（定制芯片、GPU 平台、API 优先），并将每个供应商映射到一个细分市场。
- 解释为什么"按 token"的 API 定价模型趋向于服务引擎的成本曲线，而非硬件成本曲线。
- 计算至少三个供应商的有效单请求成本，并解释按分钟计费（Baseten、Modal）何时优于按 token 计费。
- 识别对于给定工作负载（无服务器突发型、稳定高吞吐量、微调变体、多模态），哪个平台是正确的默认选择。

## 问题所在

你评估了托管超大规模云厂商平台。你决定需要一个更专注、更快的供应商 — Fireworks 追求延迟，Together 追求广度，Baseten 追求微调定制模型。现在你有六个真实选择，而定价页面无法直接对齐。Fireworks 显示每百万 token 价格；Baseten 显示每分钟价格；Modal 显示每秒价格；Replicate 显示每次预测价格。不建模工作负载就无法进行直接比较。

更糟糕的是，每个定价页面背后的商业模式不同。Fireworks 在共享 GPU 上运行自己的定制引擎（FireAttention）；按 token 费率反映了它们的利用率曲线。Baseten 给你 Truss + 专用 GPU；按分钟反映了独占性。Modal 是真正的 Python 无服务器 — 按秒计费，亚秒级冷启动。同样的输出（LLM 响应），三种不同的成本函数。

本课对这六家进行建模，告诉你何时每家胜出。

## 概念说明

### 三个细分市场

**定制芯片** — Groq（LPU）、Cerebras（WSE）、SambaNova（RDU）。在相同模型上，解码速度通常比 GPU 集群快 5-10 倍。按 token 价格更高（Groq 在 2025 年末对 Llama-70B 的价格约为 0.99 美元/百万），但对于延迟敏感的用例无可匹敌。Groq 是语音代理和实时翻译的生产首选。

**GPU 平台** — Baseten、Together、Fireworks、Modal、Anyscale。运行在 NVIDIA（2026 年的 H100、H200、B200）或偶尔 AMD 上。介于"原始 GPU 租赁"（RunPod、Lambda）和"超大规模云厂商托管服务"（Bedrock）之间的经济层。

**API 优先市场** — Replicate、DeepInfra、OpenRouter、Fal。广目录，按预测或按秒计费，强调首次调用时间。

### Fireworks — 延迟优化的 GPU 平台

- FireAttention 引擎（定制）；宣传在等效配置下比 vLLM 延迟低 4 倍。
- 批处理层约为无服务器费率的 50%，适用于非交互式工作负载。
- 微调模型与基础模型同价服务 — 与对你的 LoRA 收取溢价的供应商相比，这是一个真正的差异化优势。
- 2026 年中：2026 年 5 月 1 日起按需 GPU 租赁涨价 1 美元/小时。大规模量价可协商。
- 财务信号：40 亿美元估值，每天处理 10 万亿+ token。

### Together — 广度优化

- 200+ 模型，包括上游发布后数天内上线的开源版本。
- 在等效 LLM 模型上比 Replicate 便宜 50-70% — "AI 原生云"的定位是量和目录。
- 推理 + 微调 + 训练集成在一个 API 中。

### Baseten — 企业级优化

- Truss 框架：模型打包，将依赖、密钥、服务配置整合在一个清单中。
- GPU 范围从 T4 到 B200。按分钟计费，冷启动缓解合理。
- SOC 2 Type II、HIPAA 就绪。常见的金融科技和医疗选择。
- 50 亿美元估值，2026 年 1 月 E 轮融资（3 亿美元，来自 CapitalG、IVP、NVIDIA）。

### Modal — Python 原生优化

- 纯 Python 基础设施即代码。用 `@modal.function(gpu="A100")` 装饰函数，一条命令即可部署。
- 按秒计费。预热后冷启动 2-4 秒；小模型 <1 秒。
- 2025 年 B 轮融资 8700 万美元，估值 11 亿美元。在独立调查中获得最高开发者体验评分。

### Replicate — 多模态广度

- 按预测计费。图像、视频和音频模型的默认平台。
- 集成生态（Zapier、Vercel、CMS 插件）。
- 在 LLM 按 token 费率上竞争力较弱，但在多模态多样性上胜出。

### Anyscale — Ray 原生

- 基于 Ray 构建；RayTurbo 是 Anyscale 的专有推理引擎（与 vLLM 竞争）。
- 最适合分布式 Python 工作负载，其中推理步骤是更大计算图中的一个节点。
- 托管 Ray 集群；与 Ray AIR 和 Ray Serve 紧密集成。

### 按 token vs 按分钟 — 各自胜出的时机

当工作负载对延迟不敏感且具有突发性时，按 token 计费合理 — 你只为使用的部分付费。当利用率高且可预测时，按分钟计费合理 — 一旦 GPU 饱和，你就超过了按 token 的成本。

粗略规则：对于专用 GPU 持续利用率超过约 30% 的工作负载，按分钟（Baseten、Modal）开始优于按 token（Fireworks、Together）。低于此阈值，按 token 胜出，因为空闲时无需付费。

### 定制引擎是真正的护城河

每个平台都声称拥有超越 vLLM 和 SGLang 的定制引擎。FireAttention、RayTurbo、Baseten 的推理栈。定制引擎声明接近营销 — 诚实的说法是 vLLM + SGLang 占据了约 80% 的生产开源推理，平台层的差异化在于开发体验、归因和 SLA。

### 你应该记住的数字

- Fireworks GPU 租赁：2026 年 5 月 1 日起涨价 1 美元/小时。
- Fireworks 声称：在等效配置下比 vLLM 延迟低 4 倍。
- Together：在 LLM 上比 Replicate 便宜 50-70%。
- Baseten 估值：50 亿美元（E 轮，2026 年 1 月，3 亿美元）。
- Modal 估值：11 亿美元（B 轮，2025 年）。
- 持续利用率超过约 30% 时，按分钟优于按 token。

```figure
cost-per-token
```

## 开始使用

`code/main.py` 在合成工作负载上跨定价模型比较六家供应商。报告每天成本和有效每百万 token 成本。运行它以找到按 token 和按分钟之间的盈亏平衡点。

## 交付成果

本课生成 `outputs/skill-inference-platform-picker.md`。给定工作负载配置、SLA 和预算，选择主要推理平台并命名候选平台。

## 练习

1. 运行 `code/main.py`。在什么持续利用率下，Baseten（按分钟）在一个 H100 上运行 70B 模型时优于 Fireworks（按 token）？自己推导交叉点并与经验法则进行比较。
2. 你的产品提供图像生成、聊天和语音转文字。为每种模态选择平台，并命名统一它们的网关模式。
3. Fireworks 将你的主要模型涨价 1 美元/小时。如果 40% 的流量转移到批处理层（50% 折扣），建模混合成本影响。
4. 一个受监管客户要求 SOC 2 Type II + HIPAA + 专用 GPU。哪三个平台可行，哪个在 FinOps 上胜出？
5. 比较在 Fireworks 无服务器、Together 按需、Baseten 专用和 Replicate API 上运行 Llama 3.1 70B 每 1000 次预测的成本。每天 10 次预测时哪个最便宜？每天 10,000 次呢？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 定制芯片 | "非 GPU 芯片" | Groq LPU、Cerebras WSE、SambaNova RDU — 为解码优化 |
| FireAttention | "Fireworks 引擎" | 定制注意力内核；宣传比 vLLM 延迟低 4 倍 |
| Truss | "Baseten 的格式" | 模型打包清单；依赖 + 密钥 + 服务配置 |
| 按 token | "API 定价" | 按消耗的 token 计费；空闲时不付费 |
| 按分钟 | "专用定价" | 按 GPU 实际运行时间计费；高利用率时胜出 |
| 按预测 | "Replicate 定价" | 按模型调用计费；常见于图像/视频 |
| RayTurbo | "Anyscale 引擎" | Ray 上的专有推理；在 Ray 集群上与 vLLM 竞争 |
| 批处理层 | "50% 折扣" | 降低费率的非交互式队列；Fireworks、OpenAI 常见 |
| 微调按基础价 | "Fireworks LoRA" | LoRA 服务的请求按基础模型费率计费（差异化优势） |

## 延伸阅读

- [Fireworks 定价](https://fireworks.ai/pricing) — 按 token 费率、批处理层、GPU 租赁。
- [Baseten 定价](https://www.baseten.co/pricing/) — 按分钟费率、承诺容量、企业层级。
- [Modal 定价](https://modal.com/pricing) — 按秒 GPU 费率和免费层。
- [Together AI 定价](https://www.together.ai/pricing) — 模型目录和按 token 费率。
- [Anyscale 定价](https://www.anyscale.com/pricing) — RayTurbo 和托管 Ray 定价。
- [Northflank — Fireworks AI 替代方案](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — 比较评估。
- [Infrabase — AI 推理 API 供应商 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — 供应商格局。
