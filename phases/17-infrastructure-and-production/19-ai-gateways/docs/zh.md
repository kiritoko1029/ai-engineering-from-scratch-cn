# AI 网关 —— LiteLLM、Portkey、Kong AI Gateway、Bifrost

> 网关位于应用与模型提供商之间。核心功能包括提供商路由、故障转移、重试、速率限制、密钥引用、可观测性和护栏。2026 年市场格局：**LiteLLM** 采用 MIT 开源协议，支持 100+ 提供商，兼容 OpenAI 接口，但在约 2000 RPS 时出现瓶颈（8 GB 内存占用，公开基准测试中出现级联故障）；最适合 Python 应用、<500 RPS、开发/原型阶段。**Portkey** 定位为控制平面（护栏、PII 脱敏、越狱检测、审计追踪），2026 年 3 月转为 Apache 2.0 开源，每请求增加 20-40 毫秒延迟开销，生产版每月 49 美元。**Kong AI Gateway** 基于 Kong Gateway 构建 —— Kong 自身在相同的 12 CPU 基准测试中：比 Portkey 快 228%，比 LiteLLM 快 859%；定价为每模型每月 100 美元（Plus 版最多 5 个模型）；如果已在使用 Kong，是企业级的理想选择。**Bifrost**（Maxim AI）—— 支持可配置退避策略的自动重试，当 OpenAI 返回 429 时自动回退到 Anthropic。**Cloudflare / Vercel AI Gateway** —— 托管式、零运维、基础重试功能。数据驻留要求是自托管决策的关键驱动因素；Portkey 和 Kong 处于中间位置，提供开源 + 可选托管方案。

**类型：** 学习
**语言：** Python（标准库，网关路由模拟器）
**前置要求：** 第 17 阶段 · 01（托管 LLM 平台），第 17 阶段 · 16（模型路由）
**所需时间：** 约 60 分钟

## 学习目标

- 列举网关的六大核心功能（路由、故障转移、重试、速率限制、密钥管理、可观测性、护栏）。
- 将 2026 年的四大网关（LiteLLM、Portkey、Kong AI、Bifrost）映射到各自的规模上限和适用场景。
- 引用 Kong 的基准测试数据（比 Portkey 快 228%，比 LiteLLM 快 859%），并解释其对 >500 RPS 场景的意义。
- 根据数据驻留和运维预算，在自托管与托管方案之间做出选择。

## 问题所在

你的产品需要调用 OpenAI、Anthropic 和一个自托管的 Llama。每个提供商都有不同的 SDK、错误模型、速率限制和认证方式。你需要故障转移能力（如果 OpenAI 返回 429，尝试 Anthropic）、统一的凭证存储、统一的可观测性以及按租户的速率限制。

在应用层重复造轮子会导致每个服务都与每个提供商耦合。网关层将其整合到一个进程中，通过一个 API（通常是 OpenAI 兼容的）来分发到各个提供商。

## 概念说明

### 六大核心功能

1. **提供商路由** —— 将 OpenAI、Anthropic、Gemini、自托管等统一在一个 API 背后。
2. **故障转移** —— 当遇到 429、5xx 或质量失败时，重试其他提供商。
3. **重试** —— 指数退避，有界尝试次数。
4. **速率限制** —— 按租户、按密钥、按模型。
5. **密钥引用** —— 运行时从密钥库拉取凭证（绝不放在应用中）。
6. **可观测性** —— OTel + GenAI 属性（第 17 阶段 · 13）+ 成本归因。
7. **护栏** —— PII 脱敏、越狱检测、允许话题过滤。

### LiteLLM —— MIT 开源，Python 实现

- 支持 100+ 提供商，兼容 OpenAI 接口，路由配置，故障转移，基础可观测性。
- 在 Kong 的基准测试中约 2000 RPS 时出现瓶颈；8 GB 内存占用，持续负载下出现级联故障。
- 最适合：Python 应用、<500 RPS、开发/测试环境网关、实验性路由。
- 成本：开源版免费；有云免费版可用。

### Portkey —— 控制平面定位

- 2026 年 3 月起采用 Apache 2.0 开源协议。支持护栏、PII 脱敏、越狱检测、审计追踪。
- 每请求增加 20-40 毫秒延迟开销。
- 生产版每月 49 美元，包含数据保留和 SLA。
- 最适合：需要护栏 + 可观测性一体化的受监管行业。

### Kong AI Gateway —— 规模化方案

- 基于 Kong Gateway 构建（成熟的 API 网关产品，lua+OpenResty）。
- Kong 自身在 12 CPU 等效环境的基准测试中：比 Portkey 快 228%，比 LiteLLM 快 859%。
- 定价：每模型每月 100 美元，Plus 版最多 5 个模型。
- 最适合：已在使用 Kong；>1000 RPS；愿意付费授权。

### Bifrost（Maxim AI）

- 支持可配置退避策略的自动重试。
- 当 OpenAI 返回 429 时自动回退到 Anthropic 是其典型用例。
- 较新的参与者；商业化产品。

### Cloudflare AI Gateway / Vercel AI Gateway

- 托管式，零运维。基础重试和可观测性。
- 最适合：在 Cloudflare/Vercel 上运行的边缘 JavaScript 应用。
- 在护栏和速率限制方面不如 Kong/Portkey。

### 自托管 vs 托管

数据驻留是关键驱动因素。医疗和金融行业默认自托管（LiteLLM 或 Portkey 开源版或 Kong）。消费类产品默认托管（Cloudflare AI Gateway）或中间层（Portkey 托管版）。混合方案：受监管租户使用自托管，其他租户使用托管。

### 延迟预算

- LiteLLM：典型开销 5-15 毫秒。
- Portkey：开销 20-40 毫秒。
- Kong：开销 3-8 毫秒。
- Cloudflare/Vercel：开销 1-3 毫秒（边缘优势）。

网关延迟直接增加 TTFT。对于 TTFT P99 < 100 毫秒的 SLA，选择 Kong 或 Cloudflare。对于 P99 < 500 毫秒，任何方案都可行。

### 速率限制语义很重要

简单的令牌桶算法在中等规模下有效。多租户需要滑动窗口 + 突发允许 + 按租户分层。LiteLLM 提供令牌桶；Kong 提供滑动窗口；Portkey 提供分层方案。

### 网关 + 可观测性 + 路由的组合

第 17 阶段 · 13（可观测性）+ 16（模型路由）+ 19（网关）在生产中是同一层。选择一个覆盖全部三个功能的工具，或者仔细组合：大多数 2026 年的部署将 Helicone（可观测性）或 Portkey（护栏）与 Kong（规模化）组合使用，实现职责分离。

### 需要记住的关键数字

- LiteLLM：约 2000 RPS 时出现瓶颈，8 GB 内存。
- Portkey：开销 20-40 毫秒；2026 年 3 月起采用 Apache 2.0。
- Kong：比 Portkey 快 228%，比 LiteLLM 快 859%。
- Kong 定价：每模型每月 100 美元，Plus 版最多 5 个模型。
- Cloudflare/Vercel：边缘开销 1-3 毫秒。

## 开始使用

`code/main.py` 模拟网关路由，在 429/5xx 故障注入下实现 3 个提供商之间的故障转移。报告延迟、重试率和故障转移命中率。

## 部署产出

本课程生成 `outputs/skill-gateway-picker.md`。给定规模、运维模式、合规要求和延迟预算，推荐合适的网关。

## 练习

1. 运行 `code/main.py`。配置从 OpenAI→Anthropic→自托管的故障转移链。在 5% 提供商错误率下，预期命中率是多少？
2. 你的 SLA 是 TTFT P99 < 200 毫秒，基线延迟 300 毫秒。哪些网关能满足预算？
3. 一个医疗客户要求自托管 + PII 脱敏 + 审计。选择 Portkey 开源版还是 Kong？
4. 比较 LiteLLM 和 Kong：在什么 RPS 上限时团队应该迁移？
5. 为多租户 SaaS 设计速率限制策略：免费版、试用版、付费版。使用令牌桶还是滑动窗口？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Gateway | "API 代理" | 位于应用和提供商之间的进程 |
| LiteLLM | "那个 MIT 开源的" | Python 开源实现，100+ 提供商，2K RPS 时出现瓶颈 |
| Portkey | "护栏网关" | 控制平面 + 可观测性，Apache 2.0 |
| Kong AI Gateway | "那个做规模的" | 基于 Kong Gateway 构建，基准测试领先者 |
| Bifrost | "Maxim 的网关" | 重试 + Anthropic 回退方案 |
| Cloudflare AI Gateway | "边缘托管的" | 边缘部署的托管网关，零运维 |
| PII 脱敏 | "数据清洗" | 发送到模型前使用正则 + NER 进行掩码处理 |
| 越狱检测 | "提示词注入防护" | 对用户输入进行分类器检测 |
| 审计追踪 | "合规日志" | 每次 LLM 调用的不可篡改记录 |
| 令牌桶 | "简单限流" | 基于补充的速率限制器 |
| 滑动窗口 | "精确限流" | 基于时间窗口的速率限制器；公平性更好 |

## 延伸阅读

- [Kong AI Gateway 基准测试](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry —— 2026 AI 网关对比](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy —— 2026 年最佳 LLM 网关工具](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway 文档](https://docs.konghq.com/gateway/latest/ai-gateway/)
