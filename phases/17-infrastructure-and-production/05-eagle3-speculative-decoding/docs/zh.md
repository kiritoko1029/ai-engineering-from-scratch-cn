# EAGLE-3 推测解码在生产中的应用

> 推测解码将快速草稿模型与目标模型配对。草稿提出 K 个 token；目标在单次前向传播中验证；被接受的 token 是免费的。2026 年，EAGLE-3 是生产级变体 — 它在目标模型的隐藏状态上训练草稿头，而非原始 token 上，将接受率 alpha 推入通用聊天的 0.6-0.8 区间。正确的问题不是"草稿有多快"，而是"在我的流量上 alpha 是多少？"如果 alpha 降到约 0.55 以下，推测解码在高并发下是净负面的，因为每个被拒绝的草稿都会消耗一次目标前向传播。本课教你先测量 alpha，再打开标志。

**类型：** 学习
**语言：** Python（标准库，简易接受率模拟器）
**前置要求：** Phase 17 · 04（vLLM 服务内部机制），Phase 10 · 18（多 token 预测）
**所需时间：** 约60分钟

## 学习目标

- 说出推测解码的三代演进，解释 EAGLE-3 相对于 EAGLE-2 和经典草稿模型的变化。
- 定义接受率 alpha，根据 alpha 和 K（草稿长度）计算预期加速，并识别目标并发下的盈亏平衡 alpha。
- 解释为什么推测解码在 vLLM 2026 中是可选的（非默认），以及为什么在不测量 alpha 的情况下开启它是一种生产反模式。
- 编写一个测量方案：使用哪个基准测试、哪种提示分布、哪个并发点、以哪个指标为门槛。

## 问题所在

解码是内存受限的。在运行 Llama 3.3 70B FP8 的 H100 上，每个解码 token 读取约 140 GB/s 的权重并发出一个 token。解码期间 GPU 计算几乎空闲 — 瓶颈是 HBM 带宽，而非矩阵乘法吞吐量。

推测解码利用了这一差距。用廉价的草稿模型生成 K 个候选 token，然后让目标模型在单次前向传播中验证所有 K 个。每个被验证的 token 实际上是免费的（摊销到目标本来就必须执行的 K 批前向传播中）。

经典草稿模型方法使用同系列的较小模型（Llama 3.2 1B 为 Llama 3.3 70B 做草稿）。这可行但接受率平庸 — 小模型分布偏离目标。EAGLE，然后 EAGLE-2，然后 EAGLE-3 直接在目标模型的内部状态上训练轻量草稿头，因此草稿的分布更紧密地跟踪目标。这就是为什么 alpha 从草稿模型的 0.4 提升到 EAGLE-3 的 0.6-0.8。

关键点：EAGLE-3 在 vLLM 2026 中是可选的。`speculative_config` 必须显式设置。没有标志，就没有加速。那些不测量真实流量上的 alpha 就开启它的团队，往往会看到尾部延迟变差，而非变好。

## 概念说明

### 推测解码实际带来了什么

没有推测解码，每个 token 的成本是一次目标前向传播。使用推测解码，草稿长度为 K、接受率为 alpha 时，每次目标前向传播的预期 token 数为 `1 + K * alpha`。加速比为 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是草稿加验证的开销。对于 K=5、alpha=0.7：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1 倍`。实际数字集中在 2-3 倍，因为 alpha 在生产流量上很少那么高，且 epsilon 在高批次大小时增长。

### 为什么 alpha 是唯一重要的指标

被拒绝的 token 不会消失 — 它们强制对第一个被拒绝的 token 进行第二次目标前向传播。在 alpha 降到 0.4 的工作负载上，你支付草稿开销加验证加重新选择。在高并发（比如 256 并发）下，解码批次已经足够大，"单独目标"和"带验证的目标"之间的内存带宽差距缩小。在大多数 2026 年硬件上，alpha 低于 0.55 时，推测解码是净负面的。

Alpha 因工作负载而异。在 ShareGPT 风格的通用聊天上，基于 ShareGPT 训练的 EAGLE-3 达到 0.6-0.8。在特定领域流量（代码、医疗、法律）上，基于通用数据训练的草稿头降到 0.4-0.6。训练领域特定的草稿头可以恢复 alpha — 与目标微调相比，这是一项轻量、快速的训练任务。

### EAGLE 代际概览

- **经典草稿模型**：同系列小模型。Alpha 0.3-0.5。基础设施简单 — 加载两个模型，草稿每次目标前向传播运行 K 次前向传播。
- **EAGLE-1（2024）**：在目标隐藏状态（最后一层）上训练的单一草稿头。Alpha 约 0.5-0.6。在目标之上有少量参数开销。
- **EAGLE-2（2025）**：自适应草稿长度和树状草稿（在一次目标传播中验证多个分支）。Alpha 约 0.6-0.7。更复杂的草稿调度器。
- **EAGLE-3（2025-2026）**：在目标多层（不仅仅是最后一层）上训练的草稿头，对齐更好。通用聊天上 Alpha 约 0.6-0.8。

### 2026 年生产配方

1. 部署纯目标模型。在目标并发下测量基线 TTFT、ITL、吞吐量。
2. 通过 vLLM `speculative_config` 启用 EAGLE-3 草稿。重新运行基准测试。
3. 记录接受率 alpha。vLLM V1 报告为 `spec_decode_metrics.accepted_tokens_per_request`。除以请求的草稿长度得到 alpha。
4. 如果在生产流量分布上 alpha < 0.55，禁用推测解码或训练领域特定的 EAGLE-3 草稿。
5. 在生产并发下重新运行。确认 P99 ITL 没有变差。

### 生产陷阱：P99 尾部

推测解码降低平均 ITL。如果不调优，P99 可能变差。被拒绝的草稿触发两次传播序列（草稿 + 验证失败 + 重新选择）。在满批次下，这两次传播是串行的。关注 P99 ITL，而非 P50。

### EAGLE-3 已部署的场景

Google 在 2025 年的 AI Overviews 中部署了推测解码（相同质量，更快响应）。vLLM V1 以 `speculative_config` 作为文档化接口；V1 中的 N-gram GPU 推测解码是与 Chunked Prefill 兼容的变体。SGLang 支持 EAGLE-3 作为前缀密集工作负载的推荐草稿路径。

### 盈亏平衡的数学公式

预期加速：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。设 `S = 1` 求解 alpha：`alpha_breakeven = verify_overhead / K`。对于典型的 verify_overhead 约 0.15 和 K=5：`alpha_breakeven = 0.03`。但这是原始解码数学。在高并发下，验证开销上升且解码批次已在序列间摊销内存读取，因此实际有效 alpha_breakeven 上升到约 0.45-0.55。

### 何时不使用推测解码

- 延迟不重要的 Batch-1 离线生成。使用纯目标。
- 非常短的输出（低于 50 个 token）。草稿开销和验证成本占主导。
- 没有领域训练草稿头的专业领域。Alpha 太低。
- vLLM v0.18.0 加草稿模型推测解码加 `--enable-chunked-prefill`。此组合无法编译。文档中记录的例外是 V1 中的 N-gram GPU 推测解码。

## 开始使用

`code/main.py` 在一系列 alpha 值和草稿长度 K 下模拟有无推测解码的解码循环。它打印盈亏平衡 alpha、测量的加速比和尾部行为。在多个 (alpha, K) 组合上运行它，准确看到推测解码在何时不再划算。

## 交付成果

本课生成 `outputs/skill-eagle3-rollout.md`。给定目标模型、流量分布描述和并发目标，生成分阶段 EAGLE-3 上线方案 — 基准基线、启用配置、测量 alpha、以 alpha >= 0.55 为门槛、关注 P99 ITL。

## 练习

1. 运行 `code/main.py`。在 K=5 时，2 倍加速需要多少 alpha？3 倍加速需要多少？这对 verify_overhead 的敏感度如何？
2. 假设生产流量 70% 为通用聊天，30% 为代码。通用聊天使用基于 ShareGPT 训练的 EAGLE-3 达到 alpha 0.7；代码达到 alpha 0.4。混合 alpha 是多少，推测解码是否净正面？
3. 阅读 vLLM `speculative_config` 文档。说出三种模式（草稿模型、EAGLE、N-gram），以及哪种与 Chunked Prefill 兼容。
4. 启用 EAGLE-3 后平均 ITL 下降 25%，但 P99 ITL 上升了 15%。诊断并提出缓解方案。
5. 计算 Llama 3.3 70B 的 EAGLE-3 草稿头的内存成本。与运行 Llama 3.2 1B 作为经典草稿相比如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 推测解码 | "草稿加验证" | 用廉价模型提出 K 个 token，在一次目标前向传播中验证所有 K 个 |
| 接受率 alpha | "推测接受率" | 草稿 token 被目标接受的比例；唯一重要的指标 |
| 草稿长度 K | "推测 k" | 每次目标前向传播草稿提出多少 token；典型 4-8 |
| 验证开销 epsilon | "推测开销" | 验证加重新选择相比纯目标前向传播的额外成本；随批次增长 |
| EAGLE-3 | "最新 EAGLE" | 2025-2026 变体；在目标多层上训练草稿头；通用聊天 alpha 0.6-0.8 |
| `speculative_config` | "vLLM 推测配置" | vLLM V1 中的显式可选开关；无默认意味着无加速 |
| N-gram 推测解码 | "N-gram 草稿" | 使用提示中 N-gram 查找的 GPU 端草稿；Chunked Prefill 兼容 |
| 盈亏平衡 alpha | "无效 alpha" | 推测解码给出零加速的 alpha；在生产并发下关注此值 |
| 被拒草稿两次传播 | "重新选择成本" | 草稿被拒时的两次目标前向传播；驱动 P99 尾部 |

## 延伸阅读

- [vLLM — 推测解码文档](https://docs.vllm.ai/en/latest/features/spec_decode/) — V1 中 `speculative_config` 和 Chunked Prefill 兼容性的权威来源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — 精确的字段集。
- [EAGLE 论文（arXiv:2401.15077）](https://arxiv.org/abs/2401.15077) — 原始 EAGLE 草稿头公式。
- [EAGLE-2 论文（arXiv:2406.16858）](https://arxiv.org/abs/2406.16858) — 自适应草稿和树状结构。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — 使用推测解码的高效 LLM 系统。
- [BentoML — 推测解码](https://bentoml.com/llm/inference-optimization/speculative-decoding) — 生产上线检查清单。
