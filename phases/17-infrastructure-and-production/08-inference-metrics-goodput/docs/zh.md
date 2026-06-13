# 推理指标 — TTFT、TPOT、ITL、Goodput、P99

> 四个指标决定推理部署是否正常工作。TTFT 是预填充加队列加网络。TPOT（等价于 ITL）是每 token 的内存受限解码成本。端到端延迟是 TTFT 加 TPOT 乘以输出长度。吞吐量是整个集群聚合的每秒 token 数。但对产品真正重要的是 goodput — 同时满足所有 SLO 的请求比例。高吞吐量但低 goodput 意味着你正在处理永远不会及时到达用户的 token。2026 年 TRT-LLM 上 Llama-3.1-8B-Instruct 的参考数字：平均 TTFT 162 ms，平均 TPOT 7.33 ms，平均 E2E 1,093 ms。始终报告 P50、P90、P99 — 不要只报告平均值。注意测量陷阱：GenAI-Perf 从 ITL 计算中排除 TTFT，LLMPerf 包含它；两个工具对同一次运行的 TPOT 报告不一致。

**类型：** 学习
**语言：** Python（标准库，简易百分位计算器和 goodput 报告器）
**前置要求：** Phase 17 · 04（vLLM 服务内部机制）
**所需时间：** 约60分钟

## 学习目标

- 精确定义 TTFT、TPOT、ITL、E2E、吞吐量和 goodput，并说出每个指标测量的组件。
- 解释为什么平均值是 LLM 服务的错误统计量，以及如何阅读 P50/P90/P99。
- 构造 SLO 多约束（例如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s），并计算 goodput。
- 说出两个对同一次运行的 TPOT 报告不一致的基准测试工具，并解释原因。

## 问题所在

"我们的吞吐量是每秒 15,000 个 token。"那又怎样？如果 40% 的请求端到端超过 2 秒，用户就放弃了。仅靠吞吐量无法告诉你产品是否正常工作。

推理有多个延迟轴，每个轴的故障方式不同。预填充是计算密集型的，随提示长度增长。解码是内存受限的，随批次大小增长。队列延迟是运维问题。网络是物理距离问题。你需要为每个轴定义不同的指标，需要百分位数，还需要一个单一的综合指标说"用户是否得到了他们预期的" — 这就是 goodput。

## 概念说明

### TTFT — 首 token 时间

`TTFT = queue_time + network_request + prefill_time`

提示较长时预填充占主导。在 H100 上的 Llama-3.3-70B FP8 中，32k 提示需要约 800 ms 的纯预填充。队列时间是负载下的调度器行为。网络请求是包含 TLS 的传输时间。TTFT 是用户在任何流式内容返回之前看到的延迟。

### TPOT / ITL — token 间延迟

同一个量有多个名称。`TPOT`（每输出 token 时间）、`ITL`（token 间延迟）、`每 token 解码延迟` — 全都一样。它是第一个 token 之后连续流式 token 之间的时间。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同一 Llama-3.3-70B H100 技术栈上使用 Chunked Prefill，平均 TPOT 约 7 ms。不使用 Chunked Prefill 时，相邻序列进行长预填充期间，TPOT 可能飙升到 50 ms。关注 P99，而非平均值。

### E2E 延迟

`E2E = TTFT + TPOT * output_tokens + network_response`

对于长输出（>500 个 token），E2E 由 TPOT 主导。对于短输出加长提示，E2E 由 TTFT 主导。报告按输出长度条件化的 E2E。

### 吞吐量

`throughput = total_output_tokens / elapsed_time`

聚合指标。告诉你集群效率。不能告诉你单个请求的健康状况。

### Goodput — 你真正关心的指标

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是多约束。一个请求只有在所有约束都满足时才算"好"。Goodput 就是这个比例。高吞吐量但 60% goodput 是失败。较低吞吐量但 99% goodput 才是目标。

2026 年，goodput 是 MLPerf Inference v6.0 提交和 AI 平台提供商内部 SLA 跟踪中使用的指标。

### 为什么平均值是错误的统计量

LLM 延迟分布是右偏的。一个包含长预填充邻居的解码批次可能有 500 个 token 的 TPOT 约 7 ms，20 个 token 的 TPOT 约 60 ms。平均 TPOT 是 9 ms。P99 TPOT 是 65 ms。用户经常遇到 P99 — 这就是他们离开的原因。

始终报告三元组（P50、P90、P99）。对于用户体验，P99 是你要优化的。

### 参考数字 — 2026 年 TRT-LLM 上的 Llama-3.1-8B-Instruct

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：根据 Chunked Prefill 配置在 10-25 ms 之间变化。

这些是 NVIDIA 发布的参考点。它们随模型大小（70B 会显示 3-5 倍）、硬件（H100 vs B200 约 3 倍）和负载而变化。

### 测量陷阱

2026 年两个最常用的基准测试工具对同一次运行的 TPOT 报告不一致：

- **NVIDIA GenAI-Perf**：从 ITL 计算中排除 TTFT。ITL 从第 2 个 token 开始。
- **LLMPerf**：包含 TTFT。ITL 从第 1 个 token 开始。

对于一个 TTFT 500 ms、100 个输出 token、总解码时间 700 ms 的请求，GenAI-Perf 报告 `ITL = 700/99 = 7.07 ms`，LLMPerf 报告 `ITL = 1200/100 = 12.00 ms`。工具选择改变了数字。

始终说明使用了哪个工具。始终发布定义。

### 构造 SLO

2026 年面向消费者的 70B 聊天模型的合理 SLO：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- E2E P99 <= 3 s（<300 个输出 token）。
- Goodput 目标 >= 99%。

企业 SLO 收紧 TTFT（200-400 ms）并放宽 E2E。关键是写下来、测量所有三个指标，并将 goodput 作为单一综合指标跟踪。

### 如何测量

- 运行真实流量或真实合成流量（LLMPerf 使用 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- 基准测试运行目标为 2 倍峰值并发。
- 运行 30-50 次迭代，取合并样本的百分位数。
- 发布时附带工具名称、工具版本、模型、硬件、并发、提示分布。

```figure
throughput-latency
```

## 开始使用

`code/main.py` 是一个简易 goodput 计算器。生成合成延迟分布，应用 SLO，并计算 goodput。还展示了 GenAI-Perf vs LLMPerf 在同一轨迹上的 TPOT 差异。

## 交付成果

本课生成 `outputs/skill-slo-goodput-gate.md`。给定工作负载和 SLO，生成一个 CI/CD 就绪的基准测试方案，以 goodput 而非吞吐量作为部署门槛。

## 练习

1. 运行 `code/main.py`。生成一个 1% 尾部尖峰的分布。当 P99 TPOT 从 30 ms 收紧到 15 ms 时，goodput 如何变化？
2. 一个供应商报价"Llama 3.3 70B H100 上 15,000 tok/s"。在信任之前，说出三个要问的问题。
3. 为什么 Chunked Prefill 保护 P99 TPOT 而非平均 TPOT？
4. 为语音助手（首 token 是听到的，而非看到的）构造一个消费者 SLO。哪个指标对用户最可见？
5. 阅读 LLMPerf README 和 GenAI-Perf 文档。找出三个其他工具不一致的指标。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| TTFT | "首 token 时间" | 队列 + 网络 + 预填充；长提示时由预填充主导 |
| TPOT | "每输出 token 时间" | 第一个 token 之后的每 token 内存受限解码成本 |
| ITL | "token 间延迟" | 大多数工具中与 TPOT 相同（并非全部 — 见 GenAI-Perf） |
| E2E | "端到端" | TTFT + TPOT * output_len；上面还有响应端网络 |
| 吞吐量 | "tok/s" | 集群效率；没有延迟百分位数时毫无用处 |
| Goodput | "SLO 达标率" | 同时满足每个 SLO 约束的请求比例 |
| P99 | "尾部" | 100 次中最差的延迟；用户体验指标 |
| SLO 多约束 | "联合约束" | 三个延迟上限的 AND；任何一个被违反则请求失败 |
| GenAI-Perf vs LLMPerf | "工具陷阱" | 工具对 ITL 是否包含 TTFT 意见不一致 |

## 延伸阅读

- [NVIDIA NIM — LLM 基准测试指标](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html) — TTFT、ITL、TPOT 的权威定义。
- [Anyscale — LLM 服务基准测试指标](https://docs.anyscale.com/llm/serving/benchmarking/metrics) — 替代定义和测量方案。
- [BentoML — LLM 推理指标](https://bentoml.com/llm/inference-optimization/llm-inference-metrics) — 在真实部署上的应用测量。
- [LLMPerf](https://github.com/ray-project/llmperf) — 基于 Ray 的开源基准测试。
- [GenAI-Perf](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/client/src/c++/perf_analyzer/genai-perf/README.html) — NVIDIA 的基准测试工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 行业认可的基于 goodput 的基准测试。
