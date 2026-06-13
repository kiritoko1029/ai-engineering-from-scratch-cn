# LLM API 负载测试——为什么 k6 和 Locust 会骗你

> 传统负载测试工具并非为流式响应、可变输出长度、token 级别指标或 GPU 饱和场景而设计。大多数团队会踩到两个陷阱。GIL 陷阱：Locust 的 token 级别测量在 Python GIL 下运行分词，这会与高并发下的请求生成竞争；分词积压随后膨胀了报告的 token 间延迟——瓶颈在你的客户端，而不在服务端。提示词一致性陷阱：循环测试中使用相同提示词，只测了 token 分布上的一个点；真实流量具有可变长度和多样化的前缀匹配。LLMPerf 通过 `--mean-input-tokens` + `--stddev-input-tokens` 解决了这个问题。2026 年工具映射：LLM 专用工具（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）用于 token 级别精度；**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**——流式感知、Kubernetes 原生分布式测试，通过 TestRun/PrivateLoadZone CRDs 实现，最适合 CI/CD 门禁；Vegeta 用于 Go 恒定速率饱和测试；Locust 2.43.3 仅在配合 LLM-Locust 扩展时才支持流式测试。负载模式：稳态、爬坡、尖峰（自动扩缩容测试）、浸泡（内存泄漏）。

**类型：** 构建
**语言：** Python（标准库，简易真实提示词生成器 + 延迟收集器）
**前置要求：** Phase 17 · 08（推理指标），Phase 17 · 03（GPU 自动扩缩容）
**所需时间：** 约 75 分钟

## 学习目标

- 解释两种反模式（GIL 陷阱、提示词一致性陷阱），它们使通用负载测试工具对 LLM API 产生误导结果。
- 针对特定目标选择工具：LLMPerf（基准测试）、k6 + 流式扩展（CI 门禁）、guidellm（大规模合成测试）、GenAI-Perf（NVIDIA 参考实现）。
- 设计四种负载模式（稳态、爬坡、尖峰、浸泡），并说出每种模式捕获的故障类型。
- 使用输入 token 的均值 + 标准差构建真实提示词分布，而非使用固定长度。

## 问题所在

你用 k6 测试了 LLM 端点，500 并发用户。端点扛住了。你上了生产环境。实际 200 个真实用户时服务崩溃了——P99 TTFT 爆涨，GPU 打满。

发生了两件事。第一，k6 发送了 500 个相同的提示词——你的请求合并和前缀缓存让你以为自己在处理 500 个并发解码，实际上只在处理一个。第二，k6 不会以人眼体验的方式跟踪流式响应的 token 间延迟；它看到的是一个 HTTP 连接，而非 500 个以不同间隔到达的 token。

LLM 的负载测试是一门独立的学科。

## 概念说明

### GIL 陷阱（Locust）

Locust 使用 Python，在客户端的 GIL 下运行分词。高并发时，分词器排在请求生成之后。报告的 token 间延迟包含了客户端的分词积压。你以为服务端慢了；实际上是测试工具的问题。

解决方案：LLM-Locust 扩展将分词移到独立进程，或使用编译型语言的测试工具（k6、使用 tokenizers.rs 的 LLMPerf）。

### 提示词一致性陷阱

所有已知的负载测试工具都让你配置一个提示词。在 10,000 次迭代的循环测试中，每次都发送完全相同的提示词。服务端每次都看到相同的前缀——前缀缓存命中率接近 100%，吞吐量看起来很好。

解决方案：从提示词分布中采样。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`——多样化的长度，多样化的内容。

### 四种负载模式

1. **稳态**——恒定 RPS 持续 30-60 分钟。捕获：基线性能回归。
2. **爬坡**——在 15 分钟内将 RPS 从 0 线性增加到目标值。捕获：容量断点、预热异常。
3. **尖峰**——RPS 突然提升 3-10 倍持续 2 分钟后恢复。捕获：自动扩缩容延迟、队列饱和、冷启动影响。
4. **浸泡**——稳态运行 4-8 小时。捕获：内存泄漏、连接池漂移、可观测性溢出。

### 2026 年工具映射

**LLMPerf**（Anyscale）——Python 但 Rust 支持的分词。均值/标准差提示词。流式感知。性能测试的最佳默认选择。

**NVIDIA GenAI-Perf**——NVIDIA 的参考实现。使用 Triton 客户端；全面的指标覆盖。注意其 ITL 不包含 TTFT；LLMPerf 的 ITL 包含 TTFT。两个工具对同一服务器产生不同的 TPOT。

**LLM-Locust**（TrueFoundry）——修复 GIL 陷阱的 Locust 扩展。熟悉的 Locust DSL + 流式指标。

**guidellm**——大规模合成基准测试。

**k6 v2026.1.0** + **k6 Operator 1.0 GA（2025 年 9 月）**：
- k6 本身（Go、编译型、无 GIL）增加了流式感知指标。
- k6 Operator 使用 TestRun / PrivateLoadZone CRDs 实现 Kubernetes 原生分布式测试。
- 最适合 CI/CD 门禁和 SLA 测试。

**Vegeta**——Go，比 k6 简单。恒定 HTTP 速率饱和测试。不感知 LLM，但适合网关/限速测试。

**Locust 2.43.3 原版**——对 LLM 存在 GIL 陷阱。仅在配合 LLM-Locust 扩展时使用。

### CI 中的 SLA 门禁

在 PR 上运行 k6：

- 在基线 RPS 下各运行 30-50 次迭代。
- 门禁条件：P50/P95 TTFT、5xx < 5%、TPOT 低于阈值。
- 超出则构建失败。

### 真实提示词分布

从真实流量样本（如果你有的话）或公开分布（例如用于聊天的 ShareGPT 提示词、用于代码的 HumanEval）构建。将均值 + 标准差传给 LLMPerf。务必避免使用单一提示词的循环测试。

### 你应该记住的数字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：流式感知指标。
- 典型 LLMPerf 运行：在并发 X 下发送 100-1000 个请求。
- 典型 CI 门禁：每个 PR 30-50 次迭代。
- 四种模式：稳态（steady）、爬坡（ramp）、尖峰（spike）、浸泡（soak）。

## 开始使用

`code/main.py` 模拟具有真实提示词分布的负载测试，测量有效 TPOT，并演示统一提示词陷阱。

## 部署产出

本课产出 `outputs/skill-load-test-plan.md`。给定工作负载和 SLA，选择工具并设计四种负载模式。

## 练习

1. 运行 `code/main.py`。对比统一分布与真实分布——差异在哪里？
2. 编写 CI 门禁的 k6 脚本：100 并发下 TTFT P95 < 800 ms，运行 5 分钟。
3. 你的浸泡测试显示内存每小时增长 50 MB。说出三个原因及用于区分它们的检测手段。
4. 尖峰测试从 10 RPS 到 100 RPS。如果 Karpenter + vLLM 生产栈已部署（Phase 17 · 03 + 18），预期恢复时间是多少？
5. GenAI-Perf 报告 TPOT=6ms；LLMPerf 对同一服务器报告 TPOT=11ms。解释原因。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| LLMPerf | "LLM 测试工具" | Anyscale 基准测试工具，流式感知 |
| GenAI-Perf | "NVIDIA 工具" | NVIDIA 参考测试工具 |
| LLM-Locust | "LLM 版 Locust" | 修复 GIL 陷阱的 Locust 扩展 |
| guidellm | "合成基准测试" | 大规模合成测试工具 |
| k6 Operator | "K8s 版 k6" | 基于 CRD 的分布式 k6 |
| GIL 陷阱 | "Python 客户端开销" | 分词积压膨胀报告延迟 |
| 提示词一致性陷阱 | "单提示词骗局" | 使用相同提示词的循环命中缓存，膨胀吞吐量 |
| 稳态 | "恒定负载" | 持续 N 分钟的平稳 RPS |
| 爬坡 | "线性上升" | 在持续时间内从 0 到目标值 |
| 尖峰 | "突发测试" | 突然倍增后恢复 |
| 浸泡 | "长时间测试" | 运行数小时以检测泄漏 |

## 延伸阅读

- [TianPan — LLM 应用负载测试](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — 2026 年 LLM 负载测试](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — LLM 推理基准测试入门](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
