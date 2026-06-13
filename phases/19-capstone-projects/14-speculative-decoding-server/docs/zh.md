# 毕业项目 14 — 投机解码推理服务器

> vLLM 0.7 中的 EAGLE-3 在真实流量上提供 2.5-3 倍吞吐量。P-EAGLE（AWS 2026）进一步推动了并行投机。SGLang 的 SpecForge 大规模训练草稿头。Red Hat 的 Speculators hub 发布了常见开放模型的对齐草稿。TensorRT-LLM 在 NVIDIA 上将投机解码设为一等公民。2026 年的生产服务技术栈是 vLLM 或 SGLang + EAGLE 系列草稿、FP8 或 INT4 量化，以及基于队列等待的 HPA。本毕业项目是在两个开放模型上以 2.5 倍以上基线吞吐量提供服务，并附带完整的尾延迟报告。

**类型：** 毕业项目
**语言：** Python（服务）、C++ / CUDA（内核检查）、YAML（配置）
**前置要求：** 阶段 3（深度学习）、阶段 7（Transformer）、阶段 10（从零构建 LLM）、阶段 17（基础设施）
**涉及阶段：** P3 · P7 · P10 · P17
**所需时间：** 30 小时

## 问题所在

投机解码在 2026 年成为标配。EAGLE-3 草稿头在目标模型的隐藏状态上训练，预测 N 个 Token；目标模型一次验证。60-80% 的接受率转化为 2-3 倍的端到端吞吐量。vLLM 0.7 原生集成。SGLang + SpecForge 提供训练流水线。Red Hat 的 Speculators 发布了 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的对齐草稿。

技艺在于服务运维，而非模型。接受率随流量分布漂移（ShareGPT vs 代码 vs 领域数据）。拒绝时的尾延迟比无投机更差——你必须报告多个批量大小的 p99，而非仅稳态 Token/秒。与 Anthropic / OpenAI API 对比的每百万 Token 成本是可信度杠杆。

## 概念说明

投机解码有两层。**草稿**模型（EAGLE-3 头、ngram 或更小的目标对齐模型）每步提议 k 个候选 Token。**目标**模型一次验证所有 k 个；任何被接受的前缀替换贪心路径。接受率取决于草稿—目标对齐和输入分布。

EAGLE-3 在大多数流量上优于 ngram 草稿。P-EAGLE 运行并行投机以获得更深的草稿树。权衡：拒绝时的 P99 延迟更高，因为验证遍更大。服务配置必须报告批量大小分桶的延迟以暴露此问题。

部署是 Kubernetes。vLLM 0.7 每 GPU 或张量并行分片运行一个副本。HPA 在队列等待而非 CPU 上自动扩展。FP8（Marlin）和 INT4（AWQ）量化将 GPU 内存控制在 H100 / H200 范围内。端到端报告是吞吐量、接受率、批量 1/8/32 的 p50/p99，以及 $/百万 Token。

## 架构

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技术栈

- 服务：vLLM 0.7 或 SGLang 0.4
- 投机方法：EAGLE-3 草稿头、P-EAGLE 并行投机、ngram 备选
- 草稿训练：SpecForge（SGLang）或 Red Hat Speculators
- 目标模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8（Marlin）、INT4 AWQ
- 部署：Kubernetes + NVIDIA device plugin；基于队列等待指标的 HPA
- 评估：ShareGPT、MT-Bench-v2、GSM8K、HumanEval 用于领域分布接受率衡量
- 参考：TensorRT-LLM 投机解码作为厂商基线

## 开始构建

1. **目标模型准备。** 选择 Llama 3.3 70B。通过 Marlin 量化到 FP8。在 1xH100（或 2x 张量并行）上使用 vLLM 0.7 部署。

2. **草稿源。** 从 Red Hat Speculators 拉取对齐的 EAGLE-3 草稿头（或通过 SpecForge 训练一个）。加载到 vLLM 的投机解码配置中。

3. **基线数字。** 投机前：批量 1/8/32 的 Token/s、p50/p99 延迟、GPU 利用率。发布。

4. **启用 EAGLE-3。** 翻转配置；重跑同一基准。报告加速、接受率、p99 尾延迟差异。

5. **P-EAGLE。** 启用并行投机；衡量更深草稿树 vs 串行 EAGLE-3。报告 P-EAGLE 有帮助 vs 有害的拐点。

6. **领域流量。** 通过同一服务器运行 ShareGPT vs HumanEval vs 领域特定流量。衡量每分布的接受率。识别草稿漂移时机。

7. **第二个目标模型。** 在 Qwen3-Coder-30B MoE 上运行同一流水线。草稿更棘手（MoE 路由噪声）。报告。

8. **K8s HPA。** 在 K8s 下部署，HPA 追踪 `queue_wait_ms`。当负载增加三倍时演示扩展。

9. **成本对比。** 计算 $/百万 Token 与 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 在同一评估上的对比。发布。

## 使用示例

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付成果

`outputs/skill-inference-server.md` 描述了交付成果。带投机解码的可衡量服务技术栈、完整基准报告和 K8s 部署。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 与基线的衡量加速 | 两个模型在匹配质量上 2.5 倍以上吞吐量 |
| 20 | 真实流量上的接受率 | 每分布接受率报告 |
| 20 | P99 尾延迟纪律 | 批量 1/8/32 有无投机的 p99 |
| 20 | 运维 | K8s 部署、基于队列等待的 HPA、平滑发布 |
| 15 | 报告和方法论 | 清晰解释什么变了以及为什么 |
| **100** | | |

## 练习

1. 衡量草稿落后目标一个版本时（例如 Llama 3.3 -> 3.4 漂移）接受率的退化。构建监控告警。

2. 实现 ngram 备选：如果 EAGLE-3 接受率低于阈值，切换到 ngram 草稿。报告可靠性提升。

3. 运行受控 MoE 实验：同一 Qwen3-Coder-30B，注入路由噪声 vs 不注入。衡量草稿接受率敏感度。

4. 扩展到 H200（141 GB）。报告获得的每副本模型大小余量，以及是否可以服务未量化的 Llama 3.3 70B。

5. 在同一 H100 硬件上基准 TensorRT-LLM 投机解码。报告它在哪些方面优于 vLLM。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 草稿模型 | "投机器" | 提议 N 个 Token 供目标验证的小模型 |
| EAGLE-3 | "2026 草稿架构" | 在目标隐藏状态上训练的草稿头；约 75% 接受率 |
| P-EAGLE | "并行投机" | 草稿分支树在一次目标遍中验证 |
| 接受率 | "命中率" | 被接受无需重采样的草稿 Token 比例 |
| 量化 | "FP8 / INT4" | 低精度权重以在 GPU 内存中装入更多模型 |
| 队列等待 | "HPA 指标" | 请求在推理开始前在待处理队列中等待的时间 |
| Speculators hub | "对齐草稿" | Red Hat Neural Magic 的 EAGLE 草稿 hub，用于常见开放模型 |

## 延伸阅读

- [vLLM EAGLE 和 P-EAGLE 文档](https://docs.vllm.ai) — 参考服务技术栈
- [P-EAGLE（AWS 2026）](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 并行投机解码论文 + 集成
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 草稿头训练流水线
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 对齐草稿 hub
- [TensorRT-LLM 投机解码](https://nvidia.github.io/TensorRT-LLM/) — 厂商替代
- [Fireworks.ai 服务架构](https://fireworks.ai/blog) — 商用参考
- [EAGLE-3 论文（arXiv:2503.01840）](https://arxiv.org/abs/2503.01840) — 方法论文
- [vLLM 仓库](https://github.com/vllm-project/vllm) — 代码和基准
