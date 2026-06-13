# TensorRT-LLM 在 Blackwell 上的应用 — FP8 和 NVFP4

> TensorRT-LLM 仅限 NVIDIA，但在 Blackwell 上胜出。在 GB200 NVL72 上使用 Dynamo 编排，SemiAnalysis InferenceX 在 2026 年 Q1-Q2 测得 120B 模型每百万 token 0.012 美元，而 H100 + vLLM 为 0.09 美元/百万 — 7 倍的经济差距。技术栈是三种浮点格式的叠加：FP8 对 KV 缓存和注意力内核仍然至关重要，因为它具有所需的动态范围；NVFP4（4 位微缩放）处理权重和激活；多 token 预测（MTP）和分离式预填充/解码在此基础上再增加 2-3 倍。Day-0 模型支持直接加载 FP4 权重，无需训练后转换。2026 年工程团队的关键点：TRT-LLM 是封闭的 NVIDIA 技术栈，采用它是在用可移植性换取吞吐量。在承诺之前，先根据你的模型和硬件组合算一算。

**类型：** 学习
**语言：** Python（标准库，简易 FP8/NVFP4 内存和成本计算器）
**前置要求：** Phase 17 · 04（vLLM 服务内部机制），Phase 10 · 13（量化）
**所需时间：** 约75分钟

## 学习目标

- 解释为什么即使权重使用 NVFP4，FP8 对 KV 缓存和注意力仍然至关重要。
- 计算前沿模型在 BF16、FP8 和 NVFP4 下的 HBM 占用，并推理节省来自哪里。
- 说出 TRT-LLM 利用的 Blackwell 特定功能（Day-0 FP4、MTP、分离式服务、All-to-All 原语）。
- 决定 TRT-LLM 的 NVIDIA 锁定何时值得与 Hopper 上的 vLLM 之间 7 倍的成本差距。

## 问题所在

2026 年推理经济学的前沿是"每美元多少 token"。答案取决于四个叠加选择：硬件世代（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、服务引擎（vLLM vs SGLang vs TRT-LLM）和编排（普通 vs 分离式 vs Dynamo）。

在 Hopper 上使用 vLLM，120B MoE 的运行成本约为每百万 token 0.09 美元。在 Blackwell 上使用 TRT-LLM + Dynamo，同一模型约为 0.012 美元 — 便宜 7 倍。部分差距来自硬件（Blackwell 每 GPU LLM 吞吐量是 Hopper 的 11-15 倍）。部分来自技术栈：FP4 权重、MTP 草稿、分离式预填充/解码，以及 NVLink 5 All-to-All 用于 MoE 专家通信。

你无法在 NVIDIA 技术栈之外复制这一点。这就是权衡 — 可移植性换取经济性。理解哪些技术栈选择贡献了多少差距是本课的重点。

## 概念说明

### 为什么 FP8 仍然是 KV 缓存的底限

2026 年的常见错误：认为 NVFP4 处处适用。并非如此。KV 缓存需要 FP8（8 位浮点数），因为它存储的注意力键和值跨越宽动态范围。将 KV 量化为 FP4 会导致灾难性的精度损失 — 分布尾部衰减，注意力分数崩塌。FP8 的指数位给予 KV 缓存所需的范围。

NVFP4（2025-2026）适用于权重和激活。微缩放：每个权重块有自己的缩放因子，使小块可以跨越不同的动态范围而不会因每张量缩放而损失。对于激活，FP4 表现良好，因为激活在层内范围较小。

典型的 Blackwell 配置：

- 权重：NVFP4（4 位微缩放）。
- 激活：NVFP4。
- KV 缓存：FP8。
- 累加器：FP32（softmax 稳定性）。

### TRT-LLM 使用的 Blackwell 特定原语

- **Day-0 FP4 权重**：模型提供商直接发布 FP4 权重；TRT-LLM 无需训练后转换即可加载。FP4 无需 AWQ / GPTQ 步骤。
- **多 token 预测（MTP）**：与 EAGLE（Phase 17 · 05）相同的理念，但集成到 TRT-LLM 构建中。
- **分离式服务**：预填充和解码在独立 GPU 池上运行，KV 缓存通过 NVLink 或 InfiniBand 传输。与 Dynamo（Phase 17 · 20）相同的理念。
- **All-to-All 通信原语**：NVLink 5 将 MoE 专家通信延迟降低 3 倍（相比 Hopper）。TRT-LLM 的 MoE 内核为此做了调优。
- **NVFP4 + MXFP8 微缩放**：Blackwell Tensor Cores 上的硬件加速缩放因子处理。

### 你应该记住的数字

- HGX B200 使用 TRT-LLM 在 GPT-OSS-120B 上为 0.02 美元/百万 token。
- GB200 NVL72 使用 Dynamo（编排 TRT-LLM）为 0.012 美元/百万 token。
- H100 + vLLM 在可比工作负载上约为 0.09 美元/百万 token。
- TRT-LLM 更新三个月内吞吐量提升 2.8 倍（2026 年）。
- 每 GPU LLM 吞吐量：Blackwell vs Hopper 为 11-15 倍。
- MLPerf Inference v6.0（2026 年 4 月）：Blackwell 在每个提交的任务上占主导。

### FP4 实际在质量上付出什么代价

NVFP4 是激进的。在推理密集型工作负载（思维链、数学、长上下文代码生成）上，FP4 权重明显退化。每块校准缓解但无法消除。发布推理模型的团队通常使用 FP8 权重 + FP4 激活作为折中，或坚持在 H200 上全程使用 FP8。

规则：在承诺 NVFP4 权重之前，始终在你的评估集上验证任务质量。

### 为什么这是一个 NVIDIA 锁定决策

TRT-LLM 是 C++ + CUDA + 闭源内核。模型需要为特定 GPU SKU 编译。不支持 AMD、Intel、ARM。如果你的基础设施策略是多供应商，TRT-LLM 对 TRT-LLM 服务层来说不是可行选择 — 你仍然可以在混合硬件上使用 vLLM 服务。如果你是纯 NVIDIA，7 倍差距值得锁定。

### 2026 年实用配方

对于年推理费用超过 1 亿美元的场景，在 Hopper + vLLM 上运行会损失 7-10 倍。将成本主导的工作负载迁移到 Blackwell + TRT-LLM + Dynamo。将实验层保留在 H100 + vLLM 上以获得模型迭代速度。在生产前验证每个 NVFP4 转换模型的质量。

### 分离式服务的额外收益

TRT-LLM 的分离式服务（独立的预填充和解码池）在 Phase 17 · 20 中有深入介绍。在 Blackwell 上，乘数叠加：FP4 权重 × MTP 加速 × 分离式放置 × 缓存感知路由。7 倍的数字假设了这个完整技术栈。

```figure
pipeline-parallel
```

## 开始使用

`code/main.py` 计算模型在三种技术栈上的 HBM 占用、解码吞吐量（内存受限场景）和每百万 token 成本：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。运行它以查看叠加效应以及每次变化贡献的差距份额。

## 交付成果

本课生成 `outputs/skill-trtllm-blackwell-advisor.md`。给定工作负载、模型大小和年度 token 量，决定 Blackwell + TRT-LLM 技术栈是否值得 NVIDIA 锁定。

## 练习

1. 运行 `code/main.py`。对于一个 30% 活跃参数的 120B MoE，计算 H100 BF16、H100 FP8 和 B200 NVFP4/FP8 上的内存带宽受限解码吞吐量。最大的跳跃来自哪里？
2. 一个客户每年在 H100 + vLLM 上花费 200 万美元。在 7 倍经济差距下，需要购买多少 Blackwell GPU 才能在 12 个月内摊销迁移到 TRT-LLM 的成本？
3. NVFP4 权重转换后 MATH 准确率下降 3 分。说出两条恢复路径：一条质量优先（保留 FP8 权重），一条成本优先（使用领域内数据校准）。
4. 阅读 MLPerf v6.0 推理结果。哪个任务的 Blackwell 对 Hopper 差距最小，为什么？
5. 计算 405B 模型在 NVFP4 权重 + FP8 KV 缓存、128k 上下文下所需的 HBM。它能装进单个 GB200 NVL72 节点吗？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| FP8 | "8 位浮点" | 8 位浮点数；因动态范围用于 KV 缓存和注意力 |
| NVFP4 | "4 位微缩放" | NVIDIA 的 4 位微缩放 FP 格式；Blackwell 上的权重和激活 |
| MXFP8 | "MX 8" | 微缩放 FP8 变体；Blackwell Tensor Cores 上硬件加速 |
| Day-0 FP4 | "直接发布 FP4 权重" | 模型提供商已发布 FP4 格式的权重；无训练后转换步骤 |
| MTP | "多 token 预测" | TRT-LLM 的集成推测解码草稿（Phase 17 · 05） |
| 分离式服务 | "分离预填充/解码" | 预填充和解码在独立 GPU 池上运行；KV 通过 NVLink/IB 传输 |
| All-to-All | "MoE 专家通信" | 将 token 路由到专家 GPU 的通信模式；NVLink 5 降低 3 倍 |
| InferenceX | "SemiAnalysis 推理基准" | 2026 年行业认可的每 token 成本基准 |

## 延伸阅读

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/) — 2026 年 4 月 MLPerf 结果。
- [NVIDIA — Blackwell 上的 MoE 推理](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/) — NVLink 5 All-to-All 和 MoE 内核。
- [TensorRT-LLM 概览](https://nvidia.github.io/TensorRT-LLM/overview.html) — 官方引擎文档。
- [NVIDIA — Dynamo 介绍](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) — TRT-LLM 之上的分离式编排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/) — 发布 Blackwell 数字的基准测试套件。
