# 分离式 Prefill/Decode —— NVIDIA Dynamo 与 llm-d

> Prefill 是计算密集型；decode 是内存密集型。在同一 GPU 上运行两者会浪费其中一种资源。分离式架构将它们拆分到独立的资源池，并通过 NIXL（RDMA/InfiniBand 或 TCP 回退）在两者之间传输 KV cache。NVIDIA Dynamo（GTC 2025 发布，1.0 GA）位于 vLLM/SGLang/TRT-LLM 之上——其 Planner Profiler + SLA Planner 可自动匹配 prefill:decode 比例以满足 SLO。NVIDIA 发布了吞吐量提升数据——developer.nvidia.com（2025-06）显示在 GB200 NVL72 + Dynamo 上，DeepSeek-R1 MoE 在中等延迟场景下可获得约 6 倍提升；Dynamo 产品页面（developer.nvidia.com，未注明日期）宣传在 GB300 NVL72 + Dynamo 上相比 Hopper 最高可达 50 倍 MoE 吞吐量。"30 倍"这个数字是社区对全栈 Blackwell + Dynamo + DeepSeek-R1 报告的综合汇总；我们未找到单一原始来源确切给出 30 倍的说法，因此将其视为方向性参考。llm-d（Red Hat + AWS）是 Kubernetes 原生方案：prefill / decode / router 作为独立的 Service，支持按角色的 HPA。llm-d 0.5 新增了层次化 KV 卸载、缓存感知 LoRA 路由、UCCL 网络和缩容至零。经济效益：多个客户披露信息的内部汇总显示，从共置部署切换到使用 Dynamo 的分离式架构，在保持相同 SLA 的情况下，$2M 级别的推理支出可节省 30–40%（即 $600-800K/年）；$2M→$600-800K 这个具体数字是内部综合数据，不是单一公开案例——将其作为数量级参考而非引用出处。短提示（<512 tokens，短输出）无法证明传输开销的合理性。

**类型：** 学习
**语言：** Python（标准库，简易分离式 vs 共置模拟器）
**前置要求：** 第 17 阶段 · 04（vLLM Serving 内部原理），第 17 阶段 · 08（推理指标）
**所需时间：** 约 75 分钟

## 学习目标

- 解释为什么 prefill 和 decode 有不同的最优 GPU 分配方案，并量化共置部署下的浪费。
- 绘制分离式架构图：prefill 池、decode 池、通过 NIXL 的 KV 传输、路由器。
- 说明分离式架构不划算的条件（短提示、短输出）。
- 区分 NVIDIA Dynamo（上层编排）与 llm-d（Kubernetes 原生），并匹配各自的适用场景。

## 问题所在

你在 8 张 H100 上运行 Llama 3.3 70B。在混合工作负载（长提示 + 短输出）下，GPU 在 decode 阶段空闲，因为大部分计算已经消耗在 prefill 上。在另一种工作负载（短提示 + 长输出）下，则出现相反的情况。共置 prefill + decode 意味着你对两者都进行了过度配置。

预算影响：20-40% 的 GPU 时间浪费在了错误的资源上。你花 H100 的计算资源来跑内存密集型的 decode，或者花 H100 的 HBM 带宽来跑计算密集型的 prefill。两者都是昂贵的浪费。

分离式架构将 prefill 和 decode 拆分到独立的池中，各自按瓶颈进行配置。KV cache 从 prefill 池通过高带宽互连传输到 decode 池。

## 概念说明

### 为什么瓶颈不同

**Prefill** —— 在一次前向传播中对完整输入提示运行 Transformer。矩阵乘法占主导；计算密集型。H100 FP8 提供约 2000 TFLOPS 的有效吞吐。批处理效率高——一次前向处理多个 token。

**Decode** —— 每次只生成一个 token，每次迭代都要读取完整权重。内存带宽密集型。HBM3 提供约 3 TB/s。批处理效率仅在高并发下才好——权重读取可在整个批次中分摊。

共置部署意味着你需要为两种场景都优化的 GPU。H100 两方面都不错，但成本固定。在大规模场景下，你需要在 H100 / 计算密集型上配置 prefill 池，在 H200 / 内存密集型上配置 decode 池，或采用激进量化。

### 架构

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的节点间传输层。优先使用 RDMA/InfiniBand，不可用时回退到 TCP。传输延迟是真实存在的——在 70B FP8 上，4K token 提示的 KV cache 传输通常需要 20-80 毫秒。这就是短提示不适合分离式架构的原因：传输开销超过了收益。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 发布，1.0 GA）：
- 位于 vLLM、SGLang、TRT-LLM 之上作为编排器。
- Planner Profiler 测量工作负载，SLA Planner 自动配置 prefill:decode 比例。
- Rust 核心，Python 可扩展性。
- 吞吐量提升：NVIDIA 报告在 GB200 NVL72 + Dynamo 上，DeepSeek-R1 MoE 在中等延迟场景下可达 6 倍（developer.nvidia.com，2025-06）；社区报告的全 Blackwell + Dynamo + DeepSeek-R1 栈上"最高 30 倍"缺乏单一原始来源，应视为方向性参考。
- GB300 NVL72 + Dynamo：根据 Dynamo 产品页面（developer.nvidia.com，未注明日期），MoE 吞吐量相比 Hopper 最高可达 50 倍。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- Prefill / decode / router 作为独立的 Kubernetes Service。
- 按角色的 HPA，使用队列深度（prefill）/ KV 利用率（decode）信号。
- `topologyConstraint packDomain: rack` 将 prefill+decode 集群部署在同一机架上，实现高带宽 KV 传输。
- llm-d 0.5（2026）：层次化 KV 卸载、缓存感知 LoRA 路由、UCCL 网络、缩容至零。

如果你需要托管的上层编排器，选择 Dynamo。如果你需要 Kubernetes 原生原语并致力于 CNCF 生态系统，选择 llm-d。

### 经济效益

内部综合数据（非单一公开案例——数量级参考）：

- 共置部署推理年支出 $2M。
- 切换到使用 Dynamo 的分离式架构。
- 相同请求量，相同 P99 延迟 SLA。
- 报告的节省：$600K–$800K/年（降低 30–40%）。
- 无新增硬件。

该数字综合自多个客户披露信息，而非单一可引用案例；最接近的公开数据点是 Baseten 使用 Dynamo KV 路由实现 2 倍 TTFT 提升 / 61% 吞吐量提升（baseten.co，2025-10），以及 VAST + CoreWeave 在 40–60% KV 命中率下预计 60–130% tokens/$ 提升（vastdata.com，2025-12）。节省来自对各池的合理配置；prefill 密集型工作负载（带 8K+ 前缀的 RAG）比均衡型获益更多。

### 何时不适合分离

- 提示 < 512 tokens 且输出 < 200 tokens：传输开销超过收益。
- 小型集群（< 4 GPU）：池的多样性不足。
- 团队无法运维两个带按角色伸缩的 GPU 池：Dynamo 有帮助但并非易事。
- 无 RDMA 网络：TCP 传输开销更大。

### 路由器与第 17 阶段 · 11 的集成

分离式路由器是 KV cache 感知的（第 17 阶段 · 11）。请求被路由到持有其前缀的 decode 池——如果没有匹配，则走 prefill → decode 流程。命中率与分离式架构相辅相成——缓存感知路由器决定了是否需要进行新的 prefill。

### MoE 在 Blackwell 上的表现才是真正的数字

GB300 NVL72 + Dynamo 显示 MoE 吞吐量相比 Hopper 基线提升 50 倍。MoE 专家路由在 prefill 阶段计算密集，在 decode 阶段内存密集（专家缓存），因此分离式架构是双重收益。2026 年前沿模型服务以 MoE 为主（DeepSeek-V3、未来 GPT-5 变体）。

### 需要记住的数字

基准数字会变化——NVIDIA 和推理栈每季度都会发布更新结果。引用前请重新核实。

- DeepSeek-R1 在 GB200 NVL72 + Dynamo 上：中等延迟场景下相比基线约 6 倍吞吐量（developer.nvidia.com，2025-06）；社区全 Blackwell + Dynamo 栈上"最高 30 倍"的说法是无单一原始来源的方向性汇总。
- GB300 NVL72 + Dynamo：MoE 吞吐量相比 Hopper 最高可达 50 倍（developer.nvidia.com，未注明日期）。
- 节省参考（内部综合数据，非单一案例）：$2M 年支出在保持相同 SLA 下可节省 $600-800K/年。
- 分离阈值：提示 >512 tokens + 输出 >200 tokens。
- 通过 NIXL 的 KV 传输：70B FP8 上 4K 提示的 KV 传输需 20-80 毫秒。

## 开始使用

`code/main.py` 模拟共置 vs 分离式部署。报告吞吐量、单请求成本和提示长度分界点。

## 部署产出

本课程产出 `outputs/skill-disaggregation-decider.md`。给定工作负载和集群配置，判断是否应采用分离式架构。

## 练习

1. 运行 `code/main.py`。在什么提示长度下分离式架构优于共置部署？
2. 为一个 P99 前缀长度 8K、输出 300 的 RAG 服务设计 prefill 池和 decode 池。
3. Dynamo vs llm-d：对于一个没有 Python 运行时偏好、纯 Kubernetes 的团队，选择哪个？
4. 计算 KV 传输成本：70B FP8 上 4K prefill 约 500 MB KV。RDMA 100 GB/s 下传输 = 5 毫秒。TCP 10 GB/s 下 = 50 毫秒。哪个对你的 SLA 有影响？
5. MoE 专家路由改变了 KV 访问模式。对于每个 token 激活不同专家的 MoE，分离式架构表现如何？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 分离式服务 | "拆分 prefill/decode" | 为每个阶段使用独立的 GPU 池 |
| NIXL | "NVIDIA 传输层" | Dynamo 的节点间 KV 传输（RDMA/TCP） |
| NVIDIA Dynamo | "编排器" | 位于 vLLM/SGLang/TRT-LLM 之上的上层协调器 |
| llm-d | "Kubernetes 原生" | Red Hat + AWS 的 K8s 分离式栈 |
| Planner Profiler | "Dynamo 自动配置" | 测量工作负载，配置池比例 |
| SLA Planner | "Dynamo 策略" | 自动匹配 prefill:decode 以满足 SLO |
| `packDomain: rack` | "llm-d 拓扑" | 将 prefill+decode 打包在同一机架上以加速 KV |
| UCCL | "统一集合通信" | llm-d 0.5 的缩容至零网络层 |
| MoE 专家路由 | "每 token 一个专家" | DeepSeek-V3 模式；分离式架构有益 |

## 延伸阅读

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
