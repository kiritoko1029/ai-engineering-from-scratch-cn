# Kubernetes 上的 GPU 自动扩缩 — Karpenter、KAI Scheduler、Gang Scheduling

> 三层，而非一层。Karpenter 动态预配节点（不到一分钟，比 Cluster Autoscaler 快 40%）。KAI Scheduler 处理 Gang Scheduling、拓扑感知和分层队列 — 它防止了"8 中缺 1"的部分分配陷阱，即 7 个节点等待并因缺少一个 GPU 而烧钱。应用层自动扩缩器（NVIDIA Dynamo Planner、llm-d 工作负载变体自动扩缩器）基于推理特定信号进行扩缩 — 队列深度、KV 缓存利用率 — 而非 CPU/DCGM 占空比。经典 HPA 陷阱在于 `DCGM_FI_DEV_GPU_UTIL` 是占空比度量：100% 可能是 10 个请求或 100 个请求。vLLM 预分配 KV 缓存内存，因此内存永远不会触发缩容。本课教你组合这三层，并避免 Karpenter 默认的 `WhenEmptyOrUnderutilized` 策略在推理过程中终止正在运行的 GPU 任务。

**类型：** 学习
**语言：** Python（标准库，简易队列深度自动扩缩模拟器）
**前置要求：** Phase 17 · 02（推理平台经济学），Phase 17 · 04（vLLM 服务内部机制）
**所需时间：** 约75分钟

## 学习目标

- 画出三层自动扩缩体系（节点预配、Gang Scheduling、应用层），并命名每层使用的工具。
- 解释为什么 `DCGM_FI_DEV_GPU_UTIL` 是 vLLM 的错误 HPA 信号，并命名两个替代方案（队列深度、KV 缓存利用率）。
- 描述 Gang Scheduling 以及 KAI Scheduler 防止的部分分配失败模式（8 GPU 中 7 个空闲）。
- 命名 Karpenter 的合并策略（`WhenEmptyOrUnderutilized`），该策略会终止正在运行的 GPU 任务，并说明 2026 年的安全替代方案。

## 问题所在

你的团队在 Kubernetes 上部署了 LLM 服务。你设置了 HPA，使用 `DCGM_FI_DEV_GPU_UTIL` 作为信号。工作时间内服务的利用率钉在 100%。HPA 永远不会扩容 — 它已经认为你满了。你手动添加一个副本；TTFT 下降。HPA 仍然不扩容。信号在欺骗你。

另外，你使用 Cluster Autoscaler 管理节点。凌晨 2 点一个 100 万 token 的提示到达；集群花了 3 分钟预配一个节点，请求超时了。

再另外，你部署了一个需要跨 2 个节点使用 8 个 GPU 的 70B 模型。集群有 7 个空闲 GPU 和分散在 3 个节点上的 1 个 GPU。Cluster Autoscaler 为缺失的 1 个 GPU 预配了一个节点。7 个节点等待 4 分钟，烧钱等 Kubernetes 把最后一个 GPU 拉起来。

三层，三种不同的失败模式。2026 年的 GPU 感知自动扩缩不是"打开 HPA"。而是组合节点预配、Gang Scheduling 和应用信号自动扩缩。

## 概念说明

### 第 1 层 — 节点预配（Karpenter）

Karpenter 监视待处理的 Pod，在约 45-60 秒内预配节点（Cluster Autoscaler 对 GPU 节点通常需要 90-120 秒）。它根据 `NodePool` 约束动态选择实例类型 — 如果你的 Pod 需要 8 个 H100 且集群没有匹配的节点，Karpenter 直接预配一个，而不是扩展现有的组。

**合并陷阱**：Karpenter 默认的 `consolidationPolicy: WhenEmptyOrUnderutilized` 对 GPU 池来说是危险的。它会终止正在运行的 GPU 节点，将 Pod 迁移到更便宜的合适实例。对于推理工作负载，这意味着驱逐正在运行的请求并在新节点上重新加载 70B 模型。损失是几分钟的容量加上请求失败。

GPU 池的安全设置：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

允许 Karpenter 在一小时后合并真正空闲的节点，但从不驱逐正在运行的任务。

### 第 2 层 — Gang Scheduling（KAI Scheduler）

KAI Scheduler（项目原名 "Karp"，后更名）处理默认 kube-scheduler 不做的事情：

**Gang Scheduling** — 要么全部调度，要么全部不调度。一个需要 8 个 GPU 的分布式推理 Pod，要么全部 8 个一起启动，要么一个都不启动。没有这个，你会遇到部分分配陷阱：8 个 Pod 中 7 个启动了，无限等待，烧钱。

**拓扑感知** — 知道哪些 GPU 共享 NVLink，哪些在同一机架上，哪些之间有 InfiniBand 连接。相应地放置 Pod。DeepSeek-V3 67B 张量并行工作负载必须保持在一个 NVLink 域内；KAI Scheduler 遵循这一约束。

**分层队列** — 多个团队以优先级和配额竞争同一个 GPU 池。Team A 的生产抢占仅在优先级规则允许时才会被 Team B 的训练任务抢占。

KAI 作为二级调度器与 kube-scheduler 一起部署；你通过注解工作负载来使用它。Ray 和 vLLM production-stack 都已集成。

### 第 3 层 — 应用层信号

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是占空比度量 — 它在每个采样间隔测量 GPU 是否在执行工作。100% 利用率可能意味着 10 个并发请求或 100 个；GPU 无论如何都是忙碌的。基于占空比扩缩就是盲目扩缩。

更糟糕的是，vLLM 和类似引擎预分配 KV 缓存内存（最高 `--gpu-memory-utilization`）。即使只有 1 个请求，内存使用率也保持在约 90%。基于内存的 HPA 永远不会缩容。

**2026 年替代信号**：

- 队列深度（等待预填充的请求数）。
- KV 缓存利用率（已分配给活跃序列的块比例）。
- 每副本 P99 TTFT（你的 SLA 信号）。
- Goodput（每秒满足所有 SLO 的请求数）。

NVIDIA Dynamo Planner 和 llm-d 工作负载变体自动扩缩器消费这些信号并扩缩副本。它们完全替代了 HPA 用于 LLM 服务。

### 何时使用什么

| 扩缩决策 | 工具 |
|----------|------|
| 添加/移除节点 | Karpenter |
| 调度多 GPU 任务 | KAI Scheduler |
| 添加/移除副本 | Dynamo Planner / llm-d WVA（或基于队列深度的自定义 HPA） |
| 选择 GPU 类型 | Karpenter NodePool |
| 抢占低优先级 | KAI Scheduler 队列 |

### 分离式预填充/解码使一切复杂化

如果你运行分离式预填充/解码（Phase 17 · 17），你有两个具有不同扩缩触发器的 Pod 类别：预填充 Pod 基于队列深度扩缩，解码 Pod 基于 KV 缓存压力扩缩。llm-d 将它们暴露为独立的 `Service`，每个角色有自己的 HPA。不要试图在两者前面放一个 HPA。

### 冷启动在这里也很重要

冷启动缓解（Phase 17 · 10）是节点预配时间变得对用户可见的地方。Karpenter 的 45-60 秒启动加上 20GB 模型加载加上引擎初始化意味着从零开始的请求需要 2-5 分钟。为 SLO 关键路径保持热池（`min_workers=1`），或在应用层使用 Modal 风格的检查点。

### 你应该记住的数字

- Karpenter 节点预配：约 45-60 秒 vs Cluster Autoscaler 约 90-120 秒（GPU 节点）。
- KAI Scheduler 防止部分分配浪费 — 8 中缺 1 陷阱。
- `DCGM_FI_DEV_GPU_UTIL` 作为 HPA 信号：无效；使用队列深度或 KV 利用率。
- Karpenter `WhenEmptyOrUnderutilized`：终止正在运行的 GPU 任务。推理场景使用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 开始使用

`code/main.py` 在突发 GPU 工作负载上模拟三层自动扩缩器。比较朴素 HPA（占空比）、队列深度 HPA 和 KAI Gang Scheduled 扩缩。报告未满足的请求数、空闲 GPU 分钟数和综合评分。

## 交付成果

本课生成 `outputs/skill-gpu-autoscaler-plan.md`。给定集群拓扑、工作负载形状和 SLO，设计一个三层自动扩缩方案。

## 练习

1. 运行 `code/main.py`。在突发工作负载下，朴素占空比 HPA 丢弃了多少请求是队列深度 HPA 能捕获的？差异从何而来？
2. 为一个在 H100 SXM5 上服务 Llama 3.3 70B FP8 的集群设计一个 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及一个将非 GPU 工作负载排除在这些节点之外的污点。
3. 你的团队报告部署卡在 Pending 状态，因为"GPU 可用但 Pod 无法调度"。诊断 — 这是 Karpenter、kube-scheduler 还是 KAI Scheduler 的问题？哪些指标可以确认？
4. 选择一个信号来自动扩缩分离式预填充 Pod，选择另一个不同的信号用于解码 Pod。论证两者的选择。
5. 计算 `WhenEmptyOrUnderutilized` 合并陷阱在 24x7 生产服务上的成本，该服务平均每天发生 60 次 P99 TTFT > 10 秒的请求丢弃事件。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Karpenter | "节点预配器" | Kubernetes 节点自动扩缩器；亚分钟级预配 |
| Cluster Autoscaler | "旧扩缩器" | Kubernetes 节点自动扩缩器前身；较慢，基于组 |
| KAI Scheduler | "GPU 调度器" | 用于 Gang + 拓扑 + 队列的二级调度器 |
| Gang Scheduling | "要么全部要么全无" | 原子地调度 N 个 Pod 或全部延迟 |
| 拓扑感知 | "机架感知" | 基于 NVLink/IB/机架放置 Pod |
| `DCGM_FI_DEV_GPU_UTIL` | "GPU 利用率" | 占空比度量；不是 LLM 的扩缩信号 |
| 队列深度 | "等待中的请求" | 预填充受限扩缩的正确 HPA 信号 |
| KV 缓存利用率 | "内存压力" | 解码受限扩缩的正确 HPA 信号 |
| 合并 | "Karpenter 合并" | 将节点终止为更便宜的实例类型 |
| `WhenEmpty + 1h` | "安全合并" | 不驱逐正在运行的 GPU 任务的策略 |

## 延伸阅读

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler) — 设计文档和配置示例。
- [Karpenter 中断控制](https://karpenter.sh/docs/concepts/disruption/) — 合并策略语义和 GPU 安全默认值。
- [NVIDIA — Kubernetes 上的分离式 LLM 推理](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/) — Dynamo Planner 扩缩信号。
- [Ray 文档 — RayClusters 的 KAI Scheduler](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html) — Ray 集成模式。
- [AWS EKS 计算与自动扩缩最佳实践](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html) — 托管 Kubernetes 特定指导。
- [llm-d GitHub](https://github.com/llm-d/llm-d) — 工作负载变体自动扩缩器设计。
