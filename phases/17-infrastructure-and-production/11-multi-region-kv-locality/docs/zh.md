# 多区域 LLM 服务与 KV 缓存局部性

> 轮询负载均衡对缓存 LLM 推理是有害的。一个没有落到持有其前缀的节点上的请求要支付完整的预填充成本 — 在长提示上 P50 约 800 ms，而缓存命中约 80 ms。2026 年的生产模式是缓存感知路由器（Rust 版 vLLM Router、llm-d router），消费 KV 缓存事件并基于前缀哈希匹配进行路由。最新研究（GORGO）将跨区域网络延迟作为路由目标中的显式项。商业"跨区域推理"产品（Bedrock 跨区域推理、GKE 多集群网关）将推理视为不透明 — 它们处理可用性，而非 TTFT。JPMorgan 和 Mayo Clinic 在 2024 年 11 月进行了 us-east-1 故障转移，耗时约 22 分钟。灾备现实：32% 的 LLM 灾备失败是因为团队备份了权重但忘记了分词器文件或量化配置。

**类型：** 学习
**语言：** Python（标准库，简易前缀缓存感知路由器模拟器）
**前置要求：** Phase 17 · 04（vLLM 服务），Phase 17 · 06（SGLang RadixAttention）
**所需时间：** 约60分钟

## 学习目标

- 解释为什么轮询负载均衡破坏缓存推理，并量化 TTFT 惩罚。
- 画出缓存感知路由器：输入（KV 缓存事件）、算法（前缀哈希匹配）、平局决胜（GPU 利用率）。
- 说出 LLM 32% 灾备失败的原因（缺失的分词器文件/量化配置），并说明三文件灾备检查清单。
- 区分商业跨区域产品（Bedrock CRI、GKE 多集群网关）和 KV 感知路由。

## 问题所在

你的服务运行在 us-east-1、us-west-2 和 eu-west-1。你在前面放了一个使用轮询的 ALB。生产中的前缀缓存命中率降到 8%。TTFT P50 翻了三倍。你的 vLLM 日志显示每个请求都在支付完整的预填充成本。

轮询对无状态服务是最优的。LLM 推理在设计上是有状态的 — KV 缓存编码了模型所见过的一切。盲目路由就是路由到错误的缓存。

另外，你的团队有一个灾备方案。你将模型权重备份到跨区域 S3。一个区域中断发生了；你尝试故障转移；副本拒绝启动。你忘记 tokenizer.json、量化配置和 RoPE 缩放配置在一个你没有同步的独立存储桶中。

多区域 LLM 服务是一个缓存问题、一个路由问题和一个灾备卫生问题 — 不是负载均衡器问题。

## 概念说明

### 缓存感知路由

请求带着提示到达。路由器对前缀进行哈希（比如前 512 个 token）；它询问每个副本"你缓存了这个前缀吗？"副本在分配和驱逐块时通过发布/订阅通道发布 KV 缓存事件。路由器选择匹配的副本，如果没有匹配则回退到基于 GPU 利用率的平局决胜。

**vLLM Router**（Rust，2026 production-stack）：订阅 `kv.cache.block_added` 事件，维护前缀哈希 → 副本索引，O(1) 查找路由。无匹配时回退到最少队列深度。

**llm-d router**：相同模式，Kubernetes 原生。通过 ControlPlane API 发布事件。

**SGLang RadixAttention**（Phase 17 · 06）是副本内等效方案。跨副本路由严格在上游。

### 数字

2K token 提示上的 TTFT P50，Llama 3.3 70B FP8，H100：
- 缓存命中（同一副本，前缀驻留）：约 80 ms。
- 缓存未命中（冷预填充）：约 800 ms。

10 倍差距。如果你的路由器跨副本命中 60-80% 的前缀缓存，你在 N 副本容量下近似单副本性能。如果命中 10%，你近似朴素扩缩。

### 跨区域有一个新约束 — 网络延迟

区域间 RTT：
- us-east-1 ↔ us-west-2：约 65 ms。
- us-east-1 ↔ eu-west-1：约 75 ms。
- us-east-1 ↔ ap-southeast-1：约 220 ms。

如果路由将一个请求从 us-east-1 发送到 ap-southeast-1 的热前缀，节省的预填充（800 → 80 ms）被 440 ms 的往返淹没。GORGO（2026 年研究）使这一点显式 — 联合最小化 `prefill_time + network_latency`，而非仅预填充。通常答案是保持路由在区域内，除非在多 MB 前缀且预填充占主导的场景。

### 商业"跨区域推理"在这里无济于事

AWS Bedrock 跨区域推理在容量压力期间自动将请求路由到其他区域。它优化可用性，而非 TTFT，并将推理视为不透明。GKE 多集群网关也是如此 — 服务级故障转移，无 KV 缓存感知。

即使使用这些产品，你仍然需要一个应用层缓存感知路由器。它们处理"us-east-1 着火了"的情况。缓存感知路由处理 TTFT 情况。

### 灾备卫生 — 32% 缺失文件问题

广泛引用的 2026 年统计数据：32% 的 LLM 灾备失败是因为团队备份了权重但忘记了：

- `tokenizer.json` 或 `tokenizer.model`
- 量化配置（`quantize_config.json`、AWQ scales、GPTQ zero-points）
- 模型特定配置（RoPE 缩放、注意力掩码、聊天模板）
- 引擎配置（`vllm_config.yaml`、采样默认值、LoRA 适配器清单）

解决方案是一个三文件最低灾备清单：

1. HF 模型仓库下的所有文件（权重 + 配置 + 分词器）。
2. 引擎特定的服务配置。
3. 部署清单（K8s YAML、Dockerfile、依赖锁）。

加上：每季度进行一次灾备演练。JPMorgan us-east-1 演练在 2024 年 11 月达到 22 分钟恢复，只因为剧本经过了排练。

### 数据驻留是正交的

欧盟客户 PHI 不能离开欧盟。如果你的缓存感知路由器将巴黎发起的请求发送到 us-east-1 以匹配前缀，你就违反了 GDPR，无论 TTFT 收益如何。在优化缓存之前，按驻留边界分区路由器。

### 你应该记住的数字

- 缓存命中 vs 未命中 TTFT 差距：约 10 倍（2K 提示上 80 ms vs 800 ms）。
- 区域间 RTT 美国-欧盟：约 75 ms。
- 灾备失败：32% 缺失分词器/量化配置。
- JPMorgan us-east-1 故障转移 2024 年 11 月：22 分钟（30 分钟 SLA）。

## 开始使用

`code/main.py` 在多区域工作负载上模拟三种路由策略（轮询、缓存感知区域、缓存感知全局）。报告缓存命中率、TTFT P50/P99 和跨区域费用。

## 交付成果

本课生成 `outputs/skill-multi-region-router.md`。给定区域、驻留约束和 SLA，设计路由方案。

## 练习

1. 运行 `code/main.py`。在给定 75 ms RTT 下，多长的提示时跨区域路由优于纯本地路由？
2. 你的缓存命中率从 70% 降到 12%。诊断三个可能原因以及确认每个原因的可观测指标。
3. 为一个在 vLLM 中服务的 70B AWQ 量化模型设计灾备清单，带有 5 个 LoRA 适配器。列出每个文件和配置。
4. 论证 Bedrock 跨区域推理对于有严格 TTFT SLO 的金融科技是否"足够"。引用具体行为。
5. 一个巴黎发起的请求匹配 us-east-1 的前缀。你路由它吗？编写策略。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 缓存感知路由 | "智能 LB" | 基于前缀哈希匹配路由到持有 KV 缓存的副本 |
| KV 缓存事件 | "缓存发布订阅" | 副本发布块添加/驱逐；路由器索引 |
| 前缀哈希 | "缓存键" | 前 N 个 token 的哈希，用作路由器查找 |
| GORGO | "跨区域路由研究" | arXiv 2602.11688；网络延迟作为显式项 |
| 跨区域推理 | "Bedrock CRI" | AWS 产品；可用性故障转移，非 TTFT 感知 |
| 灾备清单 | "备份列表" | 恢复所需的每个文件 — 不仅仅是权重 |
| 数据驻留 | "GDPR 边界" | 哪个区域可以看到用户数据的法律约束 |
| RTT | "往返时间" | 网络延迟；美国-欧盟 75 ms，美国-亚太 220 ms |
| LLM 感知 LB | "缓存命中 LB" | 作为产品类别的缓存感知路由器 |

## 延伸阅读

- [BentoML — 多云和跨区域推理](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO（2602.11688）](https://arxiv.org/html/2602.11688v1) — 带网络延迟项的跨区域 KV 缓存重用。
- [TianPan — 多区域 LLM 服务缓存局部性](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock 跨区域推理](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html) — 可用性故障转移文档。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack) — 缓存感知路由器源代码。
