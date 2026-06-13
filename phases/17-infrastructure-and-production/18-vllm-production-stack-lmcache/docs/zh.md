# 基于 LMCache KV 卸载的 vLLM 生产环境部署

> vLLM 的 production-stack 是参考级 Kubernetes 部署方案——路由器、引擎和可观测性组件协同工作。LMCache 是 KV 卸载层，能够将 KV 缓存从 GPU 显存中提取出来，在多个查询和引擎之间复用（先使用 CPU DRAM，再使用磁盘/Ceph）。vLLM 0.11.0 的 KV 卸载连接器（2026 年 1 月发布）通过连接器 API（v0.9.0+）实现了异步化和可插拔设计，卸载延迟不会影响用户请求。即使没有共享前缀，LMCache 依然有价值——当 GPU 的 KV 插槽耗尽时，被抢占的请求可以从 CPU 恢复，而无需重新计算预填充。在 4 台 a3-highgpu-4g 上共 16 块 H100（80GB HBM）的公开基准测试表明：当 KV 缓存超出 HBM 容量时，原生 CPU 卸载和 LMCache 都能显著提升吞吐量；当 KV 占用较低时，所有配置与基线性能一致，仅产生少量开销。

**类型：** 学习
**语言：** Python（标准库，简易 KV 溢出模拟器）
**前置要求：** Phase 17 · 04（vLLM 服务内部机制）、Phase 17 · 06（SGLang/RadixAttention）
**所需时间：** 约 60 分钟

## 学习目标

- 画出 vLLM 生产环境部署的各层结构：路由器、引擎、KV 卸载、可观测性。
- 解释 KV 卸载连接器 API（v0.9.0+）的工作原理，以及 0.11.0 的异步路径如何隐藏卸载延迟。
- 量化 LMCache CPU-DRAM 何时有帮助（KV 超过 HBM）以及何时增加开销（KV 足够小可以放入 HBM）。
- 根据部署约束选择原生 vLLM CPU 卸载还是 LMCache 连接器。

## 问题所在

你的 vLLM 服务在并发上升时，GPU 显存占用率达到 100%，频繁出现抢占事件。请求被逐出、重新入队，同一段 2K token 的提示词在一分钟内被预填充了四次。GPU 算力被浪费在重复的预填充上，有效吞吐量远低于原始吞吐量。

增加 GPU 数量意味着线性成本增长。增加 HBM 容量又不可行。但 CPU DRAM 价格低廉——单个插槽拥有 512 GB 以上容量，延迟虽然比 HBM 差几个数量级，但对于"临时暂存"的 KV 缓存来说完全可以接受。

LMCache 将 KV 缓存提取到 CPU DRAM，使被抢占的请求能够快速恢复，同时跨引擎共享的前缀无需每个引擎重复预填充。

## 概念说明

### vLLM 生产环境部署

`github.com/vllm-project/production-stack` 是参考级 Kubernetes 部署方案：

- **路由器**——缓存感知（Phase 17 · 11），消费 KV 事件。
- **引擎**——vLLM 工作节点，每块 GPU 或每个 TP/PP 组一个。
- **KV 缓存卸载**——LMCache 部署或原生连接器。
- **可观测性**——Prometheus 采集、Grafana 仪表盘、OTel 链路追踪。
- **控制平面**——服务发现、配置管理、滚动更新。

以 Helm chart + Operator 形式发布。

### KV 卸载连接器 API（v0.9.0+）

vLLM 0.9.0 引入了连接器 API，用于可插拔的 KV 缓存后端。引擎将块卸载到连接器，连接器将其存储（RAM、磁盘、对象存储、LMCache）。请求需要某个块时，连接器将其加载回来。

vLLM 0.11.0（2026 年 1 月）新增了异步卸载路径——卸载可以在后台进行，引擎在常见情况下无需阻塞等待。端到端延迟和吞吐量仍然取决于工作负载特征、KV 缓存命中率和系统压力；vLLM 官方文档指出，自定义内核卸载在低命中率时可能降低吞吐量，且异步调度与推测解码之间存在已知的交互问题。

### 原生 CPU 卸载与 LMCache 对比

**原生 vLLM CPU 卸载**：引擎本地，将 KV 块存储在主机 RAM 中。实现简单，零网络开销，但无法跨引擎共享。

**LMCache 连接器**：集群级，将块存储在共享的 LMCache 服务器中（CPU DRAM + Ceph/S3 层）。任何引擎都可以访问这些块。已有 16 块 H100 的公开基准测试。

当单引擎面临 HBM 压力时选择原生方案。当多个引擎共享前缀（如使用相同系统提示词的 RAG、使用共享模板的多租户场景）时选择 LMCache。

### 基准测试表现

在 4 台 a3-highgpu-4g 上的 16 块 H100（80 GB HBM）测试中：

- KV 占用低（短提示词、低并发）：所有配置与基线一致，LMCache 带来约 3-5% 的开销。
- 中等占用：LMCache 开始在跨引擎前缀复用方面体现出优势。
- KV 超出 HBM：原生 CPU 卸载和 LMCache 都能显著提升吞吐量；LMCache 增益更大，因为它支持跨引擎共享。

### LMCache 决定性优势的场景

- 多租户服务中系统提示词跨租户共享。
- RAG 中文档片段在不同查询中重复出现。
- 同一基座上的微调变体（LoRA），基座模型 KV 复用可减少冗余计算。
- 抢占密集型工作负载：从 CPU 恢复比重做预填充更经济。

### 不应启用 LMCache 的场景

- HBM 压力较小——付出开销却没有收益。
- 短上下文（少于 1K token）——传输时间超过重新预填充时间。
- 单租户单提示词工作负载——没有可捕获的复用机会。

### 与分离式服务的集成

Phase 17 · 17 分离式服务与 LMCache 的组合能产生叠加效果：预填充池到解码池的 KV 传输如果未被使用，会缓存到 LMCache 中；后续查询可直接从 LMCache 获取。Phase 17 · 11 的缓存感知路由器可以将请求路由到本地缓存或 LMCache 共享缓存匹配的引擎。

### 关键数字

- vLLM 0.9.0：连接器 API 发布。
- vLLM 0.11.0（2026 年 1 月）：异步卸载路径；端到端延迟影响取决于工作负载、KV 命中率和系统压力（并非绝对保证）。
- 16 块 H100 基准测试：当 KV 占用超出 HBM 时 LMCache 有帮助。
- HBM 压力较小时：3-5% 的开销但无收益。

```figure
zero-sharding
```

## 开始使用

`code/main.py` 模拟了一个抢占密集型工作负载，分别在启用和未启用 LMCache 的情况下运行。报告避免的重新预填充次数、吞吐量提升幅度以及 HBM 利用率的盈亏平衡点。

## 部署产出

本课程生成 `outputs/skill-vllm-stack-decider.md`。根据工作负载特征和 vLLM 部署情况，决定使用原生方案、LMCache 还是两者都不使用。

## 练习

1. 运行 `code/main.py`。在什么 HBM 利用率下 LMCache 开始产生收益？
2. 一个租户跨 200 次查询/小时共享一段 6K token 的系统提示词。计算每个租户预期的 LMCache 节省量。
3. LMCache 服务器是单点故障。设计高可用策略（副本、回退到原生方案）。
4. LMCache 存储到机械硬盘上的 Ceph。对于 4K token 的 KV，70B FP8（500 MB），读取时间与重新预填充相比如何？
5. 论证 vLLM 0.11.0 的异步路径是否"免费"——开销隐藏在哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Production-stack | "参考部署方案" | vLLM 的 Kubernetes Helm chart + Operator |
| Connector API | "KV 后端接口" | vLLM 0.9.0+ 的可插拔 KV 存储接口 |
| 原生 CPU 卸载 | "引擎本地溢出" | 将 KV 存储在同一引擎的主机 RAM 中 |
| LMCache | "集群 KV 缓存" | 基于 CPU DRAM + 磁盘的跨引擎 KV 缓存服务器 |
| 0.11.0 异步 | "非阻塞卸载" | 卸载操作隐藏在引擎流之后 |
| 抢占 | "逐出以腾出空间" | HBM 满载时的 KV 缓存调度 |
| 前缀复用 | "相同的系统提示词" | 多个查询共享开头部分，产生缓存命中 |
| Ceph 层 | "磁盘层" | 缓存层级中位于 DRAM 之下的持久化存储 |

## 延伸阅读

- [vLLM 博客——KV 卸载连接器（2026 年 1 月）](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack)——Helm chart + Operator。
- [面向企业级 LLM 推理的 LMCache（arXiv:2510.09665）](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache)——连接器实现。
- [vLLM 0.11.0 发行说明](https://github.com/vllm-project/vllm/releases)——异步路径细节。
