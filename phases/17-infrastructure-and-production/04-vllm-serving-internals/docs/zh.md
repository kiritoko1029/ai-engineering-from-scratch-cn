# vLLM 服务内部机制：PagedAttention、Continuous Batching、Chunked Prefill

> vLLM 在 2026 年的主导地位建立在三个叠加的默认配置之上，而非单一技巧。PagedAttention 始终开启。Continuous Batching 在解码迭代之间将新请求注入活跃批次。Chunked Prefill 将长提示切片，使解码 token 永远不会被饿死。三者同时开启时，一个 H100 SXM5 上的 Llama 3.3 70B FP8 可以在 128 并发下达到 2,200-2,400 tok/s — 大约比 vLLM 自身默认值高 25%，比朴素 PyTorch 循环高 3-4 倍。本课在可以画图的层面上阅读调度器和注意力内核，并以 `code/main.py` 中一个简易的 Continuous Batcher 结束，该调度器以 vLLM 的方式调度预填充和解码。

**类型：** 学习
**语言：** Python（标准库，简易 Continuous Batching 调度器）
**前置要求：** Phase 17 · 01（模型服务），Phase 11（LLM 工程）
**所需时间：** 约75分钟

## 学习目标

- 将 PagedAttention 解释为 KV 缓存分配器：块、块表，以及为什么在生产负载下碎片率保持在 4% 以下。
- 在迭代层面画出 Continuous Batching：已完成的序列如何离开批次，新序列如何加入而不排空。
- 用一句话描述 Chunked Prefill，并说出它保护的延迟指标（提示：是 TTFT 尾部，而非平均吞吐量）。
- 说出 2026 年 vLLM v0.18.0 中同时启用所有优化时会踩到的坑。

## 问题所在

朴素的 PyTorch 服务循环一次处理一个请求：分词、预填充、解码直到 EOS、返回。一个用户时这没问题。一百个用户时，这就是一队耐心等待的人。显而易见的解决方案 — 静态批处理 — 将每个请求填充到窗口中最长提示的长度，将每次解码填充到最长预期输出的长度，并在最慢的序列上阻塞整个批次。你为从未使用的填充付费，快速请求等待慢速请求。

vLLM 同时解决了三个问题。PagedAttention 防止 KV 缓存碎片像经典连续分配那样消耗 60-80% 的 GPU 内存。Continuous Batching 允许请求在每次解码迭代之间加入和离开批次，因此批次始终充满真实工作。Chunked Prefill 将 32k token 的提示切分为约 512 token 的切片，与解码交错，因此长提示不会冻结 GPU 上的每个解码 token。

2026 年的生产默认是三者全部开启。你需要理解每个的功能，因为故障模式全在调度器上，而非模型上。

## 概念说明

### PagedAttention 作为虚拟内存系统

KV 缓存是每个序列 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。对于 Llama 3.3 70B 在 8192 token 下，BF16 约为每个序列 1.25 GB。如果你为每个请求预保留 8192 个槽位，但平均请求只使用 1500 个 token，你浪费了约 82% 预留的 HBM。经典批处理付出这种浪费。

PagedAttention 从操作系统虚拟内存中借鉴了这个想法。KV 缓存不是每个序列连续的。它以固定大小的块（默认 16 个 token）分配。每个序列有一个块表，将其逻辑 token 位置映射到物理块 ID。当序列增长超过已分配的块时，再添加一个块。当它完成时，其块返回到池中。

碎片率从 60-80%（经典）降至 4% 以下（PagedAttention）。你不需要通过标志启用 PagedAttention — 它是 vLLM 唯一的分配器。调节旋钮是 `--gpu-memory-utilization`（默认 0.9），它告诉 vLLM 在加载权重和激活后为 KV 块预留多少 HBM。

### 迭代层面的 Continuous Batching

旧的"动态批处理"等待一个窗口（比如 10 毫秒）来填充批次，然后运行预填充 + 解码 + 解码 + 解码直到每个序列完成。快速序列提前离开，在 GPU 完成慢速序列时空闲。

Continuous Batching 在每个解码步骤之间操作。将正在运行的序列集合称为 `RUNNING` 列表。每次迭代：

1. `RUNNING` 中刚刚触发 EOS 或 max_tokens 的序列被移除。
2. 调度器查看等待队列。如果有空闲 KV 块，它接纳新序列（预填充或恢复的）。
3. 前向传播对 `RUNNING` 中的所有内容运行，每个序列发出一个新 token。

批次大小永远不会填充到固定数字。输出中不同位置的序列共享一个融合前向传播。2026 年的 vLLM 中这被称为 `V1 scheduler`。关键不变量：调度器每个解码迭代运行一次，而非每个请求一次。

### Chunked Prefill 保护 TTFT 尾部

预填充是计算密集型的。Llama 3.3 70B 上的 32k token 提示在一个 H100 上需要约 800 毫秒的纯预填充。在预填充运行时，批次中所有其他序列的解码 token 等待。在服务循环中，一个长提示的首 token 延迟（TTFT）成为其他数十个用户的 token 间延迟（ITL）尖峰。

Chunked Prefill 将预填充切分为固定大小的块（默认 512 个 token），并将每个块作为一个单元调度。在块之间，调度器可以将解码序列推进一个 token。你用较小的绝对预填充延迟损失（每块几毫秒）换取更低的解码时间抖动。在混合负载下，P99 ITL 从约 50 毫秒降至约 15 毫秒（已发布的基准测试数据）。

### 三个默认配置相互作用

三个功能都假设彼此存在。PagedAttention 为调度器提供了细粒度的 KV 资源来进行权衡。Continuous Batching 需要这种细粒度资源，以便接纳新序列时不会强制全局重排。Chunked Prefill 是调度器在同一个 `RUNNING` 列表上做出的决策 — 它是又一个调度器策略，而非独立系统。

你不需要知道每个标志。你需要知道调度器优化什么：在 KV 块预算下的 goodput，受 Chunked Prefill 切片约束。

### 2026 年 v0.18.0 的坑

在 vLLM v0.18.0 中，你不能将 `--enable-chunked-prefill` 与草稿模型推测解码（`--speculative-model`）组合使用。文档中记录的例外是 V1 调度器中的 N-gram GPU 推测解码。那些不看发布说明就打开所有标志的团队会在启动时遇到运行时错误，而非软性回退。如果你的推测增益值得启用 Chunked Prefill，重新审视这个选择 — 2026 年的正确答案通常是不带 Chunked Prefill 的 EAGLE-3，而非无法编译的草稿模型加 Chunked Prefill。

### 你应该记住的数字

- Llama 3.3 70B FP8，H100 SXM5，128 并发，三者全开：2,200-2,400 tok/s。
- 同一模型，默认 vLLM（无 Chunked Prefill）：约 1,800 tok/s。
- 同一模型，朴素 PyTorch 前向循环：约 600 tok/s。
- 生产负载下 PagedAttention 的 KV 碎片浪费：<4%。
- 混合负载下 P99 ITL：Chunked Prefill 开启约 15 毫秒，关闭约 50 毫秒。

### 调度器长什么样

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是这个循环的 Python 标准库实现，使用假 token 计数和假前向延迟。运行它可以看到 Chunked Prefill 如何在长预填充期间保持解码序列活跃。

```figure
tensor-parallel
```

## 开始使用

`code/main.py` 模拟了一个 vLLM 风格的调度器，支持可切换功能。运行它可以看到：

- `NAIVE` 模式：一次一个请求，无批处理。
- `STATIC` 模式：填充并等待，经典批处理。
- `CONTINUOUS` 模式：迭代级接纳和释放。
- `CONTINUOUS + CHUNKED` 模式：预填充切片与解码交错。

输出显示总吞吐量（每虚拟秒 token 数）、TTFT 平均值和 P99 ITL。`CONTINUOUS + CHUNKED` 行在混合流量上应该占优。

## 交付成果

本课生成 `outputs/skill-vllm-scheduler-reader.md`。给定服务配置（批次大小、KV 内存利用率、Chunked Prefill 大小、推测配置），生成一个调度器诊断，指出三个默认配置中哪个是瓶颈以及该调整什么。

## 练习

1. 运行 `code/main.py`。在混合长短请求的工作负载上比较 `STATIC` 和 `CONTINUOUS`。吞吐量差距来自哪里 — 预填充效率、解码效率还是尾部延迟？
2. 修改简易调度器以添加 `--max-num-batched-tokens`。对于在 H100 上运行 Llama 3.3 70B FP8，正确的值是多少？（提示：它是 KV 块大小和空闲块数量的函数，而非原始 HBM。）
3. 重新阅读 vLLM v0.18.0 发布说明。哪些标志组合是互斥的？列出它们。
4. 计算 1000 个请求的轨迹的 KV 缓存碎片浪费，平均 1500 个输出 token，标准差 600 个 token，分别在（a）8192 最大值的连续每请求分配和（b）16 token 块的 PagedAttention 下。
5. 用一段话解释为什么 Chunked Prefill 有助于 P99 ITL 但单独来看不影响吞吐量。实际上吞吐量提升来自哪里？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| PagedAttention | "KV 技巧" | KV 缓存的固定大小块分配器；碎片率 <4% |
| 块表 | "页表" | 每个序列从逻辑 token 位置到物理 KV 块的映射 |
| Continuous Batching | "动态批处理，但做对了" | 每次解码迭代做出接纳/释放决策 |
| Chunked Prefill | "预填充切片" | 将长预填充切分为 512 token 切片与解码交错 |
| TTFT | "首 token 时间" | 预填充 + 队列 + 网络；长提示时由预填充主导 |
| ITL | "token 间延迟" | 连续解码 token 之间的时间；由批次大小主导 |
| Goodput | "满足 SLO 的吞吐量" | 每秒 token 数，其中每个请求仍达到 TTFT 和 ITL 目标 |
| V1 scheduler | "新调度器" | vLLM 的 2026 调度器；N-gram 推测解码是 Chunked Prefill 兼容路径 |
| `--gpu-memory-utilization` | "内存旋钮" | 加载权重和激活后为 KV 块预留的 HBM 比例 |

## 延伸阅读

- [vLLM 文档 — 推测解码](https://docs.vllm.ai/en/latest/features/spec_decode/) — Chunked Prefill 和推测解码兼容性的官方来源。
- [vLLM 发布说明（NVIDIA）](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html) — 2026 年发布节奏和版本特定行为。
- [vLLM 博客 — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html) — 仍然定义如何思考分配器的原始文章。
- [PagedAttention 论文（arXiv:2309.06180）](https://arxiv.org/abs/2309.06180) — 碎片分析和调度器设计。
- [Aleksa Gordic — vLLM 内部](https://www.aleksagordic.com/blog/vllm) — 详细的 V1 调度器演练和火焰图。
