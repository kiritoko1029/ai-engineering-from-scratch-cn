# DualPipe 并行

> DeepSeek-V3 在 2,048 块 H800 GPU 上训练，MoE 专家分散在各节点间。跨节点专家 all-to-all 通信每 1 GPU 小时计算对应 1 GPU 小时通信。GPU 一半时间在空转。DualPipe（DeepSeek，2024 年 12 月）是一种双向流水线，将前向和反向计算与它们触发的 all-to-all 通信重叠。气泡减少，吞吐上升，而保留两份模型参数副本（"Dual"名称的由来）的代价在专家并行已经将专家分散到各 rank 之后变得可以接受。本课程是学习型走读，讲解 DualPipe 实际做了什么，以及 Sea AI Lab 的 DualPipeV 改进如何以略微更大的气泡为代价去掉 2 倍参数成本。

**类型：** 学习
**语言：** Python（标准库，调度模拟器）
**前置要求：** 第 10 阶段 · 05（分布式训练、FSDP、DeepSpeed），第 10 阶段 · 14（开放模型架构和 MoE）
**所需时间：** 约60分钟

## 学习目标

- 说出 DualPipe 前向-反向块的四个组件，以及每个组件为什么有自己的重叠窗口。
- 解释规模化下的流水线气泡问题，以及"无气泡"在实践中与营销中的含义。
- 手动追踪 8 个 PP rank 和 16 个微批次的 DualPipe 调度，确认前向和反向流填充了彼此的空闲槽。
- 说出 DualPipeV（Sea AI Lab，2025）的权衡：以专家并行不活跃时略大气泡为代价去掉 2 倍参数复制。

## 问题所在

在 2k 块 H800 GPU 上训练 671B MoE 模型会遇到三个复合瓶颈：

1. **内存压力。** 每块 GPU 持有模型的一个切片。61 层、128 头、序列 8k 的激活内存巨大。
2. **流水线气泡。** 传统流水线并行（GPipe、1F1B）在 GPU 等待其阶段的输入或梯度时空闲。在 8 个阶段下，即使使用 1F1B 调度，约 12% 的 GPU 时间可能是气泡。
3. **跨节点 all-to-all。** 带专家并行的 MoE 将专家分散到各节点。每次前向传播触发一次 all-to-all 将 token 路由到其专家，另一次将结果合并。在 2k GPU 下，这很容易变成 1:1 的计算-通信比。

每个问题都有单独的解决方案：梯度检查点解决内存，Zero Bubble（Sea AI Lab，2023）解决流水线气泡，专家并行通信 kernel 解决 all-to-all。DualPipe 的作用是让它们协同工作。调度在单个前向-反向块内重叠计算和通信，从流水线两端同时注入微批次，并利用所得调度将 all-to-all 隐藏在计算窗口内。

报告结果：近乎消除流水线气泡，DeepSeek-V3 的 14.8T token 训练中 GPU 利用率超过 95%。

## 概念说明

### 流水线并行回顾

将 N 层模型分割到 P 个设备上。设备 i 持有层 `i * N/P .. (i+1) * N/P - 1`。微批次从设备 0 正向流到 P-1，然后从 P-1 反向流回 0。每个设备只能在前一个设备发送其输出后才能开始其前向阶段，只能在下游设备发送上游梯度后才能开始反向。

GPipe（Huang 等，2019）一次调度一个微批次，浪费了大部分 GPU 时间。1F1B（Narayanan 等，2021）为多个微批次交错前向和反向传播。Zero Bubble（Qi 等，2023）将反向传播分成两部分——反向求输入梯度（B）和反向求权重梯度（W）——并调度它们来填充气泡。Zero Bubble 之后，流水线几乎紧密。

DualPipe 是下一步。它在此基础上添加两个思想：

### 思想 1：块分解

每个前向块被分成四个组件：

- **注意力。** Q/K/V 投影、注意力计算、输出投影。
- **All-to-all 调度。** 将 token 发送到其专家的跨节点通信。
- **MLP。** MoE 专家计算。
- **All-to-all 合并。** 将专家输出带回的跨节点通信。

反向块添加每个组件的梯度版本。DualPipe 调度它们使 all-to-all 调度与下一个块的注意力计算并行，all-to-all 合并与后续块的 MLP 计算并行。

### 思想 2：双向调度

大多数流水线调度从阶段 0 注入微批次，流向阶段 P-1。DualPipe 从两端注入微批次。阶段 0 看到从那里起源的前向微批次；阶段 P-1 也看到从那里起源的前向微批次。两股流在中间相遇。

为此，设备 i 必须同时持有早期流水线层 i 和晚期流水线层 P - 1 - i。这就是 DualPipe 的"Dual"部分：每个设备保留它需要服务的模型层的两份副本（每个方向一份）。在 DeepSeek-V3 的规模下，这是 2 倍参数复制成本。它可以承受，因为专家并行已经将 MoE 专家分散得很薄，复制非专家层两次只是小钱。

关键的是，一个方向的前向流和另一个方向的反向流正好在单方向调度中气泡所在的位置重叠。气泡消失了。

### 手动追踪的调度

考虑 P = 4 个 rank，8 个微批次，4 个正向 / 4 个反向。时间从左到右移动；行是设备 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

读取"F4/F5R"标记：rank 1 在同一时间槽运行微批次 4 的前向（流水线中从左到右）和微批次 5 的前向（从右到左）。这就是"双向"在操作上的含义。

在 rank 2 处交叉流更早重叠，在 rank 0 和 P-1 处最晚重叠。在调度的稳定中间阶段，每个 rank 运行 X 方向的前向与 Y 方向的反向重叠。计算在忙碌。前向传播的 all-to-all 调度隐藏在反向计算中。all-to-all 合并隐藏在前向计算中。气泡被挤出。

### 气泡核算

标准 1F1B 流水线气泡（每 rank 浪费的时间）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 改进将其降低但不到零。DualPipe 在稳定阶段，如果微批次数可被 2 倍流水线深度整除，则气泡为零。在稳定阶段之外（预热和冷却），有一些气泡但不随微批次数增长——论文强调的关键属性。

营销用语："无气泡"。技术用语：气泡不随微批次数增长。Sea AI Lab 的后续分析（DualPipeV / Cut-in-half）表明，只有在专家并行不是瓶颈时才真正零气泡；在 EP 驱动的 all-to-all 下，总有一些调度妥协。

### DualPipeV——改进版

Sea AI Lab（2025）观察到，当 EP 通信重叠不是重点时，2 倍参数复制是浪费的。他们的 DualPipeV 调度将双向注入折叠为在单参数副本上运行的"V 形"调度。气泡比 DualPipe 略大，但内存节省显著。DeepSeek 在其开源 DualPipe 实现中采用 DualPipeV 作为 EP-off 模式。

权衡：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|------|---------|-----------|------|------------|
| 每设备参数副本 | 2 | 1 | 1 | 1 |
| 气泡与微批次关系 | 恒定 | 小幅增长 | 增长 | 增长 |
| 计算-通信重叠 | 完全 | 部分 | 最小 | 部分 |
| 使用场景 | EP 密集 MoE | 稠密或 EP 轻量 | 基线 | 任意流水线 |

### 对 14.8T token 运行的意义

DeepSeek-V3 的预训练在 2,048 块 H800 GPU 上消耗了 14.8T token，大约 280 万 GPU 小时。使用朴素 1F1B，他们会损失 12-15% 到流水线气泡——34-42 万 GPU 小时，足够训练一个完整的 70B 模型。DualPipe 回收了其中大部分。没有内部日志很难直接量化贡献，但论文声称训练期间平均 GPU 利用率超过 95%。

对于较小的运行（1k GPU 以下），DualPipe 是过度设计——流水线气泡相对于总成本更小，稠密模型训练很少遇到 all-to-all 瓶颈。对于多千 GPU 规模的前沿 MoE 训练，它实际上是必需的。

### 在栈中的位置

- 与 **FSDP**（第 10 阶段 · 05）互补。FSDP 跨 rank 切分模型参数；DualPipe 跨 rank 调度计算。两者结合。
- 兼容 **ZeRO-3** 梯度切分。两份副本复制的记账需要与 ZeRO 的切分梯度协作。
- 需要针对特定集群拓扑调优的**自定义 all-to-all kernel**。DeepSeek 的开源 kernel 是参考实现。

```figure
expert-capacity
```

## 使用它

`code/main.py` 是流水线调度模拟器。它接收 `(P, n_micro_batches, schedule)` 并打印 1F1B、Zero Bubble、DualPipe 和 DualPipeV 的稳定阶段利用率。它是教学工具——数字与论文中的定性声明匹配，不是对生产实测加速的声明。

模拟器的价值：用不同的 P 和微批次运行它，观察气泡分数对 1F1B 增长但对 DualPipe 不增长。

真实训练运行的集成考量：

- 选择能被微批次数整除的流水线并行深度。
- 确保你的专家并行网格支持双向 all-to-all。DeepSeek 的 kernel 是参考。
- 第一次调试调度本身预计要花一周时间。记账很繁琐。
- 监控每 rank 的 GPU 利用率，不仅仅是聚合。DualPipe 的收益来自收紧尾部。

## 交付它

本课程产出 `outputs/skill-dualpipe-planner.md`。给定训练集群规范（GPU 数量、拓扑、互联、模型形状），它推荐流水线并行策略、使用的调度算法和目标规模下的预期气泡分数。

## 练习

1. 在 `(P=8, micro_batches=16, schedule=dualpipe)` 和 `(P=8, micro_batches=16, schedule=1f1b)` 上运行 `code/main.py`。计算 GPU 利用率差异并表示为每百万 token 训练回收的 GPU 小时。

2. 手动绘制 `(P=4, micro_batches=8, schedule=dualpipe)` 的调度表。用微批次 ID 和方向标记每个时间槽。找出气泡消失的第一个时间槽。

3. 阅读 DeepSeek-V3 技术报告（arXiv:2412.19437）的图 5。找出 DualPipe 前向块内 all-to-all 调度的重叠窗口。解释计算调度如何将其隐藏。

4. 计算 DualPipe 对 P=8 流水线阶段的 70B 稠密模型和 P=16 流水线阶段的 671B MoE 模型的 2 倍参数开销。展示为什么 MoE 情况的开销按比例更小（大部分参数是专家，分散在大 EP 组中）。

5. 将 DualPipe 与 Chimera（2021 年的竞争性双向调度器）比较。使用论文第 3.4 节作为参考，找出 DualPipe 添加的两个 Chimera 没有的具体属性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 流水线气泡 | "每 rank 空闲时间" | 流水线阶段等待输入或梯度浪费的 GPU 周期 |
| 1F1B | "默认流水线调度" | 一个前向 / 一个反向交错调度；DualPipe 击败的基线 |
| Zero Bubble | "Sea AI Lab 2023" | 将反向分成 B（输入梯度）和 W（权重梯度）；几乎完全收紧流水线 |
| DualPipe | "DeepSeek-V3 调度" | 双向流水线 + 计算-通信重叠；气泡不随微批次数增长 |
| DualPipeV | "Cut-in-half" | 以略大气泡为代价去掉 2 倍参数复制的 V 形改进 |
| 块 | "流水线工作单元" | 一个微批次通过一个流水线阶段的一次前向或反向传播 |
| All-to-all 调度 | "将 token 发送给专家" | 将 token 路由到其分配的 MoE 专家的跨节点通信 |
| All-to-all 合并 | "将专家输出带回" | MLP 后收集专家输出的跨节点通信 |
| 专家并行（EP） | "专家分散在 GPU 上" | 将 MoE 专家切分到各 rank，不同 GPU 持有不同专家 |
| 流水线并行（PP） | "层分散在 GPU 上" | 将模型层切分到各 rank；DualPipe 调度的维度 |
| 气泡分数 | "浪费的 GPU 时间" | (气泡时间 / 总时间)；DualPipe 将其推向零的分数 |

## 延伸阅读

- [DeepSeek-AI -- DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437) -- DualPipe 的主要参考
- [DeepSeek -- DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe) -- 开源参考实现，包括 DualPipeV（Cut-in-half）模式
- [Qi et al. -- Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241) -- Zero Bubble 前身
- [Sea AI Lab -- DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63) -- 为 DeepSeek 的 EP-off 模式提供信息的 DualPipeV 分析
- [Narayanan et al. -- PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377) -- DualPipe 比较的 1F1B 调度
- [Huang et al. -- GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965) -- 原始流水线并行论文和气泡问题
