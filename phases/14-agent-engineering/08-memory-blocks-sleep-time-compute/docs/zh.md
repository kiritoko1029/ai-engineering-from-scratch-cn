# 记忆块与睡眠时间计算（Letta）

> MemGPT 在 2024 年变成了 Letta。2026 年的演进增加了两个想法：模型可以直接编辑的离散功能记忆块，以及在主智能体空闲时异步合并记忆的睡眠时间智能体。这是你将记忆扩展到单次对话之外的方式。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 07（MemGPT）
**所需时间：** 约75分钟

## 学习目标

- 说出 Letta 使用的三个记忆层（core、recall、archival）及其各自的角色。
- 解释记忆块模式：Human 块、Persona 块和用户自定义块作为一等类型化对象。
- 描述什么是睡眠时间计算、为什么它不在关键路径上、以及为什么它可以运行比主智能体更强的模型。
- 实现一个脚本化的双智能体循环，其中主智能体响应请求，睡眠时间智能体在轮次之间合并块。

## 问题所在

MemGPT（第 07 课）解决了虚拟内存控制流。三个生产问题浮现：

1. **延迟。** 每个记忆操作都在关键路径上。如果智能体在用户等待时必须修剪、摘要或协调，尾部延迟会爆炸。
2. **记忆腐化。** 写入不断积累。被矛盾的事实留存。检索淹没在陈旧内容中。
3. **结构丢失。** 扁平的档案存储无法表达"Human 块始终在提示中；Persona 块始终在提示中；Task 块按会话切换。"

Letta（letta.com）是 2026 年的重写。记忆块使结构显式化；睡眠时间计算将合并移出关键路径。

## 概念说明

### 三层结构

| 层 | 范围 | 存储位置 | 写入者 |
|----|------|---------|--------|
| Core | 始终可见 | 主提示内 | 智能体工具调用 + 睡眠时间重写 |
| Recall | 对话历史 | 可检索 | 自动轮次日志 |
| Archival | 任意事实 | 向量 + KV + 图 | 智能体工具调用 + 睡眠时间摄入 |

Core 是 MemGPT 的核心。Recall 是带淘汰尾部的对话缓冲区。Archival 是外部存储。这种划分清理了 MemGPT 两层的重载。

### 记忆块

块是 Core 层的类型化、持久、可编辑区段。原 MemGPT 论文定义了两个：

- **Human 块**——关于用户的事实（姓名、角色、偏好、目标）。
- **Persona 块**——智能体的自我概念（身份、语气、约束）。

Letta 泛化为任意用户自定义块：`Task` 块用于当前目标、`Project` 块用于代码库事实、`Safety` 块用于硬约束。每个块有 `id`、`label`、`value`、`limit`（字符上限）、`description`（让模型知道何时编辑它）。

块通过工具接口可编辑：

- `block_append(label, text)`
- `block_replace(label, old, new)`
- `block_read(label)`
- `block_summarize(label)`——压缩接近上限的块。

### 睡眠时间计算

2025 年 Letta 新增：在后台、非关键路径上运行第二个智能体。睡眠时间智能体处理对话记录和代码库上下文，将 `learned_context` 写入共享块，并合并或失效档案记录。

由此产生的特性：

- **无延迟成本。** 主响应不等待记忆操作。
- **允许使用更强的模型。** 睡眠时间智能体可以是更昂贵、更慢的模型，因为它不受延迟约束。
- **自然的合并窗口。** 在用户不等待时去重、摘要、失效被矛盾的事实。

这种形态匹配人类的工作方式：你做任务，你睡一觉，长期记忆在夜间沉淀。

### Letta V1 与原生推理

Letta V1（`letta_v1_agent`，2026）弃用了 `send_message`/心跳和内联 `Thought:` token，转向原生推理。Responses API（OpenAI）和带扩展思考的 Messages API（Anthropic）在独立通道上发出推理，跨轮传递（生产环境中跨提供商加密）。控制循环仍然是 ReAct。思考轨迹是结构性的，而非提示式的。

### 此模式出错的地方

- **块膨胀。** 无限 `block_append` 很快触及上限。在写入超出上限前接入块摘要器。
- **静默漂移。** 睡眠时间智能体重写块而主智能体从未注意到。对块进行版本控制，在轨迹中呈现差异。
- **投毒合并。** 睡眠时间智能体将攻击者可达的内容处理到 Core 中。第 27 课也适用于睡眠时间接口。

## 开始构建

`code/main.py` 实现了：

- `Block`——id、label、value、limit、description。
- `BlockStore`——CRUD + `near_limit(label)` 辅助函数。
- 两个脚本化智能体——`PrimaryAgent` 响应一轮，`SleepTimeAgent` 在轮次之间合并。
- 一个轨迹，展示三轮对话带块写入，加上一次睡眠时间遍历——摘要一个块并失效一个陈旧事实。

运行它：

```
python3 code/main.py
```

记录展示了分离：主轮次快速产出原始写入；睡眠遍历压缩并清理。

## 使用它

- **Letta**（letta.com）作为参考实现。自托管或托管云。
- **Claude Agent SDK 技能**作为块形态的知识——技能是智能体按需加载的命名、版本化、可检索的指令块。
- **自定义构建**适用于想要控制存储后端的团队。使用 Letta API 契约以便日后迁移。

## 交付它

`outputs/skill-memory-blocks.md` 为任何运行时生成 Letta 形态的块系统，带睡眠时间钩子，包含安全规则和引用接线。

## 练习

1. 添加 `block_summarize` 工具，当 `near_limit` 返回 true 时用模型生成的摘要替换块值。哪个触发阈值能最小化摘要调用和块溢出？
2. 在档案上实现睡眠时间去重：文本 token 重叠 >90% 的两条记录合并为一条。仅在睡眠遍历中执行，不在关键路径上。
3. 对块进行版本控制。每次写入记录旧值和差异。暴露 `block_history(label)` 以便运维调试"为什么智能体忘记了 X"。
4. 将睡眠时间智能体视为不可信的写入者。当它们触及 Persona 或 Safety 块时，在提交前要求第二个智能体审核。
5. 将示例移植到使用 Letta API（`letta_v1_agent`）。块 schema 有什么变化，原生推理如何改变轨迹形态？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 记忆块（Memory block） | "可编辑的提示区段" | Core 记忆的类型化、持久、LLM 可编辑区段 |
| Human 块 | "用户记忆" | 关于用户的事实，固定在 Core 中 |
| Persona 块 | "智能体身份" | 自我概念、语气、约束，固定在 Core 中 |
| 睡眠时间计算（Sleep-time compute） | "异步记忆工作" | 非关键路径上的第二个智能体做合并 |
| Core / Recall / Archival | "层" | 三层记忆划分：始终可见 / 对话 / 外部 |
| 块上限（Block limit） | "上限" | 每个块的字符限制；强制摘要 |
| 原生推理（Native reasoning） | "思考通道" | 提供商级别的推理输出，非提示级 `Thought:` |
| 学习上下文（Learned context） | "睡眠输出" | 睡眠时间智能体写入共享块的事实 |

## 延伸阅读

- [Letta，Memory Blocks 博客](https://www.letta.com/blog/memory-blocks)——块模式
- [Letta，Sleep-time Compute 博客](https://www.letta.com/blog/sleep-time-compute)——异步合并
- [Letta，Rearchitecting the Agent Loop](https://www.letta.com/blog/letta-v1-agent)——原生推理重写
- [Packer 等人，MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)——起源
