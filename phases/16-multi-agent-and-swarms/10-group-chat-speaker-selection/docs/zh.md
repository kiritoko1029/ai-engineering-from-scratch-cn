# 群聊与发言者选择

> AutoGen GroupChat 和 AG2 GroupChat 让 N 个代理共享一个对话空间；一个选择器函数（LLM、轮询或自定义）决定谁下一个发言。这是涌现式多智能体对话的原型——代理不知道自己在静态图中的角色，只是对共享消息池做出反应。AutoGen v0.2 的 GroupChat 语义在 AG2 分支中得以保留；AutoGen v0.4 将其重写为事件驱动的 Actor 模型。微软于 2026 年 2 月将 AutoGen 设为维护模式，并将其与 Semantic Kernel 合并为 Microsoft Agent Framework（2026 年 2 月 RC 版）。GroupChat 原语在 AG2 和 Microsoft Agent Framework 中都得以延续——学一次，到处用。

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 第 16 阶段 · 第 04 课（基础模型）
**所需时间：** 约 60 分钟

## 问题所在

当工作流已知时，静态图（LangGraph）非常合适。但真实的对话不是静态的：有时程序员问审查员，有时问研究员，有时问撰稿人。将每种可能的交接都硬编码会导致边的数量爆炸。你需要的是*代理对共享消息池做出反应*，由某个函数决定谁下一个发言。

这正是 AutoGen GroupChat 的功能。

## 概念说明

### 结构形态

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

每个代理都能看到所有消息。每轮会调用一个选择器函数来决定谁下一个发言。

### 三种选择器风格

**轮询式。** 固定循环。确定性。线性扩展但忽略上下文——当话题是法务审查时，程序员仍然会轮到发言。

**LLM 选择式。** 调用一个 LLM，读取最近的消息池，返回最佳的下一个发言者。上下文感知但速度慢：每轮都增加一次 LLM 调用。这是 AutoGen 的默认方式。

**自定义。** 一个包含任意逻辑的 Python 函数。典型用法：LLM 选择式加上回退规则（例如"程序员发言后总是轮到验证者"）。

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` 持有选择器。当一个代理完成一轮发言时，管理器调用选择器，选择器返回下一个代理。循环持续直到触发终止条件。

### 终止条件

三种常见模式：

- **最大轮数。** 对总轮数的硬性上限。
- **"TERMINATE" 标记。** 代理可以发出一个哨兵消息；管理器在看到它时停止。
- **目标达成检查。** 每轮运行一个轻量级验证器，完成后停止对话。

### AutoGen 到 AG2 的分裂与 Microsoft Agent Framework 的合并

2025 年初，微软开始对 AutoGen（v0.4）进行重大重写，转向事件驱动的 Actor 模型。社区将 AutoGen v0.2 的 GroupChat 语义分支为 AG2，保留了早期集成者所依赖的 API。

2026 年 2 月，微软宣布 AutoGen 进入维护模式，事件驱动的 Actor 模型合并入 **Microsoft Agent Framework**（2026 年 2 月 RC 版，已与 Semantic Kernel 合并）。GroupChat 概念在两条路线中都得以延续；实现细节有所不同。AG2 是 v0.2 兼容代码的首选上游。

### 适用场景

- **涌现式对话。** 你不想预先连接所有可能的下一位发言者。
- **角色交叉任务。** 程序员问研究员，研究员问档案管理员，档案管理员再问程序员。流程不是 DAG。
- **探索性问题解决。** 想象"头脑风暴会议"，而非"流水线"。

### 不适用场景

- **严格确定性。** LLM 选择器可能不一致。相同的提示，不同的运行，产生不同的下一位发言者。
- **谄媚级联。** 代理服从发言最自信的那个。需要通过提示明确设置对抗角色。
- **上下文膨胀。** 每个代理读取所有消息；10 轮后上下文会变得很大。使用投影（第 15 课）来限定视图范围。
- **热门发言者。** 一个代理主导对话，因为选择器偏好其专业领域。将发言者均衡性引入选择器特征。

### 群聊与监督者的对比

相同的原语，不同的默认方式：

- 监督者：一个代理规划，其他代理执行。选择器是"问规划者该做什么"。
- 群聊：所有代理是对等的；选择器是基于共享消息池的函数。

两者都使用第 04 课的四个原语。群聊默认使用 LLM 选择式编排和全池共享状态。

## 开始构建

`code/main.py` 使用标准库从零实现了一个 GroupChat。三个代理（程序员、审查员、管理者），轮询式和 LLM 选择式两种变体，以及基于 `TERMINATE` 标记的终止机制。

演示打印了对话记录以及两种变体的选择器决策追踪。

运行：

```
python3 code/main.py
```

## 实际应用

`outputs/skill-groupchat-selector.md` 为给定任务配置 GroupChat 选择器——轮询式 vs LLM 选择式 vs 自定义，以及使用哪些选择器输入（最近消息、代理专业领域、轮次计数）。

## 投入生产

检查清单：

- **最大轮数上限。** 必须设置。典型任务为 10-20 轮。
- **发言者均衡指标。** 跟踪每个代理的轮次；当不均衡超过阈值时告警。
- **终止标记。** `TERMINATE` 或一个专门的验证者代理。
- **投影或作用域内存。** 超过约 10 条消息后，考虑给每个代理一个限定的视图，防止上下文膨胀。
- **选择器日志。** 对于 LLM 选择式变体，记录选择器的输入和选择结果。否则调试无从谈起。

## 练习

1. 运行 `code/main.py`。对比轮询式和 LLM 选择式下的对话。哪种方式下哪个代理占主导？
2. 在选择器中添加"每个代理最大发言次数"规则。这对对话记录有什么影响？
3. 实现目标达成终止：当审查员返回"已批准"时停止。它在轮次上限之前触发的频率有多高？
4. 阅读 AutoGen 稳定版的 GroupChat 文档（https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html）。找出 `GroupChatManager` 使用的默认选择器。
5. 阅读 AG2 仓库（https://github.com/ag2ai/ag2），比较其 v0.2 GroupChat 与 v0.4 事件驱动版本。v0.4 增加了什么具体特性（吞吐量、容错性、可组合性）？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| GroupChat | "代理在一个聊天室" | 共享消息池 + 选择器函数。AutoGen / AG2 原语。 |
| 发言者选择 | "谁下一个说话" | 选择下一个代理的函数。轮询式、LLM 选择式或自定义。 |
| GroupChatManager | "会议主持人" | AutoGen 组件，持有选择器并循环执行各轮次。 |
| ConversableAgent | "基础代理" | AutoGen 基类；可以发送和接收消息的代理。 |
| 终止标记 | "停止词" | 结束对话的哨兵字符串（通常是 `TERMINATE`）。 |
| 热门发言者 | "一个代理独占话语权" | 选择器持续选择同一个代理的失败模式。 |
| 上下文膨胀 | "消息池无限增长" | 每个代理读取所有先前消息；上下文随轮次增长。 |
| 投影 | "限定视图" | 针对特定角色的共享池视图，用于防止上下文膨胀。 |

## 延伸阅读

- [AutoGen 群聊文档](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — 参考实现
- [AG2 仓库](https://github.com/ag2ai/ag2) — 社区对 AutoGen v0.2 的延续
- [Microsoft Agent Framework 文档](https://microsoft.github.io/agent-framework/) — 合并后的继任者，2026 年 2 月 RC 版
- [AutoGen v0.4 发布说明](https://microsoft.github.io/autogen/stable/) — 事件驱动 Actor 模型重写的详细信息
