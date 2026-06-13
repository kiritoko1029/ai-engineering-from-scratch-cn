# Reflexion：语言强化学习

> 基于梯度的强化学习需要数千次试验和一个 GPU 集群来修复一个失败模式。Reflexion（Shinn 等人，NeurIPS 2023）用自然语言做到了这一点：每次试验失败后，智能体写一段反思，存入情景记忆，并在下一次试验中以该记忆为条件。这就是 Letta 的睡眠时间计算、Claude Code 的 CLAUDE.md 学习记录和 pro-workflow 的 learn-rule 背后的模式。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）、第 14 阶段 · 02（ReWOO）
**所需时间：** 约60分钟

## 学习目标

- 说出 Reflexion 的三个组件（Actor、Evaluator、Self-Reflector）以及情景记忆的作用。
- 用标准库实现一个包含二元评估器、反思缓冲区和全新重试的 Reflexion 循环。
- 为给定任务选择标量、启发式和自评估反馈源。
- 解释为什么语言强化能捕获基于梯度的强化学习需要数千次试验才能修复的错误。

## 问题所在

一个智能体任务失败了。在标准强化学习中，你需要运行数千次更多试验，计算梯度，更新权重。昂贵、缓慢，而且大多数生产智能体没有为每次失败准备训练预算。

Reflexion（Shinn 等人，arXiv:2303.11366）提出了一个不同的问题：如果智能体只是思考一下为什么失败了，然后带着这个思考重试呢？没有权重更新。没有梯度。只有存储在试验之间的自然语言。

结果：在 ALFWorld 上它击败了 ReAct 和其他非微调基线。在 HotpotQA 上它超越了 ReAct。在代码生成（HumanEval/MBPP）上它在当时达到了最优水平。全程没有一个梯度步骤。

## 概念说明

### 三个组件

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

加上一个数据结构：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次试验运行 Actor。Evaluator 评分。如果分数低，Self-Reflector 产出一段反思（"我选错了工具，因为我把问题误读为询问 X，而实际上是在问 Y"）。反思进入情景记忆。下一次试验重新开始，但能看到反思。

### 三种评估器类型

1. **标量**——外部二元信号。ALFWorld 成功或失败。HumanEval 测试通过或失败。最简单，信号最强。
2. **启发式**——预定义的失败特征。"如果智能体连续两次产生相同动作，标记为卡住。""如果轨迹超过 50 步，标记为低效。"
3. **自评估**——LLM 给自己的轨迹打分。在没有真值时需要。信号较弱；与工具锚定验证配合良好（第 05 课——CRITIC）。

2026 年的默认做法是混合使用：有标量时用标量，没有时用自评估，启发式作为安全护栏。

### 为什么可以泛化

Reflexion 不是一种新算法，更像是一个命名模式。几乎每个生产中的"自愈"智能体都运行某种变体：

- Letta 的睡眠时间计算（第 08 课）：一个独立的智能体反思过去的对话并写入记忆块。
- Claude Code 的 `CLAUDE.md` / "保存记忆"模式：反思作为学习记录被捕获，添加到未来的会话中。
- pro-workflow 的 `/learn-rule` 命令：纠错作为显式规则被捕获。
- LangGraph 的反思节点：一个评分输出并在需要时路由到优化的节点。

都源于同一个洞见：自然语言是一种足够丰富的媒介，可以在运行之间传递"我从失败中学到了什么"。

### 何时有效，何时无效

Reflexion 在以下情况有效：

- 有清晰的失败信号（测试失败、工具错误、错误答案）。
- 任务类别可复现（同类问题可以再次被问到）。
- 反思有空间改进轨迹（足够的行动预算）。

Reflexion 在以下情况无效：

- 智能体首次尝试就成功了。
- 失败是外部的（网络中断、工具损坏）——反思"网络断了"对未来的运行没有帮助。
- 反思变成了迷信——存储关于一次偶发运行的叙述。

2026 年陷阱：记忆腐化。反思不断积累；有些已过时或错误；随着情景缓冲区增长，重运行变慢。缓解措施：定期压缩（第 06 课）、反思的 TTL，或独立的睡眠时间清理智能体（Letta）。

```figure
react-trace
```

## 开始构建

`code/main.py` 在一个简易谜题上实现了 Reflexion：生成一个 3 元素列表使其和为目标值。Actor 发出候选列表；Evaluator 检查和；Self-Reflector 写一行关于出错原因的诊断。反思进入情景记忆供下次试验使用。

组件：

- `Actor`——一个在看到反思时会改进的脚本化策略。
- `Evaluator.binary()`——对目标和的通过/失败判定。
- `SelfReflector`——生成一行失败诊断。
- `EpisodicMemory`——带 TTL 语义的有界列表。

运行它：

```
python3 code/main.py
```

轨迹显示三次试验。试验 1 失败，存储反思，试验 2 看到反思并改进但仍失败，试验 3 成功。与基线运行（无反思）对比——它始终停留在试验 1 的答案上。

## 使用它

LangGraph 将反思作为节点模式提供。Claude Code 的 `/memory` 命令和 pro-workflow 的 `/learn-rule` 将情景缓冲区外部化为 markdown 文件。Letta 的睡眠时间计算在空闲时运行 Self-Reflector，使主智能体保持低延迟。OpenAI Agents SDK 没有直接提供 Reflexion；你可以用自定义 Guardrail（按分数拒绝轨迹）和跨运行持久化的记忆 `Session` 来构建它。

## 交付它

`outputs/skill-reflexion-buffer.md` 创建并维护一个情景缓冲区，具备反思捕获、TTL 和去重功能。给定任务类别和失败，它产出一段真正有助于下一次试验的反思（而非泛泛的"更仔细点"）。

## 练习

1. 从二元评估器切换到返回距离度量（离目标多远）的标量评估器。收敛更快吗？
2. 给反思添加 10 次试验的 TTL。超过该时间后，旧反思是有害还是有益？
3. 实现启发式评估器：如果相同动作重复则标记试验为卡住。这与 Self-Reflector 如何交互？
4. 用忽略反思的对抗性 Actor 运行 Reflexion。迫使 Actor 注意反思的最小提示工程是什么？
5. 阅读 Reflexion 论文关于 AlfWorld 的第 4 节。概念性地重现 130% 成功率提升：相对于标准 ReAct 的关键差异是什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Reflexion | "自我纠错" | Shinn 等人 2023 年——Actor、Evaluator、Self-Reflector 加情景记忆 |
| 语言强化（Verbal reinforcement） | "无梯度学习" | 自然语言反思添加到下一次试验的提示中 |
| 情景记忆（Episodic memory） | "按任务的反思" | 单个任务类别的前置反思的有界缓冲区 |
| 标量评估器（Scalar evaluator） | "二元成功信号" | 来自真值的通过/失败或数值分数 |
| 启发式评估器（Heuristic evaluator） | "基于模式的检测器" | 预定义的失败特征（如卡循环、步数过多） |
| 自评估器（Self-evaluator） | "LLM 对自己轨迹的评判" | 无真值时的低信号后备——与工具锚定验证配合使用 |
| 记忆腐化（Memory rot） | "陈旧的反思" | 情景缓冲区充满过时条目；用压缩/TTL 修复 |
| 睡眠时间反思（Sleep-time reflection） | "异步自反思" | 在非关键路径上运行 Self-Reflector，使主智能体保持快速 |

## 延伸阅读

- [Shinn 等人，Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366)——经典论文
- [Letta，Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute)——生产中的异步反思
- [Anthropic，Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)——将情景缓冲区作为上下文的一部分管理
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)——反思节点模式
