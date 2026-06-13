# 技能库与终身学习（Voyager）

> Voyager（Wang 等人，TMLR 2024）将可执行代码视为技能。技能是命名的、可检索的、可组合的、可通过环境反馈优化的。这是 Claude Agent SDK 技能、skillkit 和 2026 年技能库模式的参考架构。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 07（MemGPT）、第 14 阶段 · 08（Letta 块）
**所需时间：** 约75分钟

## 学习目标

- 说出 Voyager 的三个组件——自动课程、技能库、迭代提示——及其各自的角色。
- 解释为什么 Voyager 将动作空间设为代码而非原始命令。
- 用标准库实现一个带注册、检索、组合和失败驱动优化的技能库。
- 将 Voyager 的模式映射到 2026 年 Claude Agent SDK 技能和 skillkit 生态。

## 问题所在

每次会话都从零重建所有能力的智能体做错了三件事：

1. **浪费 token。** 每个任务重新引导相同的推理。
2. **丢失进展。** 会话 A 中学到的纠错无法转移到会话 B。
3. **在长周期组合上失败。** 复杂任务需要能力层级；一次性提示无法表达它们。

Voyager 的答案：将每个可复用能力视为一段命名代码，存储在库中，可按相似性检索，可与其他技能组合，并通过执行反馈优化。

## 概念说明

### 三个组件

Voyager（arXiv:2305.16291）围绕以下结构组织智能体：

1. **自动课程。** 一个好奇心驱动的提议器根据智能体当前的技能集和环境状态选择下一个任务。探索是自下而上的。
2. **技能库。** 每个技能是可执行代码。任务成功时添加新技能。技能按查询到描述的相似性检索。
3. **迭代提示机制。** 失败时，智能体接收执行错误、环境反馈和自验证输出，然后优化技能。

Minecraft 评估（Wang 等人，2024）：独特物品 3.3 倍、石器工具速度 8.5 倍、铁器工具速度 6.4 倍、地图遍历长度 2.3 倍于基线。数据是 Minecraft 特定的，但模式可迁移。

### 动作空间 = 代码

大多数智能体发出原始命令。Voyager 发出 JavaScript 函数。一个技能是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子技能组合而成。以描述和嵌入为键存储。作为程序而非提示被检索。

这就是 2026 年 Claude Agent SDK 技能：智能体按需加载的命名、可检索的代码块加指令。

### 技能检索

新任务"制作钻石镐"。智能体：

1. 嵌入任务描述。
2. 查询技能库获取 top-k 相似技能。
3. 检索 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 从检索的原始技能加新逻辑组合新技能。

这就是 MCP 资源（第 13 阶段）和 Agent SDK 技能实现的模式：在知识/代码表面上检索，范围限定到当前任务。

### 迭代优化

Voyager 的反馈循环：

1. 智能体写一个技能。
2. 技能对环境运行。
3. 返回三种信号之一：`success`、`error`（带堆栈跟踪）、`self-verification failure`。
4. 智能体使用信号作为上下文重写技能。
5. 循环直到成功或达到最大轮次。

这就是 Self-Refine（第 05 课）应用于代码生成，带环境锚定验证。CRITIC（第 05 课）是同一模式，以外部工具为验证器。

### 课程与探索

Voyager 的课程模块根据智能体已有的和尚未做过的事情提出任务，如"在湖边建一个庇护所"。提议器使用环境状态加技能库存选择刚好超出当前能力的任务——探索的最佳点。

对于生产智能体，这转化为"缺失什么"算子：给定当前技能库和一个领域，我们尚未覆盖哪些技能？团队通常将此作为课程审查手动实现。

### 此模式出错的地方

- **技能库腐化。** 同一技能以略微不同的描述添加了 10 次。在写入时添加去重；检索只返回一个。
- **组合技能漂移。** 父技能依赖一个已被优化的子技能。对技能进行版本控制；固定在 v1 的父技能不会自动获得 v3。
- **检索质量。** 技能描述上的向量检索在库增长到几百个后退化。用标签过滤器和硬约束（"仅 `category=tooling` 的技能"）补充。

## 开始构建

`code/main.py` 实现了一个标准库技能库：

- `Skill`——名称、描述、代码（字符串形式）、版本、标签、依赖。
- `SkillLibrary`——注册、搜索（token 重叠）、组合（依赖拓扑排序）和优化（更新时版本递增）。
- 一个脚本化智能体，注册三个原始技能，组合第四个，遇到失败，然后优化。

运行它：

```
python3 code/main.py
```

轨迹展示了库写入、检索、组合、一次失败执行和 v2 优化——Voyager 循环端到端。

## 使用它

- **Claude Agent SDK 技能**（Anthropic）——2026 年参考：每个技能有描述、代码和指令；在智能体会话中按需加载。
- **skillkit**（npm: skillkit）——面向 32+ AI 编码智能体的跨智能体技能管理。
- **自定义技能库**——领域特定（数据智能体的 SQL 技能、基础设施智能体的 Terraform 技能）。Voyager 模式可以缩小规模。
- **OpenAI Agents SDK `tools`**——低端；每个工具是一个轻量级技能。

## 交付它

`outputs/skill-skill-library.md` 为任何目标运行时生成 Voyager 形态的技能库，带注册、检索、版本控制和优化。

## 练习

1. 给 `compose()` 添加依赖循环检测器。当技能 A 依赖 B 而 B 依赖 A 时会怎样？错误还是警告？
2. 实现每个技能的版本固定。当父技能组合子技能 `crafting@1` 时，对 `crafting@2` 的优化不应静默升级父技能。
3. 用 sentence-transformers 嵌入（或 BM25 标准库实现）替换 token 重叠检索。在 50 个技能的简易库上测量 retrieval@5。
4. 添加一个"课程"智能体：给定当前库和领域描述，提议 5 个缺失的技能。每周调用一次。
5. 阅读 Anthropic 的 Claude Agent SDK 技能文档。将简易库移植到 SDK 的技能 schema。可发现性有什么变化？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 技能（Skill） | "可复用能力" | 命名的代码块 + 描述，可按相似性检索 |
| 技能库（Skill library） | "智能体的操作记忆" | 技能的持久存储，可搜索和组合 |
| 课程（Curriculum） | "任务提议器" | 由当前能力差距驱动的自下而上目标生成器 |
| 组合（Composition） | "技能 DAG" | 技能调用技能；执行时拓扑排序 |
| 迭代优化（Iterative refinement） | "自纠错循环" | 环境反馈 + 错误 + 自验证折叠到下一个版本 |
| 动作空间即代码（Action-space-as-code） | "编程式动作" | 发出函数而非原始命令，用于时间延展行为 |
| 写入去重（Dedup on write） | "技能坍缩" | 近似重复描述坍缩为一个规范技能 |

## 延伸阅读

- [Wang 等人，Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291)——原始技能库论文
- [Claude Agent SDK 概览](https://platform.claude.com/docs/en/agent-sdk/overview)——技能作为 2026 年产品化
- [Anthropic，Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)——技能和子智能体实践
- [Madaan 等人，Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)——Voyager 底层的优化循环
