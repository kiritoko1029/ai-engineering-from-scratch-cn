# 自主编码智能体全景（2026 年）

> SWE-bench Verified 在不到三年内从 4% 提升到 80.9%。同一个 Claude Sonnet 4.5 在 SWE-agent v1 上得分为 43.2%，在 Cline 自主模式下为 59.8%——模型周围的脚手架现在与模型本身同等重要。OpenHands（前身 OpenDevin）是最活跃的 MIT 许可平台，其 CodeAct 循环直接在沙箱中执行 Python 动作，而非 JSON 工具调用。头条数字掩盖了一个方法论问题：500 个 SWE-bench Verified 任务中有 161 个只需要 1-2 行更改，而 SWE-bench Pro（10 行以上的任务）在相同的前沿模型上只有 23-59%。

**类型：** 学习
**语言：** Python（标准库，CodeAct vs JSON 工具调用比较）
**前置要求：** 第 14 阶段 · 07（工具使用），第 15 阶段 · 01（长时域智能体）
**所需时间：** 约45分钟

## 问题所在

"哪个编码智能体最好"是错误的问题。正确的问题是：在一个匹配我工作的任务分布上，使用我将在生产中运行的脚手架，我能获得什么样的端到端可靠性？

2022 年到 2026 年间，该领域认识到脚手架——检索层、规划器、沙箱、编辑-验证循环、反馈格式——是承重的。Claude Sonnet 4.5 在 SWE-agent v1 上的 SWE-bench Verified 得分为 43.2%；同一模型在 Cline 的自主脚手架内得分为 59.8%。16.6 个绝对点的差异，相同的权重。基础模型是组件；循环是产品。

伴随的问题是基准饱和掩盖了回归。SWE-bench Verified 接近饱和，简单任务的尾巴（500 个任务中 161 个需要 ≤2 行）将最高分拉高。现实世界质量更好地在 SWE-bench Pro（10 行以上更改）等分布上衡量，同样的领先者仍在 23-59%。

## 概念说明

### SWE-bench，一段话

SWE-bench（Jimenez 等人）取带有真实补丁的 GitHub issue，要求智能体生成使测试套件通过的补丁。SWE-bench Verified（OpenAI，2024）是人工策展的 500 个任务子集，移除了模糊和损坏的任务。SWE-bench Pro 是更难的后续——需要 10 行以上更改的任务，当前前沿智能体在 23-59%。

### 2022 → 2026 曲线实际展示了什么

- **2022**：研究模型在原始 SWE-bench 上约 4%。
- **2024**：GPT-4 + Devin 风格脚手架约 14%；SWE-agent 约 12%。
- **2025**：Claude 3.5/3.7 Sonnet 在 Aider 和 SWE-agent 内推入 40-55% 范围。
- **2026**：Claude Sonnet 4.5 和前沿竞争者在 SWE-bench Verified 上 70-80%+。Epoch AI 的排行榜实时跟踪。

斜率来自三个复合来源：更好的基础模型、更好的脚手架（CodeAct、反射、验证器循环）和更好的基准（Verified 移除噪声）。

### CodeAct vs JSON 工具调用

OpenHands（All-Hands-AI，arXiv:2407.16741，前身 OpenDevin）做了一个特定的架构选择：模型不发出由主机解码和执行的 JSON 工具调用，而是发出 Python 代码，由 Jupyter 风格的内核在沙箱中运行。智能体可以在一个动作内循环文件、链接工具和捕获自己的异常。

权衡：

- **JSON 工具调用**：每个动作一个回合；易于审计；有限的组合性；默认安全，因为每个调用经过显式验证器。
- **CodeAct**：一个动作可以是一个完整的程序；组合性强；需要加固的沙箱（OpenHands 使用 Docker 隔离）；故障模式包括沙箱运行时允许的任何事情。

两种架构都在生产中使用。CodeAct 在开放平台（OpenHands、smolagents）中占主导。JSON 工具调用在托管服务（Anthropic Managed Agents、OpenAI Assistants）中仍然占主导，提供商控制执行器。

### 2026 年全景中的脚手架

| 脚手架 | 许可证 | 执行模型 | 显著特性 |
|---|---|---|---|
| OpenHands（OpenDevin） | MIT | Docker 中的 CodeAct | 最活跃的开放平台；事件流可重放 |
| SWE-agent | MIT | Agent-Computer Interface (ACI) | 第一个端到端 SWE-bench 脚手架 |
| Aider | Apache-2 | 本地仓库中的 diff 编辑 | 最小脚手架，回归稳定性强 |
| Cline | Apache-2 | 带工具策略的 VS Code 智能体 | Sonnet 4.5 上得分最高的开放脚手架 |
| Devin（Cognition） | 专有 | 托管 VM + 规划器 | 第一个"AI 软件工程师"产品类别 |
| Claude Code | 专有 | 权限模式 + 例程 | 第 10 课详细介绍智能体循环 |

### 为什么脚手架占主导

编码运行是长时域轨迹（第 1 课）。可靠性在步骤间复合。脚手架在三个地方获取分数：

1. **检索**：找到正确的文件阅读是静默的瓶颈。SWE-agent 的 ACI、OpenHands 的文件索引和 Aider 的仓库地图都在攻克这一点。
2. **验证器循环**：运行测试、阅读堆栈跟踪和重试是 SWE-bench 上 10 分以上的增量。
3. **故障遏制**：出错时回滚的沙箱防止复合损害。同一模型有和没有验证器循环看起来像两个不同的产品。

### 基准饱和与真实分布

OpenHands 作者和 Epoch AI 都指出 SWE-bench Verified 有一个简单的尾巴：500 个任务中 161 个只需要 1-2 行更改。高分部分由这个尾巴驱动。SWE-bench Pro 限制为 10 行以上更改，即使对前沿系统也返回 23-59% 的分数。你的生产分布几乎肯定更接近 Pro 而非 Verified。

选择智能体的含义：在你自己的 bug 积压上运行一个类似 Pro 的子集。重要的分数是在你实际交付的任务代表上的分数。

## 开始构建

`code/main.py` 在固定的迷你任务分布上比较两种玩具智能体脚手架：

1. **JSON 工具调用**脚手架，每个回合执行一个动作。
2. **CodeAct**脚手架，每个动作可以发出一小段 Python 代码片段。

两者都使用存根"模型"（确定性规则），因此比较将脚手架与模型质量隔离。输出显示 CodeAct 脚手架以更大的每动作爆炸半径为代价，用更少的回合解决了更多任务。

## 交付产出

`outputs/skill-scaffold-audit.md` 帮助你在采用前审计拟议的编码智能体脚手架：检索质量、验证器存在性、沙箱隔离以及基准与分布的匹配度。

## 练习

1. 运行 `code/main.py`。每个脚手架在相同任务集上需要多少回合？各自的每动作爆炸半径是多少？

2. 阅读 OpenHands 论文（arXiv:2407.16741）。论文论证 CodeAct 在复杂任务上优于 JSON 工具调用。识别论文承认的一个故障模式，写一句话说明该模式在生产中何时会占主导。

3. 从你的 bug 积压中选择一个需要跨两个文件 10 行以上更改的任务。估计前沿模型在（a）JSON 工具调用和（b）CodeAct 下的端到端成功概率。解释差距。

4. SWE-bench Verified 有 161 个单文件、1-2 行任务。构建一个排除它们的分数。排行榜如何重新排列？

5. 阅读"Introducing SWE-bench Verified"（OpenAI）。解释用于移除模糊任务的具体方法论，并说出策展会遗漏的一个类别。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|---|---|---|
| SWE-bench | "编码基准" | 带有真实补丁和测试套件的真实 GitHub issue |
| SWE-bench Verified | "清理后的子集" | 500 个人工策展的任务，简单尾巴仍然存在 |
| SWE-bench Pro | "更难的子集" | 10 行以上更改；前沿在 23-59% |
| CodeAct | "代码即动作" | 智能体发出 Python；Jupyter 风格内核在沙箱中执行 |
| JSON tool call（JSON 工具调用） | "函数调用" | 每个动作是执行前经过验证的结构化 JSON 载荷 |
| Scaffold（脚手架） | "智能体框架" | 基础模型周围的检索 + 规划器 + 执行器 + 验证器循环 |
| ACI（Agent-Computer Interface） | "SWE-agent 的格式" | 为 LLM 人体工学设计的命令集，而非人类 shell |
| Verifier loop（验证器循环） | "测试并重试" | 运行测试、阅读输出、修改补丁；最大的非模型可靠性增益 |

## 延伸阅读

- [Jimenez et al. — SWE-bench](https://www.swebench.com/) — 原始基准和方法论。
- [OpenAI — Introducing SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/) — 策展子集的构建方式。
- [Wang et al. — OpenHands: An Open Platform for AI Software Developers](https://arxiv.org/abs/2407.16741) — CodeAct 架构和事件流设计。
- [Epoch AI — SWE-bench leaderboard](https://epoch.ai/benchmarks) — 实时跟踪分数。
- [Anthropic — Measuring agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 长时域编码智能体可靠性框架。
