# CrewAI：基于角色的团队与流程

> CrewAI 是 2026 年基于角色的多智能体框架。四个原语：Agent、Task、Crew、Process。两种顶层形态：Crews（自主的、基于角色的协作）和 Flows（事件驱动的、确定性的）。文档直言："对于任何生产就绪的应用，从 Flow 开始。"

**类型：** 学习 + 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 12（工作流模式）、第 14 阶段 · 14（Actor 模型）
**所需时间：** 约75分钟

## 学习目标

- 说出 CrewAI 的四个原语（Agent、Task、Crew、Process）及其各自拥有什么。
- 区分 Sequential、Hierarchical 和计划中的 Consensus 过程；按工作负载选择。
- 区分 Crews（自主的、基于角色的）和 Flows（事件驱动的、确定性的），并解释文档的生产建议。
- 使用 `@tool` 装饰器和 `BaseTool` 子类接入工具；推理结构化输出 vs 自由文本。
- 说出 CrewAI 的四种记忆类型及其各自的收益场景。
- 用标准库实现一个三智能体团队（研究员、作者、编辑）产出简报。
- 识别 CrewAI 的三种失败模式：提示膨胀、管理器 LLM 税、脆弱的移交。

## 问题所在

采用多智能体框架的团队都会撞上同一面墙。"自主协作"在演示中听起来很好。然后客户提交 bug 你需要确定性重放。或者财务问一个 LLM 路由的团队每次运行花多少钱。或者值班需要知道凌晨 3 点哪个智能体卡住了。

自由形式的 LLM 路由团队无法干净地回答这些。纯 DAG 回答了所有但失去了头脑风暴智能体需要的探索形态。

CrewAI 的划分诚实地面对了这个权衡。Crews 用于协作的、基于角色的、探索性的工作。Flows 用于事件驱动的、代码拥有的、可审计的生产。同一个框架，两种形态，按表面选择。

## 概念说明

### 四个原语

CrewAI 的接口很小。记住这些，其余都是配置。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。背景故事是承重的。它塑造语气、判断、智能体何时停止。工具是智能体可以调用的函数（详见下文）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可复用的工作单元。`expected_output` 是契约。`context` 列出其输出被传入的上游任务。`output_pydantic` 强制结构化形状。
- **Crew。** 容器。拥有 `agents` 列表、`tasks` 列表、`process`，以及可选的 `memory` + `verbose` + `manager_llm` 设置。
- **Process。** 执行策略。Sequential、Hierarchical、Consensus（计划中）。决定运行的形状。

Agent 之间不能直接看到对方。Task 引用 Agent。Crew 排序 Task。Process 决定谁选择下一个 Task。这就是全部心智模型。

> **验证版本** CrewAI 0.86（2026-05）。新版本可能重命名或合并过程类型；依赖特定形状前请检查 [CrewAI Processes 文档](https://docs.crewai.com/concepts/processes)。

### Sequential vs Hierarchical vs Consensus

- **Sequential。** Task 按声明顺序运行。Task N 的输出作为 `context` 对 Task N+1 可用。最低成本。最可预测。顺序固定时使用。
- **Hierarchical。** 管理器 Agent（独立的 LLM 调用）在专家间路由。CrewAI 根据你的 `manager_llm` 配置或默认值生成管理器。管理器每轮选择下一个 Task，可以拒绝或重路由。当你有四个以上专家且顺序真正取决于先前输出时使用。
- **Consensus。** 计划中，目前未在公开 API 中实现。文档为未来的基于投票的过程保留了名称。今天不要依赖它。

Hierarchical 在每个专家调用之上添加每轮一次的 LLM 调用（管理器）。五步运行的 token 成本可能翻三倍。仅在需要路由时支付。

### Crews vs Flows

这是文档在 2026 年首先提出的框架。

- **Crew。** LLM 驱动的自主性。框架在运行时选择形状。适用于：研究、头脑风暴、初稿、路径是答案一部分的地方。难以重放。难以测试。原型设计便宜。
- **Flow。** 你拥有的事件驱动图。`@start` 标记入口。`@listen(topic)` 标记在另一个步骤发出该主题时触发的步骤。每个步骤是普通 Python（可以在内部调用 Crew）。适用于：生产。可观察。可测试。确定性。

文档的 2026 年生产建议：从 Flow 开始。当自主性能证明其成本时，在 Flow 步骤内部将 Crews 作为 `Crew.kickoff()` 调用折叠进去。Flow 给你审计轨迹，Crew 给你探索。组合，而非选择。

### 工具集成

三种方式给 Agent 工具。选择最简单的一种。

1. **`@tool` 装饰器。** 纯函数变成工具。签名是 schema；文档字符串是 LLM 看到的描述。最适合一次性辅助函数。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子类。** 基于类的工具，带显式参数 schema、异步支持、重试。当工具有状态（客户端、缓存）或需要结构化参数时使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **内置工具包。** CrewAI 提供第一方适配器：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一个导入即可接入。

结构化输出使用 Pydantic。在 Task 上设置 `output_pydantic=MyModel`。CrewAI 根据模型验证 LLM 响应，要么强制转换要么重试。配合严格的 `expected_output` 字符串使用。自由文本输出适合初稿；结构化输出是下游 Flow 可以消费的。

### 记忆钩子

CrewAI 开箱即提供四种记忆类型。它们可组合：一个 Crew 可以同时启用全部四种。

> **验证版本** CrewAI 0.86（2026-05）。近期版本通过统一的 `Memory` 系统路由一切，该系统包装了这四个存储。下面的概念模型仍然成立，但公开类接口可能在新版本中坍缩为单一 `Memory` 入口点；请检查 [CrewAI memory 文档](https://docs.crewai.com/concepts/memory) 了解当前 API。

- **短期。** 单次运行内的对话缓冲区。结束时清除。
- **长期。** 跨运行持久。存储在向量数据库中（默认 Chroma，可替换）。按与当前任务的相似性检索。
- **实体。** 每个实体的事实。"客户 X 使用企业计划。"按键而非相似性索引。跨运行存活。
- **上下文。** 组装时检索。在 Agent 需要的时刻拉取相关记忆，而非预加载。

在 Crew 上用 `memory=True` 或按类型配置启用。由你配置的嵌入提供商支持（默认 OpenAi，可替换为本地）。记忆是 CrewAI 相对于更薄的框架证明其价值的地方之一；纯 LangGraph 需要你自己接入每一个。

### CrewAI 适合的场景

- 三到六个带命名角色和协作工作流的 Agent。起草、审查、规划、头脑风暴。
- LLM 关于下一步的判断是价值一部分的路由（Hierarchical）。
- 团队更喜欢读 `role + goal + backstory` 而非读图定义的地方。

### CrewAI 不适合的场景

- 严格排序的确定性 DAG。使用 LangGraph（第 13 课）。图形状是正确的抽象；CrewAI 的角色框架是摩擦。
- 亚秒级延迟预算。Hierarchical 添加往返。即使 Sequential 也序列化包含背景故事和先前输出的提示。
- 单智能体循环。跳过框架；智能体循环（第 1 课）加工具注册表更短。

第 17 课（智能体框架权衡）在矩阵中列出了这些。简版：CrewAI 位于"协作的、基于角色的"角落。

### 依赖形态

独立于 LangChain。Python 3.10 到 3.13。使用 `uv`。星数：参见 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 快照）。AWS Bedrock 集成已有文档；供应商基准报告在 QA 工作负载上比 LangGraph 有显著加速，但方法论（数据集、硬件、评估指标）未公开，所以将框架供应商数据仅视为方向性。

### 此模式出错的地方

- **背景故事导致的提示膨胀。** 每个 Agent 2000 字的背景故事和五 Agent 团队在第一次工具调用前就烧完了上下文预算。保持背景故事在 200 字以内。跨 Agent 复用短语；不要将公司风格重复五遍。
- **管理器 LLM token 税。** Hierarchical 过程在每个专家调用前添加管理器 LLM 调用。五 Task 的团队变成六次 LLM 调用而非五次，管理器调用携带完整的任务列表加先前输出。除非路由依赖输出，否则切换到 Sequential。
- **脆弱的移交。** Task N 的 `expected_output` 是"一个大纲"。Task N+1 读取它作为 `context` 并尝试解析三个部分。LLM 产出四个。下游 Agent 即兴发挥。在 Task N 上用 `output_pydantic` 修复，使 Task N+1 读取类型化对象而非自由文本。
- **Crew 当生产用。** 自由形式的 Crew 没有 Flow 包装就发布到生产。输出变异性高；重放不可能；值班无法将坏运行与好运行做 diff。用 Flow 包装。

## 开始构建

`code/main.py` 用标准库实现了两种形态和一个三智能体团队。

形态：

- `Agent`、`Task` 数据类匹配 CrewAI 的接口。
- `SequentialCrew.kickoff(inputs)` 按声明顺序运行 Task，将输出作为 `context` 传递。
- `HierarchicalCrew.kickoff(topic)` 添加管理器 Agent 每轮选择下一个专家，在"done"时停止。
- `Flow` 带 `@start` 和 `@listen(topic)` 装饰器、一个微小事件循环和轨迹。
- `tool(name)` 装饰器镜像 CrewAI 的 `@tool` 形状。
- `Memory` 带 `short_term`、`long_term`、`entity` 存储；模拟相似度使用 numpy。
- 模拟 LLM 响应是硬编码字符串，按键为角色加输入前缀。无网络。确定性。

具体演示：研究员、作者、编辑团队产出关于"2026 年智能体工程"的简报。研究员拉取（模拟的）来源。作者起草。编辑精简。同一团队通过 Flow 运行以展示确定性形态。

运行它：

```bash
python3 code/main.py
```

轨迹覆盖：Sequential 团队通过 `context` 传递输出、Hierarchical 团队带管理器选择（研究员、作者、编辑，然后"done"）、Flow 用显式主题（`researched`、`drafted`、`edited`）运行相同的三步、通过 `@tool` 路由的工具调用、以及跨两次 kickoff 存活的长期记忆。

Crew 轨迹是流动的；管理器原则上可以重新排序。Flow 轨迹是固定的。这个选择就是课程。

## 使用它

- **CrewAI Flow** 用于生产。即使 Flow 只是一个调用 `Crew.kickoff()` 的步骤。Flow 给你审计边界。
- **CrewAI Crew（Sequential）** 用于顺序明确的协作工作，尤其是初稿和审查循环。
- **CrewAI Crew（Hierarchical）** 当路由依赖输出且你有四个以上专家时。
- **LangGraph**（第 13 课）用于显式状态机、持久恢复、严格排序。
- **AutoGen v0.4**（第 14 课）用于 Actor 模型并发和故障隔离。
- **OpenAI Agents SDK**（第 16 课）用于 OpenAI 优先的产品，带 Handoffs 和 Guardrails。
- **Claude Agent SDK**（第 17 课）用于 Claude 优先的产品，带子智能体和会话存储。

## 交付它

`outputs/skill-crew-or-flow.md` 为任务选择 Crew vs Flow 并脚手架最小实现。硬拒绝：没有背景故事的 Crew、没有显式主题的 Flow、少于三个专家的 Hierarchical。

## 陷阱

- **背景故事作为调味料。** 它塑造输出。测试三个变体；方差是真实的。选一个，冻结。
- **跳过 `expected_output`。** 没有每个任务的契约，下游任务拿到 LLM 产出的任何东西。Crew 运行；审计失败。
- **记忆常开。** 长期记忆每次运行都写入。向量数据库增长。检索变噪。将写入范围限定到事实持久的任务。
- **管理器提示漂移。** Hierarchical 的管理器提示是隐式的。如果路由变得奇怪，在 verbose 模式下转储并阅读。
- **Crew 中的工具副作用。** Crew 可以比预期更多次调用工具。POST、DELETE、支付属于 Flow 步骤，永远不属于 Crew 工具。

## 练习

1. 将 Sequential 团队转换为 Flow。数一下变异性下降的接触点。注意可读性下降的地方。
2. 给团队添加实体记忆：关于客户的事实跨 kickoff 持久。验证检索拉取了正确的实体。
3. 实现一个 Hierarchical 过程，管理器在作者输出至少三段之前拒绝路由到编辑器。追踪重试。
4. 为（模拟的）网页搜索接入 `BaseTool` 子类。比较轨迹形状与 `@tool` 装饰器版本。
5. 给编辑 Task 添加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。让作者 Task 产出一次格式错误的 JSON；在轨迹中验证 CrewAI 的重试行为。
6. 阅读 CrewAI 的文档介绍。将简易实现移植到真实的 `crewai` API。标准库版本跳过了什么保证？
7. 将 AgentOps 或 Langfuse（第 24 课）接入真实运行。标准库版本遗漏了什么轨迹？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| Agent | "角色" | 角色 + 目标 + 背景故事 + 工具 |
| Task | "工作单元" | 描述 + 预期输出 + 负责人 + 可选结构化输出 |
| Crew | "智能体团队" | Agents + Tasks + Process 的容器 |
| Process | "执行策略" | Sequential / Hierarchical / Consensus（计划中） |
| Flow | "确定性工作流" | 事件驱动、代码拥有、可测试 |
| 背景故事（Backstory） | "角色提示" | Agent 的语气和判断塑造器 |
| `@tool` | "函数工具" | 将函数变为 Agent 可调用工具的装饰器 |
| `BaseTool` | "类工具" | 带参数 schema、重试、异步支持的基于类的工具 |
| 实体记忆（Entity memory） | "每实体事实" | 范围限定到客户/账户/问题的记忆 |
| 长期记忆（Long-term memory） | "跨运行记忆" | 在 kickoff 之间存活的向量支持记忆 |
| 上下文记忆（Contextual memory） | "即时检索" | 在 Agent 需要的时刻拉取的记忆 |
| 管理器 LLM（Manager LLM） | "路由器智能体" | Hierarchical 过程中选择下一个 Task 的额外 LLM |
| `expected_output` | "Task 契约" | 告诉 Agent（和审计）返回什么形状的字符串 |

## 延伸阅读

- [CrewAI 文档介绍](https://docs.crewai.com/en/introduction)：概念和推荐的生产路径
- [CrewAI Flows 指南](https://docs.crewai.com/en/concepts/flows)：事件驱动形态、`@start`、`@listen`
- [CrewAI 工具参考](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、内置工具包
- [CrewAI 记忆](https://docs.crewai.com/en/concepts/memory)：短期、长期、实体、上下文
- [Anthropic，Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：多智能体何时有用何时无用
- [LangGraph 概览](https://docs.langchain.com/oss/python/langgraph/overview)：状态机替代方案
