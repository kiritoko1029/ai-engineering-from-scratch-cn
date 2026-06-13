# 生产运行时：队列、事件、定时任务

> 生产智能体运行在六种运行时形态上：请求-响应、流式、持久执行、基于队列的后台、事件驱动和定时调度。先选形态，再选框架。在每种形态下，可观测性都是承重的。

**类型：** 学习
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 13（LangGraph），第 14 阶段 · 22（语音）
**所需时间：** 约60分钟

## 学习目标

- 说出六种生产运行时形态，并将每种与框架/产品模式匹配。
- 解释为什么持久执行（LangGraph）对长程任务至关重要。
- 描述事件驱动运行时以及 Claude Managed Agents 的适用场景。
- 解释可观测性作为承重层对多步智能体的意义。

## 问题所在

生产智能体的失败方式是 Jupyter Notebook 无法暴露的：第 37 步的网络超时、用户在语音通话中挂断、定时任务在机器重启时终止、后台工作者内存溢出。运行时形态决定了哪些失败可以存活。

## 概念说明

### 请求-响应

- 同步 HTTP。用户等待完成。
- 仅适用于短任务（<30s）。
- 技术栈：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- 可观测性：标准 HTTP 访问日志 + OTel span。

### 流式

- SSE 或 WebSocket 实现渐进输出。
- LiveKit 将此扩展到 WebRTC 用于语音/视频（第 22 课）。
- 技术栈：任何支持流式的框架 + 处理 SSE/WS 的前端。
- 可观测性：每块计时、首 token 延迟、尾部延迟。

### 持久执行

- 每步后检查点状态；失败时自动恢复。
- AutoGen v0.4 Actor 模型将故障隔离到单个智能体（第 14 课）。
- LangGraph 的核心差异化特性（第 13 课）。
- 当步数未知且恢复成本高时至关重要。

### 基于队列/后台

- 任务进入队列，工作者接取，结果通过 webhook 或发布/订阅回传。
- 对长程智能体至关重要（每个任务数十到数百步，根据 Anthropic 的计算机使用公告）。
- 技术栈：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自定义。
- 可观测性：队列深度、每任务延迟分布、死信队列大小。

### 事件驱动

- 智能体订阅触发器：新邮件、PR 打开、定时触发。
- Claude Managed Agents 开箱即用支持此模式（第 17 课）。
- CrewAI Flows（第 15 课）构建事件驱动的确定性工作流。
- 可观测性：触发源、事件到启动延迟、智能体延迟。

### 定时调度

- 定时任务形态的智能体，周期性运行。
- 结合持久执行，使失败的夜间运行能在下次执行时恢复。
- 技术栈：Kubernetes CronJob + 持久框架；托管（Render cron、Vercel cron）。

### 2026 年部署模式

- **CrewAI Flows** — 事件驱动生产。
- **Agno** 无状态 FastAPI — Python 微服务。
- **Mastra** 服务器适配器（Express、Hono、Fastify、Koa）— 嵌入场景。
- **Pipecat Cloud / LiveKit Cloud** — 托管语音（第 22 课）。
- **Claude Managed Agents** — 托管长时异步。

### 可观测性是承重的

没有 OpenTelemetry GenAI span（第 23 课）加 Langfuse/Phoenix/Opik 后端（第 24 课），你无法调试在第 40 步失败的多步智能体。这不是生产中的可选项。这是"我们快速调试"和"我们从头重放并增加更多日志"之间的区别。

### 生产运行时失败的地方

- **形态选择错误。** 为 5 分钟的任务选择请求-响应。用户挂断；工作者堆积；重试叠加。
- **没有死信队列。** 队列工作者没有死信。失败的任务消失。
- **后台工作不透明。** 后台智能体运行时没有追踪导出。在用户报告之前失败不可见。
- **跳过持久状态。** 任何超过 30 秒且不能承受重启的运行都需要持久执行。

## 开始构建

`code/main.py` 是一个标准库多形态演示：

- 请求-响应端点（普通函数）。
- 流式处理器（生成器）。
- 带死信队列的基于队列的工作者。
- 事件触发器注册表。
- 定时调度器。

运行方式：

```bash
python3 code/main.py
```

输出：五条追踪，展示每种形态在同一任务上的行为。相同的智能体逻辑，不同的外壳。持久执行（第六种形态）特意在第 13 课中通过 LangGraph 检查点讲解。

## 使用建议

- **请求-响应** — 聊天式用户体验。
- **流式** — 渐进响应。
- **持久** — 长程任务。
- **队列** — 批量/异步/长时运行。
- **事件** — 智能体响应性。
- **定时** — 日常维护（记忆整合、评估、成本报告）。

## 交付产物

`outputs/skill-runtime-shape.md` 为任务选择运行时形态并接入可观测性需求。

## 练习

1. 将你的第 01 课 ReAct 循程移植到你的技术栈中的全部六种形态。哪种形态适合哪种产品界面？
2. 为基于队列的演示添加死信队列。模拟 10% 任务失败；暴露死信队列大小。
3. 编写一个定时触发的评估智能体，每晚对你当天的前 20 条追踪运行。
4. 实现带背压的流式：如果客户端慢，暂停智能体。这如何与轮次预算交互？
5. 阅读 Claude Managed Agents 文档。什么时候你会将自托管的长程智能体迁移到托管？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 请求-响应 | "同步" | 用户等待；仅适用于短任务 |
| 流式 | "SSE / WS" | 渐进输出；更好的用户体验；每块延迟可观察 |
| 持久执行 | "从失败恢复" | 检查点状态；从上次步骤重启 |
| 基于队列 | "后台任务" | 生产者/工作者池/死信队列 |
| 事件驱动 | "基于触发" | 智能体响应外部事件 |
| 死信队列 | "死信队列" | 失败任务的停车场 |
| Claude Managed Agents | "托管运行器" | Anthropic 托管的长时异步，带缓存和压缩 |

## 延伸阅读

- [LangGraph 概述](https://docs.langchain.com/oss/python/langgraph/overview) — 持久执行详情
- [Claude Managed Agents 概述](https://platform.claude.com/docs/en/managed-agents/overview) — 托管长时异步
- [Anthropic，Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "每个任务数十到数百步"
- [AutoGen v0.4（微软研究院）](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — Actor 模型故障隔离
