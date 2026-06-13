# A2A — 智能体对智能体协议

> Google 于 2025 年 4 月发布 A2A；截至 2026 年 4 月，规范位于 https://a2a-protocol.org/latest/specification/，已有 150 多个组织支持。A2A 是 MCP（第 13 课）的水平补充：MCP 是垂直的（智能体 ↔ 工具），A2A 是对等的（智能体 ↔ 智能体）。它定义了 Agent Card（发现机制）、带有制品的任务（文本、结构化数据、视频）、不透明的任务生命周期和认证。生产系统越来越多地将 MCP 与 A2A 配合使用。Google Cloud 在 2025-2026 年间将 A2A 支持集成到了 Vertex AI Agent Builder 中。

**类型：** 学习 + 构建
**语言：** Python（标准库、`http.server`、`json`）
**前置要求：** 第 16 阶段 · 04（原始模型）
**所需时间：** 约 75 分钟

## 问题

你的智能体需要调用另一个系统上的另一个智能体。怎么做？你可以暴露一个 HTTP 端点，定义一个自定义的 JSON 模式，然后期望对方也能理解。每一对智能体都变成了一次定制集成。

A2A 就是这个调用的通用线路协议。标准发现机制、标准任务模型、标准传输、标准制品。就像 HTTP+REST，但以智能体为一等公民。

## 概念

### 四个要素

**Agent Card。** 位于 `/.well-known/agent.json` 的 JSON 文档，描述智能体：名称、技能、端点、支持的模态、认证要求。发现通过读取该卡片完成。

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**任务。** 工作单元。一个异步的、有状态的对象，具有生命周期：`submitted → working → completed / failed / canceled`。客户端发送任务，轮询或订阅更新。

**制品。** 任务产生的结果类型。文本、结构化 JSON、图像、视频、音频。制品是有类型的，因此不同模态都是一等公民。

**不透明的生命周期。** A2A 不规定远程智能体*如何*解决任务。客户端看到状态转换和制品；实现可以自由使用任何框架。

### MCP/A2A 的分工

- **MCP**（第 13 课）：智能体 ↔ 工具。智能体通过 JSON-RPC 读写工具服务器。默认无状态。
- **A2A**：智能体 ↔ 智能体。对等协议；双方都是具有自己推理能力的智能体。

生产级多智能体系统两者都用。A2A 对等方在其端调用 MCP 工具。这种分离使两个关注点保持清晰。

### 发现流程

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

或者使用流式传输：通过 SSE 订阅 `/tasks/{id}/events` 获取推送更新。

### 认证

A2A 支持三种常见模式：

- **Bearer 令牌** — OAuth2 或不透明令牌。
- **mTLS** — 双向 TLS；组织之间互相证明身份。
- **签名请求** — 对负载进行 HMAC 签名。

认证在 Agent Card 中声明；客户端发现并遵守。

### 截至 2026 年 4 月已有 150 多个组织

企业采用推动了 A2A 的规模化。关键点：A2A 成为企业智能体系统跨越信任边界的方式。Google Cloud 发布了 Vertex AI Agent Builder 的 A2A 支持；Microsoft Agent Framework 支持它；大多数主流框架（LangGraph、CrewAI、AutoGen）都提供了 A2A 适配器。

### A2A 的优势场景

- **跨组织调用。** 公司 A 的智能体调用公司 B 的智能体。没有 A2A，每一对都是定制合约。
- **异构框架。** LangGraph 智能体调用 CrewAI 智能体调用自定义 Python 智能体。A2A 进行标准化。
- **类型化制品。** 视频结果、结构化 JSON、音频 —— 全部是一等公民。
- **长时间运行的任务。** 不透明的生命周期 + 轮询使数小时的任务变得简单直接。

### A2A 的局限

- **延迟敏感的微调用。** A2A 的生命周期是异步的。亚毫秒级的智能体对智能体调用不适用；使用直接 RPC。
- **紧耦合的进程内智能体。** 如果两个智能体运行在同一个 Python 进程中，A2A 的 HTTP 往返是多余的。
- **小团队。** 规范开销是真实的；仅内部使用的智能体可能不需要这种形式化。

### A2A 与 ACP、ANP、NLIP 的对比

2024-2026 年间出现了几个相关规范：

- **ACP**（IBM/Linux 基金会）— A2A 的前身，范围更窄。
- **ANP**（Agent Network Protocol）— 侧重对等发现，去中心化优先。
- **NLIP**（Ecma 自然语言交互协议，2025 年 12 月标准化）— 自然语言内容类型。

截至 2026 年 4 月，A2A 是采用最广泛的对等协议。参见 arXiv:2505.02279（Liu 等人，"A Survey of Agent Interoperability Protocols"）了解比较。

## 开始构建

`code/main.py` 使用 `http.server` 和 JSON 实现了一个最小的 A2A 服务器和客户端。服务器：

- 暴露 `/.well-known/agent.json`，
- 接受 `POST /tasks`，
- 管理任务状态，
- 在 `GET /tasks/{id}` 上返回制品。

客户端：

- 获取 Agent Card，
- 提交任务，
- 轮询直到完成，
- 读取制品。

运行：

```
python3 code/main.py
```

该脚本在后台线程中启动服务器，然后运行客户端。你可以看到完整流程：发现、提交、轮询、制品。

## 实际应用

`outputs/skill-a2a-integrator.md` 设计 A2A 集成：Agent Card 内容、任务模式、认证选择、流式传输与轮询。

## 投入生产

检查清单：

- **固定规范版本。** A2A 仍在演进；Agent Card 应声明协议版本。
- **幂等的任务创建。** 重复提交（网络重试）应只产生一个任务。
- **制品模式。** 声明智能体返回的数据结构；消费者应进行验证。
- **速率限制 + 认证。** A2A 是面向公众的；应用标准的 Web 安全措施。
- **失败任务的死信队列。** 随时间检查模式以发现重复出现的失败类型。

## 练习

1. 运行 `code/main.py`。确认客户端发现了服务器并接收到正确的制品。
2. 为服务器添加第二个技能（例如"summarize"）。更新 Agent Card。编写一个根据任务类型选择技能的客户端。
3. 实现一个 SSE 流式端点：`/tasks/{id}/events`，发出状态变更。客户端需要做哪些不同的事情？
4. 阅读 A2A 规范（https://a2a-protocol.org/latest/specification/）。找出规范要求但此演示未实现的三件事。
5. 比较 A2A（Agent Card 发现）与 MCP（通过 `listTools` 进行服务器端能力列表）。自描述智能体与能力探测之间的权衡是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| A2A | "智能体对智能体" | 智能体跨系统调用其他智能体的对等协议。Google 2025 年发布。 |
| Agent Card | "智能体的名片" | 位于 `/.well-known/agent.json` 的 JSON，描述技能、端点、认证。 |
| 任务 | "工作单元" | 具有生命周期的异步有状态对象；完成后产生制品。 |
| 制品 | "结果" | 类型化输出：文本、结构化 JSON、图像、视频、音频。一等媒体。 |
| 不透明的生命周期 | "如何解决是智能体的事" | 客户端看到状态转换；服务器可自由选择框架/工具。 |
| 发现 | "找到智能体" | `GET /.well-known/agent.json` 返回卡片。 |
| MCP 与 A2A | "工具与对等方" | MCP：垂直的智能体 ↔ 工具。A2A：水平的智能体 ↔ 智能体。 |
| ACP / ANP / NLIP | "兄弟协议" | 相邻规范；截至 2026 年 A2A 采用最广泛。 |

## 延伸阅读

- [A2A 规范](https://a2a-protocol.org/latest/specification/) — 权威规范
- [Google 开发者博客 — A2A 发布公告](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — 2025 年 4 月发布文章
- [A2A GitHub 仓库](https://github.com/a2aproject/A2A) — 参考实现和 SDK
- [Liu 等人 — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP、ACP、A2A、ANP 比较
