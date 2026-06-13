# 毕业项目 13 — 带注册中心和治理的 MCP 服务器

> Model Context Protocol 在 2026 年不再是未来，而是成为默认的工具使用规范。Anthropic、OpenAI、Google 和每个主流 IDE 都发布 MCP 客户端。Pinterest 发布了其内部 MCP 服务器生态系统。AAIF Registry 在 `.well-known` 处形式化了能力元数据。AWS ECS 发布了参考的无状态部署。Block 的 goose-agent 将同一协议内置到托管助手中。2026 年的生产形态是：StreamableHTTP 传输、OAuth 2.1 范围、OPA 策略门控，以及一个让平台团队发现、验证和启用服务器的注册中心。端到端构建它。

**类型：** 毕业项目
**语言：** Python（服务器，通过 FastMCP）或 TypeScript（@modelcontextprotocol/sdk）、Go（注册中心服务）
**前置要求：** 阶段 11（LLM 工程）、阶段 13（工具与 MCP）、阶段 14（智能体）、阶段 17（基础设施）、阶段 18（安全）
**涉及阶段：** P11 · P13 · P14 · P17 · P18
**所需时间：** 25 小时

## 问题所在

MCP 成为工具使用的通用语言。Claude Code、Cursor 3、Amp、OpenCode、Gemini CLI 以及每个托管智能体现在都消费 MCP 服务器。生产挑战不在于编写服务器（FastMCP 让这变得容易），而在于以企业需求大规模部署：每租户 OAuth 范围、破坏性工具的 OPA 策略、StreamableHTTP 无状态扩展、用于发现的注册中心、每次工具调用的审计日志。Pinterest 的内部 MCP 生态系统和 AAIF Registry 规范设定了 2026 年的标准。

你将构建一个暴露 10 个内部工具（Postgres 只读、S3 列表、Jira、Linear、Datadog 等）的 MCP 服务器、一个供平台发现的注册中心 UI，以及破坏性工具的人工审批门控。负载测试演示 StreamableHTTP 水平扩展。审计追踪满足企业安全审查。

## 概念说明

MCP 2026 修订版规定 StreamableHTTP 为默认传输。不同于早期的 stdio 和 SSE 形态，StreamableHTTP 默认无状态：单个 HTTP 端点接受 JSON-RPC 请求，流式响应，并支持长连接用于通知。无状态意味着可在负载均衡器后水平扩展。

授权是 OAuth 2.1，每工具有范围。Token 携带如 `jira:read`、`s3:list`、`postgres:query:readonly` 的范围。MCP 服务器在工具调用时检查范围，而非仅在会话开始时。对于高风险工具，服务器拒绝任何范围未在过去 N 分钟内提升到 `approved:by:human` 的调用——该提升来自 Slack 审查卡片。

注册中心是独立服务。每个 MCP 服务器暴露一个 `.well-known/mcp-capabilities` 文档，包含工具清单、传输 URL、认证要求。注册中心轮询、验证和索引。平台团队使用注册中心 UI 查看可用工具、所需范围以及负责团队。

## 架构

```
MCP client (Claude Code, Cursor 3, ...)
          |
          v
StreamableHTTP over HTTPS (JSON-RPC + streaming)
          |
          v
MCP server (FastMCP) behind load balancer
          |
   +------+------+---------+----------+------------+
   v             v         v          v            v
Postgres    S3 listing  Jira       Linear     Datadog
(read-only) (paged)     (read)     (read)     (query)
          |
   +------+-------------+
   v                    v
 OPA policy gate   destructive tool MCP (separate server)
                        |
                        v
                   human approval via Slack
                        |
                        v
                   audit log (append-only, per-tenant)

  registry service
     |
     v  GET /.well-known/mcp-capabilities from each server
     v
     UI: search / validate / enable-disable / ownership
```

## 技术栈

- 服务器框架：FastMCP（Python）或 `@modelcontextprotocol/sdk`（TypeScript）
- 传输：StreamableHTTP over HTTPS（无状态）
- 认证：OAuth 2.1，通过 SPIFFE / SPIRE 的工作负载身份
- 策略：OPA / Rego 规则，每工具；每请求的策略决策服务
- 注册中心：自托管，消费 `.well-known/mcp-capabilities` 清单
- 人工审批：Slack 交互消息，用于破坏性工具
- 部署：AWS ECS Fargate 或 Fly.io，每租户一个服务器或带租户范围的共享
- 审计：结构化 JSONL，每租户存储桶，含每次调用血缘

## 开始构建

1. **工具接口。** 暴露 10 个内部工具：Postgres 只读查询、S3 列表对象、Jira 搜索/获取、Linear 搜索/获取、Datadog 指标查询、PagerDuty 值班查询、GitHub 只读、Notion 搜索、Slack 搜索、Salesforce 只读。每个工具有类型化 schema 和范围标签。

2. **FastMCP 服务器。** 挂载工具。配置 StreamableHTTP 传输。添加 OAuth Token 自省和范围强制的中间件。

3. **OPA 策略。** 每工具的 Rego 策略：哪些范围允许调用、哪些 PII 脱敏适用、哪些负载大小上限适用。每次工具调用时调用决策服务。

4. **注册中心服务。** 独立的 Go 或 TS 服务，轮询已注册服务器的 `.well-known/mcp-capabilities`，用 JSON Schema 验证，并暴露列表/搜索/验证/启用禁用 UI。

5. **能力清单。** 每个服务器暴露 `.well-known/mcp-capabilities`：工具列表、认证要求、传输 URL、负责团队、SLO。

6. **破坏性工具分离。** 改变状态的工具（Jira create、Linear create、Postgres write）位于第二个 MCP 服务器上，有更严格的认证流程：Token 必须在 15 分钟内通过 Slack 卡片获得 `approved:by:human` 范围提升。

7. **审计日志。** 每租户仅追加 JSONL：`{timestamp, user, tool, args_redacted, response_redacted, outcome}`。写入前通过 Presidio 进行 PII 脱敏。

8. **负载测试。** 100 个并发客户端在 StreamableHTTP 上。通过添加第二个副本演示水平扩展；展示负载均衡器在无会话粘性的情况下重新分配。

9. **一致性测试。** 对两个服务器运行官方 MCP 一致性套件。通过所有必选部分。

## 使用示例

```
$ curl -H "Authorization: Bearer eyJhbGc..." \
       -X POST https://mcp.internal.example.com/ \
       -d '{"jsonrpc":"2.0","method":"tools/call",
            "params":{"name":"postgres.readonly","arguments":{"sql":"SELECT 1"}}}'
[registry]   capability validated: postgres.readonly v1.2
[policy]    scope postgres:query:readonly present; allowed
[audit]     logged: user=u42 tool=postgres.readonly outcome=ok
response:    { "result": { "rows": [[1]] } }
```

## 交付成果

`outputs/skill-mcp-server.md` 描述了交付成果。生产级 MCP 服务器 + 注册中心 + 审计层，用于内部工具，带 OAuth 2.1 范围和 OPA 门控。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 规范一致性 | StreamableHTTP + 能力清单通过 MCP 一致性测试 |
| 20 | 安全性 | 范围强制、OPA 覆盖每个工具、密钥卫生 |
| 20 | 可观测性 | 每工具调用审计日志，含 PII 脱敏 |
| 20 | 规模 | 100 客户端负载测试水平扩展演示 |
| 15 | 注册中心 UX | 发现 / 验证 / 启用禁用工作流 |
| **100** | | |

## 练习

1. 添加新工具（Confluence 搜索）。通过注册中心验证流程发布，不触及核心服务器。

2. 编写 OPA 策略，脱敏包含 `email`、`ssn` 或 `phone` 列名的 Postgres 查询结果。用探针查询验证。

3. 在本地延迟上基准 StreamableHTTP vs stdio。报告每次调用的 p50/p95。

4. 实现每租户配额：每租户每工具每分钟最多 N 次调用。通过第二个 OPA 规则强制执行。

5. 从 [mcp-conformance-tests](https://github.com/modelcontextprotocol/conformance) 运行 MCP 一致性套件并修复每个失败。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| StreamableHTTP | "2026 MCP 传输" | 无状态 HTTP + 流式；替代网络服务器的 SSE + stdio |
| 能力清单 | "well-known 文档" | `.well-known/mcp-capabilities`，含工具列表、认证、传输 URL |
| OPA / Rego | "策略引擎" | Open Policy Agent，根据外部规则授权工具调用 |
| 范围提升 | "人工批准" | 通过 Slack 审批授予的短期范围，破坏性工具需要 |
| 注册中心 | "工具发现" | 从能力清单索引 MCP 服务器的服务 |
| 工作负载身份 | "SPIFFE / SPIRE" | 用于 OAuth Token 颁发的加密服务身份 |
| 一致性套件 | "规范测试" | 官方 MCP 测试集，用于 StreamableHTTP + 工具清单正确性 |

## 延伸阅读

- [Model Context Protocol 2026 路线图](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、能力元数据、注册中心
- [AAIF MCP Registry 规范](https://github.com/modelcontextprotocol/registry) — 2026 年注册中心规范
- [AWS ECS 参考部署](https://aws.amazon.com/blogs/containers/deploying-model-context-protocol-mcp-servers-on-amazon-ecs/) — 参考生产部署
- [Pinterest 内部 MCP 生态系统](https://www.infoq.com/news/2026/04/pinterest-mcp-ecosystem/) — 参考内部部署
- [Block `goose` MCP 使用](https://block.github.io/goose/) — 参考智能体消费模式
- [FastMCP](https://github.com/jlowin/fastmcp) — Python 服务器框架
- [Open Policy Agent](https://www.openpolicyagent.org/) — 策略引擎参考
- [SPIFFE / SPIRE](https://spiffe.io) — 工作负载身份参考
