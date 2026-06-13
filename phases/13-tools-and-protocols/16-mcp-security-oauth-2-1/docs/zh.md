# MCP 安全（二）—— OAuth 2.1、资源指示器、增量权限

> 远程 MCP 服务器需要授权，而不仅仅是认证。2025-11-25 规范与 OAuth 2.1 + PKCE + 资源指示器（RFC 8707）+ 受保护资源元数据（RFC 9728）对齐。SEP-835 增加了增量权限同意，在 403 WWW-Authenticate 上进行升级授权。本课将升级流程实现为状态机，以便你可以看到每一步。

**类型：** 构建
**语言：** Python（标准库，OAuth 状态机模拟器）
**前置要求：** Phase 13 · 09（传输层），Phase 13 · 15（安全一）
**所需时间：** 约75分钟

## 学习目标

- 区分资源服务器和授权服务器的职责。
- 走通 PKCE 保护的 OAuth 2.1 授权码流程。
- 使用 `resource`（RFC 8707）和受保护资源元数据（RFC 9728）防止混淆代理人攻击。
- 实现升级授权：服务器以 403 响应，附带 WWW-Authenticate 要求更高权限范围；客户端重新提示用户同意并重试。

## 问题所在

早期 MCP（2025 年之前）使用临时 API 密钥甚至无认证发布远程服务器。2025-11-25 规范通过完整的 OAuth 2.1 配置文件填补了这一空白。

三个实际需求：

- **普通远程服务器。** 用户安装访问其 Notion / GitHub / Gmail 的远程 MCP 服务器。OAuth 2.1 with PKCE 是正确的形式。
- **权限范围升级。** 已授予 `notes:read` 的笔记服务器可能在特定操作中需要 `notes:write`。无需重新执行整个流程，升级（SEP-835）请求额外权限范围。
- **防止混淆代理人。** 客户端持有一个针对服务器 A 的受众范围令牌。服务器 A 是恶意的，试图将该令牌出示给服务器 B。资源指示器（RFC 8707）将令牌固定到其预期受众。

OAuth 2.1 并不新鲜。新鲜的是 MCP 的配置文件：特定要求的流程（仅授权码 + PKCE；不使用隐式、默认不使用客户端凭据）、每个令牌请求强制使用资源指示器、发布受保护资源元数据以便客户端知道去哪里。

## 概念说明

### 角色

- **客户端。** MCP 客户端（Claude Desktop、Cursor 等）。
- **资源服务器。** MCP 服务器（笔记、GitHub、Postgres 等）。
- **授权服务器。** 颁发令牌。可以与资源服务器是同一服务，也可以是单独的 IdP（Auth0、Keycloak、Cognito）。

在 MCP 的配置文件中，资源和授权服务器可以是同一主机，但应通过 URL 加以区分。

### 授权码 + PKCE

流程：

1. 客户端生成 `code_verifier`（随机值）和 `code_challenge`（SHA256）。
2. 客户端将用户重定向到 `/authorize?response_type=code&client_id=...&redirect_uri=...&scope=notes:read&code_challenge=...&resource=https://notes.example.com`。
3. 用户同意。授权服务器重定向到 `redirect_uri?code=...`。
4. 客户端 POST 到 `/token?grant_type=authorization_code&code=...&code_verifier=...&resource=...`。
5. 授权服务器验证 verifier 的哈希是否与存储的 challenge 匹配，然后颁发访问令牌。
6. 客户端使用令牌：每次请求资源服务器时使用 `Authorization: Bearer ...`。

PKCE 防止授权码拦截攻击。资源指示器防止令牌在其他地方有效。

### 受保护资源元数据（RFC 9728）

资源服务器发布 `.well-known/oauth-protected-resource` 文档：

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:read", "notes:write", "notes:delete"]
}
```

客户端从资源服务器发现授权服务器。减少配置 —— 客户端只需要资源 URL。

### 资源指示器（RFC 8707）

令牌请求中的 `resource` 参数固定令牌的预期受众。颁发的令牌包含 `aud: "https://notes.example.com"`。接收到此令牌的另一个 MCP 服务器检查 `aud` 并拒绝。

### 权限范围模型

权限范围是空格分隔的字符串。常见 MCP 约定：

- `notes:read`、`notes:write`、`notes:delete`
- `admin:*` 用于管理功能（谨慎使用）
- `profile:read` 用于身份信息

权限范围选择应遵循最小权限原则：请求当前需要的范围，需要更多时再升级。

### 升级授权（SEP-835）

用户授予 `notes:read`。他们后来要求智能体删除一条笔记。服务器响应：

```
HTTP/1.1 403 Forbidden
WWW-Authenticate: Bearer error="insufficient_scope",
    scope="notes:delete", resource="https://notes.example.com"
```

客户端看到 insufficient_scope 错误，提示用户同意额外权限范围，为其执行一个小型 OAuth 流程，用新令牌重试请求。

### 令牌受众验证

每次请求：服务器检查 `token.aud == self.resource_url`。不匹配 = 401。这阻止了跨服务器令牌重用。

### 短有效期令牌和轮换

访问令牌应具有短有效期（默认 1 小时）。刷新令牌在每次刷新时轮换。客户端在后台处理静默刷新。

### 无令牌透传

采样服务器（Phase 13 · 11）不得将客户端的令牌透传给其他服务。采样请求就是边界。

### 防止混淆代理人

令牌绑定到 `aud`。客户端绑定到 `client_id`。每次请求都针对两者进行验证。规范明确禁止旧的"传递令牌"模式，这在前 MCP 远程工具生态系统中很常见。

### 客户端 ID 发现

每个 MCP 客户端在固定 URL 发布其元数据。授权服务器可以获取客户端的元数据文档来发现重定向 URI 和联系信息。这消除了手动客户端注册。

### 网关与 OAuth

Phase 13 · 17 展示企业网关如何处理 OAuth：网关持有上游服务器的凭据，发给客户端的令牌由网关颁发，上游令牌永远不会离开网关。这改变了信任模型 —— 用户与网关进行一次认证；网关处理 N 个服务器的授权。

## 开始构建

`code/main.py` 将完整的 OAuth 2.1 升级流程模拟为状态机。它实现了：

- PKCE code-verifier / challenge 生成。
- 带资源指示器的授权码流程。
- 受保护资源元数据端点。
- 带受众检查的令牌验证。
- `insufficient_scope` 时的升级。

本课没有 HTTP 服务器；状态机在内存中运行，你可以追踪每一步。Phase 13 · 17 的网关课程将其连接到实际传输层。

## 交付产出

本课生成 `outputs/skill-oauth-scope-planner.md`。给定一个带工具的远程 MCP 服务器，该技能设计权限范围集、固定规则和升级策略。

## 练习

1. 运行 `code/main.py`。追踪双范围升级流程。注意哪些步骤在升级时重复。

2. 添加刷新令牌轮换：每次刷新颁发新的刷新令牌并使旧令牌失效。模拟一个被盗的刷新令牌在轮换后被使用的情况，确认它会失败。

3. 使用 stdlib http.server 将受保护资源元数据端点实现为真实的 HTTP 响应。镜像第 09 课的 /mcp 端点。

4. 为 GitHub MCP 服务器设计权限范围层次结构：读取仓库、写入 PR、批准 PR、合并 PR、管理。在每个级别之间使用升级。

5. 阅读 RFC 8707 和 RFC 9728。找出 9728 中 MCP 使用方式与 RFC 示例不同的一个字段。（提示：涉及 `scopes_supported`。）

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|---------|---------|
| OAuth 2.1 | "现代 OAuth" | 强制 PKCE 并禁止隐式流程的合并 RFC |
| PKCE | "持有证明" | code verifier + challenge 防止授权码拦截 |
| 资源指示器 | "令牌受众" | RFC 8707 `resource` 参数将令牌固定到一台服务器 |
| 受保护资源元数据 | "发现文档" | RFC 9728 `.well-known/oauth-protected-resource` |
| 升级授权 | "增量同意" | SEP-835 按需添加权限范围的流程 |
| `insufficient_scope` | "403 附带 WWW-Authenticate" | 服务器发出需要重新同意更大范围的信号 |
| 混淆代理人 | "跨服务令牌重用" | 受信任的持有者不当转发令牌的攻击 |
| 短有效期令牌 | "访问令牌 TTL" | 快速过期的 Bearer；刷新令牌续期 |
| 权限范围层次结构 | "最小权限栈" | 级别的权限范围集，级别之间使用升级 |
| 客户端 ID 元数据 | "客户端发现文档" | 客户端发布自身 OAuth 元数据的 URL |

## 延伸阅读

- [MCP — Authorization spec](https://modelcontextprotocol.io/specification/draft/basic/authorization) — MCP OAuth 配置文件规范
- [den.dev — MCP November authorization spec](https://den.dev/blog/mcp-november-authorization-spec/) — 2025-11-25 变更讲解
- [RFC 8707 — Resource indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — 受众固定 RFC
- [RFC 9728 — OAuth 2.0 protected resource metadata](https://datatracker.ietf.org/doc/html/rfc9728) — 发现文档 RFC
- [Aembit — MCP OAuth 2.1, PKCE and the future of AI authorization](https://aembit.io/blog/mcp-oauth-2-1-pkce-and-the-future-of-ai-authorization/) — 实际升级流程讲解
