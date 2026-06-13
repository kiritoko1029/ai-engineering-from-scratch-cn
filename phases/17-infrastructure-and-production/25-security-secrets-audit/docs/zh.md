# 安全 — 密钥管理、API 密钥轮换、审计日志与防护栏

> 通过集中式密钥库（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除密钥散落问题。切勿将凭据存储在配置文件、VCS 中的环境变量文件或电子表格中。优先使用 IAM 角色而非静态密钥；CI/CD 使用 OIDC 认证。AI 网关模式是 2026 年的标准方案：应用 → 网关 → 模型提供商，网关在运行时从密钥库拉取凭据。在密钥库中轮换密钥，所有应用几分钟内即可生效——无需重新部署，无需在 Slack 中发送"谁有新密钥"的消息。轮换策略不超过 90 天；每次提交使用 TruffleHog / GitGuardian / Gitleaks 进行扫描。零信任架构：MFA、SSO、RBAC/ABAC、短期令牌、设备态势评估。PII 清洗通过实体识别在转发前遮蔽 PHI/PII；一致化脱敏（Mesh 方案）将敏感值映射到稳定的占位符，从而保留 LLM 的语义关系。网络出站：LLM 服务部署在专用 VPC/VNet 子网中，仅允许 `api.openai.com`、`api.anthropic.com` 等域名白名单；阻断所有其他出站流量。2026 年安全事件警示：Vercel 供应链攻击通过泄露的 CI/CD 凭据，在数千个客户部署中窃取了环境变量。

**类型：** 学习
**语言：** Python（标准库，演示用 PII 清洗器 + 审计日志写入器）
**前置要求：** 第 17 阶段 · 第 19 课（AI 网关），第 17 阶段 · 第 13 课（可观测性）
**所需时间：** 约 60 分钟

## 学习目标

- 列举四种密钥管理反模式（VCS 中的配置文件、硬编码的环境变量、电子表格、静态密钥）及其替代方案。
- 解释 AI 网关从密钥库拉取凭据的模式作为 2026 年生产标准的原因。
- 实现一个具有一致化脱敏功能的 PII 清洗器（相同值 → 相同占位符），以保留语义信息。
- 说明 2026 年 Vercel 供应链事件及其对 CI/CD 凭据管理的启示。

## 问题所在

一名实习生提交了包含 API 密钥的 `.env` 文件。他们很快删除了它，但密钥已经存在于 git 历史中——GitGuardian 扫描发现了这个问题，你的轮换流程是"在 Slack 通知团队，更新 40 个配置文件，重新部署所有服务"。8 小时后，一半的服务已经上线，另一半还在等待部署窗口。

另一个场景：用户提示词中包含"我的社会安全号码是 123-45-6789"。提示词被发送到 OpenAI。你有 BAA 协议，但内部政策要求在转发前遮蔽 PII，而你没有执行。

又一个场景：你的 EKS 集群中的 LLM Pod 可以访问任何互联网主机。有人通过 DNS 查询将数据传输到攻击者控制的域名。没有任何阻断措施。

LLM 服务的安全必须同时应对这三个攻击向量：基于密钥库的凭据管理、PII 清洗、网络出站过滤和审计日志。

## 概念说明

### 集中式密钥库 + IAM 角色拉取

**密钥库**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。唯一的可信数据源。

**IAM 角色**：应用/网关通过其 IAM 身份进行认证，而非使用静态密钥。密钥库在令牌有效期内返回凭据。

**AI 网关模式**：网关在请求时从密钥库拉取 `OPENAI_API_KEY`。在密钥库中轮换后，下一个请求即可获取新密钥。无需重新部署。

### 轮换策略不超过 90 天

所有 API 密钥、密钥库根令牌、CI/CD 凭据均适用。尽可能实现自动化轮换。手动轮换需记录并跟踪。

### 密钥扫描

- **TruffleHog** — 基于正则表达式和熵值的提交扫描。
- **GitGuardian** — 商业工具，准确率高。
- **Gitleaks** — 开源工具，可在 CI 中运行。

每次提交时运行。检测到新密钥时阻断 PR。

### 零信任架构

- 所有账户强制启用 MFA。
- 通过 SAML/OIDC 实现 SSO。
- RBAC（基于角色）或 ABAC（基于属性）实现细粒度访问控制。
- 短期令牌（小时级，而非天级）。
- 设备态势评估——仅允许经过磁盘加密的企业设备。

### PII / PHI 清洗

在提示词离开你的基础设施之前：

1. 实体识别（spaCy NER、Presidio、商业方案）。
2. 遮蔽匹配的实体：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致化脱敏（Mesh 方案）：相同值映射到相同的占位符，使 LLM 保留语义关系。
4. 可选：对 LLM 响应进行反向映射。

静态正则过滤器捕获基本模式；NER 能捕获更多内容。建议两者结合使用。

### 输入 + 输出防护栏

输入：拦截已知的越狱攻击、禁止话题；按用户进行速率限制。

输出：通过正则清洗泄露的密钥（API 密钥模式、拒绝上下文中的邮箱模式），使用分类器检测策略违规。

### 网络出站白名单

LLM 服务部署在专用子网中：
- 白名单：`api.openai.com`、`api.anthropic.com`、向量数据库端点、密钥库端点。
- 其余所有流量：丢弃。
- DNS 通过仅允许白名单的解析器进行解析（防止 DNS 隧道数据外泄）。

### 审计日志

每次 LLM 调用的不可变日志，包含：
- 时间戳。
- 用户 / 租户。
- 提示词哈希（出于隐私考虑，不记录原始提示词）。
- 模型 + 版本。
- Token 计数。
- 成本。
- 响应哈希。
- 任何防护栏触发记录。

按合规要求保留（SOC 2 保留 1 年，HIPAA 保留 6 年）。

### 2026 年 Vercel 事件

供应链攻击：泄露的 CI/CD 凭据在数千个客户部署中窃取了环境变量。教训：CI/CD 凭据等同于生产凭据。存储在密钥库中。严格限制权限范围。积极轮换。

### 关键数字

- 轮换策略：不超过 90 天。
- 每次提交扫描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026 事件：CI/CD 凭据泄露 → 数千个客户的环境变量外泄。
- 审计日志保留期限：SOC 2 = 1 年，HIPAA = 6 年。

## 开始使用

`code/main.py` 实现了一个演示用的 PII 清洗器，具有一致化脱敏功能和仅追加的审计日志。

## 部署产出

本课程生成 `outputs/skill-llm-security-plan.md`。根据合规范围和当前状态，规划密钥库迁移、清洗器、出站过滤和审计日志方案。

## 练习

1. 运行 `code/main.py`。发送两条引用相同社会安全号码的提示词。确认两条提示词获得相同的占位符。
2. 为部署在 EKS 上的 vLLM 服务设计网络出站策略，该服务需要调用 OpenAI + Anthropic + Weaviate。
3. 你在 git 历史中发现一个两年前的密钥。正确的应对措施是什么——轮换密钥、清理历史记录，还是两者都做？请说明理由。
4. 你的审计日志每天增长 10 GB。设计分层保留方案（热存储 30 天，温存储 12 个月，冷存储 6 年）。
5. 讨论反向脱敏（将真实值替换回 LLM 响应中）是否值得增加复杂度，还是保持占位符可见更为合理。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Vault | "密钥存储" | 集中式凭据管理服务 |
| IAM role | "基于身份的认证" | 应用承担的角色；返回短期凭据 |
| OIDC for CI/CD | "云签发令牌" | CI 中不使用静态密钥——通过 OIDC 进行身份认证 |
| TruffleHog / GitGuardian / Gitleaks | "密钥扫描器" | 提交时的密钥检测工具 |
| RBAC / ABAC | "访问控制" | 基于角色与基于属性的访问控制 |
| PII scrubbing | "数据遮蔽" | 移除或脱敏敏感实体 |
| Consistent tokenization | "稳定占位符" | 相同值 → 每次生成相同的脱敏令牌 |
| Mesh approach | "Mesh 脱敏" | 保留语义的脱敏模式 |
| Egress whitelist | "出站白名单" | 仅允许访问指定域名 |
| Audit log | "不可变历史记录" | 用于合规的仅追加记录 |

## 延伸阅读

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII 检测与匿名化工具。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
