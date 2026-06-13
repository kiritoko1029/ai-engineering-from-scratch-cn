# 浏览器智能体与长时域 Web 任务

> ChatGPT agent（2025 年 7 月）将 Operator 和深度研究合并为一个浏览器/终端智能体，并以 68.9% 创造了 BrowseComp SOTA。OpenAI 于 2025 年 8 月 31 日关闭了 Operator——产品层的整合。Anthropic 的 Vercept 收购将 Claude Sonnet 在 OSWorld 上从不到 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修复了原始 WebArena 中 11.3 个百分点的假阴性率，并推出了 258 个任务的 Hard 子集。数字是真实的。攻击面也是真实的：OpenAI 的 preparedness 负责人公开表示，浏览器智能体中的间接提示注入"不是一个可以完全修补的漏洞"。已记录的 2025-2026 年攻击：Tainted Memories（Atlas CSRF）、HashJack（Cato Networks）和 Perplexity Comet 中的一键劫持。

**类型：** 学习
**语言：** Python（标准库，间接提示注入攻击面模型）
**前置要求：** 第 15 阶段 · 10（权限模式），第 15 阶段 · 01（长时域智能体）
**所需时间：** 约45分钟

## 问题所在

浏览器智能体是一个阅读不受信任内容并执行重要动作的长时域智能体。智能体访问的每个页面都是用户未编写的输入。每个页面上的每个表单都是潜在的命令通道。2025-2026 年的攻击语料表明这不是假设：Tainted Memories 让攻击者通过精心构造的页面将恶意指令绑定到智能体的内存；HashJack 将命令隐藏在智能体访问的 URL 片段中；Perplexity Comet 劫持一次点击即可生效。

防御图景令人不安。OpenAI 的 preparedness 负责人说出了安静的部分：间接提示注入"不是一个可以完全修补的漏洞"。这是因为攻击存在于智能体的阅读与行动边界上，这在架构上是模糊的——模型阅读的每个 token 原则上都可以被当作指令。

本课命名了攻击面，命名了基准全景（BrowseComp、OSWorld、WebArena-Verified），并模拟了一个最小的间接提示注入场景，以便你在第 14 和 18 课中推理真实防御。

## 概念说明

### 2026 年全景，每个系统一段

**ChatGPT agent（OpenAI）。** 2025 年 7 月发布。统一了 Operator（浏览）和 Deep Research（多小时研究）。2025 年 8 月 31 日关闭了独立的 Operator。BrowseComp 上 SOTA 68.9%；OSWorld 和 WebArena-Verified 上数字强劲。

**Claude Sonnet + Vercept（Anthropic）。** Anthropic 的 Vercept 收购聚焦于计算机使用能力。将 Claude Sonnet 在 OSWorld 上从 <15% 提升到 72.5%。Claude Computer Use 作为工具 API 发布。

**Gemini 3 Pro with Browser Use（DeepMind）。** Browser Use 集成发布了计算机使用控制；FSF v3（2026 年 4 月，第 20 课）特别跟踪 ML R&D 领域的自主性。

**WebArena-Verified（ServiceNow，ICLR 2026）。** 修复了一个有据可查的问题：原始 WebArena 有约 11.3% 的假阴性率（标记为失败但实际已解决的任务）。Verified 版本使用人工策展的成功标准重新评分，并添加了 258 个任务的 Hard 子集（ICLR 2026 论文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs OSWorld vs WebArena

| 基准 | 衡量什么 | 时域 |
|---|---|---|
| BrowseComp | 在时间压力下在开放网络上查找特定事实 | 分钟 |
| OSWorld | 智能体操作完整桌面（鼠标、键盘、shell） | 数十分钟 |
| WebArena-Verified | 模拟站点中的事务性 Web 任务 | 分钟 |
| Hard subset（Hard 子集） | 具有多页面状态转换的 WebArena-Verified 任务 | 数十分钟 |

不同维度。高 BrowseComp 分数说明智能体能找到事实；它不说明智能体能订机票。OSWorld 分数更接近"它在我的桌面上能用吗"。WebArena-Verified 更接近"它能完成一个流程吗"。任何生产决策都需要匹配任务分布的基准。

### 攻击面，命名

1. **间接提示注入。** 不受信任的页面内容包含指令。智能体阅读它们。智能体执行它们。公开例子：2024 Kai Greshake 等人，2025 Tainted Memories 论文，2026 HashJack（Cato Networks）。
2. **URL 片段 / 查询注入。** 被爬取 URL 的 `#fragment` 或查询字符串包含命令。从不被可见渲染；但仍在智能体的上下文中。
3. **内存绑定攻击。** 页面指示智能体写入持久化内存（第 12 课涵盖持久化状态）。下次会话时，内存触发载荷，没有可见触发器。
4. **认证会话上的 CSRF 形状攻击。** Tainted Memories 类：智能体在某处已登录；攻击者的页面发出智能体用用户 cookie 执行的状态更改请求。
5. **一键劫持。** 视觉上无害的按钮搭载智能体遵循的载荷。Comet 类。
6. **智能体宿主面的内容安全策略漏洞。** 渲染和工具层本身可以是攻击向量；浏览器-在-浏览器-智能体栈是很宽的。

### 为什么"不可完全修补"

攻击与智能体的能力同构。智能体必须阅读不受信任的内容才能完成工作。智能体阅读的任何内容都可能包含指令。智能体遵循的任何指令都可能与用户的实际请求不对齐。防御（信任边界、分类器、工具允许列表、重要动作的人机协同）提高了攻击的成本并减少了爆炸半径。它们不封闭此类攻击。

这与 Lob 定理（第 8 课）的推理模式相同：智能体无法证明下一个 token 是安全的；它只能建立一个系统，使不安全的 token 更可检测。

### 实际部署的防御姿态

- **读/写边界。** 阅读永远不是重要的。写入（提交表单、发布内容、调用有副作用的工具）如果发起内容来自信任边界之外，需要新的人工批准。
- **每任务工具允许列表。** 智能体可以浏览；除非该工具被明确启用，否则不能发起电汇。第 13 课涵盖预算。
- **会话隔离。** 浏览器智能体会话仅以限定范围的凭证运行。无生产认证，无个人邮件。每个 HTTP 请求的日志保留供审计。
- **内容净化器。** 获取的 HTML 在拼接到模型上下文之前剥离已知恶意模式。（减少简单攻击；不阻止复杂的载荷。）
- **重要动作的人机协同。** 先提议后提交模式（第 15 课）。
- **内存上的金丝雀 token。** 如果内存条目触发，用户会看到它（第 14 课）。

## 开始构建

`code/main.py` 模拟一个小型浏览器智能体对三个合成页面的运行。一个页面是良性的，一个在可见文本中有直接提示注入块，一个有 URL 片段注入（不可见但在智能体上下文中）。脚本展示（a）朴素智能体会做什么，（b）读/写边界捕获什么，（c）净化器捕获什么，（d）两者都未捕获的。

## 交付产出

`outputs/skill-browser-agent-trust-boundary.md` 为拟议的浏览器智能体部署确定范围：它触及哪些信任区域，它被授权写什么，以及首次运行前必须具备哪些防御。

## 练习

1. 运行 `code/main.py`。识别净化器捕获但读/写边界未捕获的攻击，以及只有读/写边界捕获的攻击。

2. 扩展净化器以检测一类 HashJack 风格的 URL 片段注入。在带有合法片段的良性 URL 上测量误报率。

3. 选择一个你了解的真实浏览器智能体工作流（例如"订机票"）。列出每次读取和每次写入。标记哪些写入需要人机协同及原因。

4. 阅读 WebArena-Verified ICLR 2026 论文。识别原始 WebArena 评分不可靠的一个任务类别，并解释 Verified 子集如何解决它。

5. 为浏览器智能体场景设计一个内存金丝雀。你会存储什么，存在哪里，什么触发警报？

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着什么 |
|---|---|---|
| Indirect prompt injection（间接提示注入） | "恶意页面文本" | 智能体阅读的页面中不受信任的内容包含智能体执行的指令 |
| Tainted Memories（污点内存） | "内存攻击" | 智能体将攻击者提供的指令写入持久化内存；下次会话触发 |
| HashJack | "URL 片段攻击" | 隐藏在 URL 片段/查询字符串中的载荷在智能体上下文中但不可见渲染 |
| One-click hijack（一键劫持） | "恶意按钮" | 可见的交互元素搭载智能体执行的后续载荷 |
| BrowseComp | "Web 搜索基准" | 在开放网络上查找特定事实；分钟级时域 |
| OSWorld | "桌面基准" | 完整操作系统控制；多步骤 GUI 任务 |
| WebArena-Verified | "修复后的 Web 任务基准" | ServiceNow 重新评分的 WebArena，含 Hard 子集 |
| Read/write boundary（读/写边界） | "副作用门" | 阅读永远不是重要的；如果内容超出信任，写入需要新批准 |

## 延伸阅读

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 和深度研究的合并；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 谱系和演变为 ChatGPT agent 的架构。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始基准。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修复子集论文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含计算机使用智能体的攻击面讨论。
