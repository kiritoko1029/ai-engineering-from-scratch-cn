# FIPA-ACL 与言语行为的传承

> 在 MCP 之前，在 A2A 之前，就有了 FIPA-ACL。2000 年，IEEE 智能物理代理基金会批准了一种智能体通信语言，包含二十个施为句、两种内容语言和一组交互协议——合同网、订阅/通知、条件请求。它从工业界淡出，因为本体论开销对 Web 来说太重了，但多智能体系统的 LLM 复兴正在悄然重新实现相同的理念：JSON 契约代替施为句，自然语言代替本体论。本课认真研读 FIPA-ACL，以便你能看清 2026 年的哪些协议决策是重新发明，哪些是真正的创新，以及当前浪潮将在何处重新发现 2000 年代已经解决的问题。

**类型：** 学习
**语言：** Python（标准库）
**前置课程：** 第 16 阶段 · 01（为什么需要多智能体）
**时间：** 约 60 分钟

## 问题

2026 年的智能体协议格局非常繁忙：MCP 用于工具，A2A 用于智能体间协作，ACP 用于企业审计，ANP 用于去中心化信任，NLIP 用于自然语言内容，加上 CA-MCP 和二十多个研究提案。每个规范都宣称自己是基础性的。

坦率地说，它们中的大多数都在重新发现一个非常具体的二十年前的决策树。Austin（1962）和 Searle（1969）的言语行为理论给了我们"话语即行动"。KQML（1993）将其转化为线路协议。FIPA-ACL（2000 年批准）产生了参考标准：二十个施为句、内容语言 SL0/SL1、合同网和订阅-通知的交互协议。JADE 和 JACK 是 Java 参考平台。这项工作在 2010 年左右淡出，因为本体论开销太重，而 Web 正在获胜。

当你查看 MCP 的 `tools/call`、A2A 的任务生命周期或 CA-MCP 的共享上下文存储时，你看到的是 FIPA 决策的更柔和、JSON 原生的翻版。了解这段传承告诉你两件事：哪些新"创新"实际上是重新发明，以及新规范将重新发现哪些旧的失败模式。

## 概念

### 言语行为，一段话概括

Austin 注意到，有些句子并不描述世界——它们改变世界。"我承诺。""我请求。""我宣布。"他称这些为施为性话语。Searle 将其形式化为五个类别：断言、指令、承诺、表达、宣告。KQML（Finin 等，1993）使这对软件智能体具有可操作性：一条消息是一个施为句（行动）加上内容（行动所涉及的对象）。FIPA-ACL 清理了 KQML 的缺陷，并围绕二十个施为句进行了标准化。

### 二十个 FIPA 施为句（部分列表）

| 施为句 | 意图 |
|--------|------|
| `inform` | "我告诉你 P 为真" |
| `request` | "我要求你做 X" |
| `query-if` | "P 是否为真？" |
| `query-ref` | "X 的值是什么？" |
| `propose` | "我提议我们做 X" |
| `accept-proposal` | "我接受该提案" |
| `reject-proposal` | "我拒绝该提案" |
| `agree` | "我同意做 X" |
| `refuse` | "我拒绝做 X" |
| `confirm` | "我确认 P 为真" |
| `disconfirm` | "我否认 P" |
| `not-understood` | "你的消息无法解析" |
| `cfp` | "关于 X 的招标" |
| `subscribe` | "当 X 变化时通知我" |
| `cancel` | "取消正在进行的 X" |
| `failure` | "我尝试了 X 但失败了" |

完整列表在 `fipa00037.pdf`（FIPA ACL 消息结构）中。重点不是背诵它——重点是每一个施为句都对应一个 LLM 协议最终会重新添加的原语。

### 规范的 FIPA-ACL 消息

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

七个字段承载协议信封；一个字段（`content`）承载负载。其余字段正是你每次将重试、线程和本体论嫁接到 JSON 协议上时都会重新发明的东西。

### 两个遗留平台

**JADE**（Java Agent DEvelopment framework，1999-2020 年代）是最常用的 FIPA 兼容运行时。智能体扩展基类、交换 ACL 消息、在容器内运行，并使用"行为"进行协调。交互协议库附带了合同网、订阅-通知、条件请求和提议-接受。

**JACK**（Agent Oriented Software，商业产品）强调在 FIPA 消息之上进行 BDI（信念-愿望-意图）推理。更正式，但采用率较低。

一旦 Web 技术栈吞噬了多智能体用例，两者都衰落了。MCP 和 A2A 是 2026 年的运行时"容器"。

### FIPA 为何淡出

- **本体论开销。** FIPA 要求共享本体论来解析 `content`。就本体论达成一致是一个长达数年的标准制定过程。Web 只使用 HTTP + JSON。
- **没人使用的形式语义。** SL（语义语言）提供了严格的真值条件，但大多数生产系统使用自由格式的内容并忽略形式化。
- **工具锁定。** JADE 仅限 Java；JACK 是商业产品。多语言团队绕过了两者。
- **互联网赢得了技术栈。** REST，然后 JSON-RPC，然后 gRPC 取代了 ACL 的传输层。

### LLM 复兴就是 FIPA-lite

将 FIPA `request` 与 MCP `tools/call` 进行比较：

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

相同的信封，不同的语法。两者都携带：谁、向谁、意图、负载、关联 ID。两者都不是对对方的革命——它们是同一设计上的不同权衡。

Liu 等人 2025 年的调查（"A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP"，arXiv:2505.02279）明确展示了这种谱系：MCP 对应工具使用言语行为，A2A 对应智能体对等言语行为，ACP 对应审计追踪言语行为，ANP 对应去中心化身份扩展。新规范是具有 JSON 语法和更宽松语义的 ACL 后裔。

### 权衡，直言不讳

**FIPA 给你而现代规范丢弃的：**

- 形式语义——你可以证明 `implies` 发送者相信内容。
- 规范的施为句目录——你不必重新争论"我们是否应该有一个 `cancel`？"
- 数十年的交互协议模式——合同网、订阅-通知、提议-接受——具有已知的正确性属性。

**现代规范给你而 FIPA 没有的：**

- JSON 原生负载，兼容所有现代工具。
- LLM 无需手写本体论即可解释的自然语言内容。
- Web 技术栈传输（HTTP、SSE、WebSocket）。
- 通过自描述文档进行能力发现（MCP `listTools`、A2A Agent Card）。

更宽松的意图语义以换取更简单的实现。这就是确切的权衡。

### 值得移植的交互协议

FIPA 附带了约 15 个交互协议。有三个值得延续到 LLM 多智能体系统中：

1. **合同网协议（CNP）。** 管理者发出 `cfp`（招标）；投标者用 `propose` 响应；管理者接受/拒绝。这是经典的市场任务模式（第 16 阶段 · 16 谈判）。
2. **订阅/通知。** 订阅者发送 `subscribe`；每当主题变化时发布者发送 `inform`。这就是 2026 年的每个事件总线。
3. **条件请求（Request-When）。** "当条件 Y 成立时执行 X。"带前置条件的延迟动作。2026 年的类似物是持久工作流引擎中的延迟任务（第 16 阶段 · 22 生产扩展）。

每个都可以清晰地映射到现代消息队列、HTTP + 轮询或 SSE 流。

### 丢弃本体论后会出什么问题

没有共享本体论，智能体从自然语言内容中推断含义。2026 年有记录的失败模式是**语义漂移**：两个智能体对微妙不同的概念使用同一个词（`"customer"`），接收方智能体基于错误的解释行事，没有模式验证器能捕获它。FIPA 的本体论要求会在解析时拒绝该消息。

不采用完整本体论的缓解措施：

- 对 `content` 使用 JSON Schema——在线路层面拒绝结构错误。
- 类型化工件（A2A）——拒绝错误的模态。
- 在信封中显式施为句——即使内容是自然语言，也能使意图明确。

### 2026 年规范，映射到言语行为传承

| 现代规范 | FIPA 类比物 | 保留了什么 | 丢弃了什么 |
|---------|-----------|-----------|-----------|
| MCP `tools/call` | `request` | 显式意图、关联 ID | 形式语义、本体论 |
| MCP `resources/read` | `query-ref` | 显式意图、关联 ID | 形式语义 |
| A2A 任务生命周期 | 合同网 + 条件请求 | 异步生命周期、状态转换 | 形式完整性保证 |
| A2A 流式事件 | 订阅/通知 | 异步推送 | 类型化谓词订阅 |
| CA-MCP 共享上下文 | 黑板（Hayes-Roth 1985） | 多写者共享内存 | 逻辑一致性模型 |
| NLIP | 自然语言内容 | LLM 原生 | 模式 |

从上到下阅读表格，模式是：保留结构原语，丢弃形式化，让 LLM 弥合歧义。

## 动手构建

`code/main.py` 实现了一个纯标准库的 FIPA-ACL 翻译器。它编码和解码规范的 ACL 信封，并展示每个 MCP / A2A 消息形状如何归约为相同的七个字段。演示：

- 将五条 MCP 风格和 A2A 风格的消息编码为 FIPA-ACL。
- 将 FIPA-ACL 解码回现代等价形式。
- 使用 `cfp`、`propose`、`accept-proposal`、`reject-proposal` 在一个管理者和三个投标者之间运行一个玩具合同网谈判。

运行：

```
python3 code/main.py
```

输出是一个并排追踪，展示每条现代消息在其 2026 JSON 形式和 FIPA-ACL 形式下的对比，然后是合同网投标的往返。相同的协议原语在往返中存活下来；只有语法不同。

## 应用

`outputs/skill-fipa-mapper.md` 是一个技能，它读取任何智能体协议规范并产生 FIPA-ACL 映射。在采用新协议之前使用它来回答："这是真正的新东西，还是只是换了 JSON 语法的 `inform`？"

## 交付

不要把 FIPA-ACL 带回来。带回它的检查清单：

- 每条消息的意图原语（施为句）是什么？
- 是否有关联 ID 用于请求-响应和取消？
- 是否有显式的内容语言（JSON-RPC、纯文本、结构化类型化工件）？
- 交互协议是一等公民，还是你在从头重新实现合同网？
- 当两个智能体对内容含义产生分歧时会发生什么（语义漂移）？

在将任何新协议交付生产之前，记录这五个问题。

## 练习

1. 运行 `code/main.py`。观察往返编码。识别哪个 FIPA 施为句对应 `tools/call`、`resources/read` 和 A2A 任务创建。
2. 用 `cancel` 施为句扩展合同网演示，让管理者在投标过程中撤回任务。`cancel` 解决了哪些仅靠重试无法解决的失败情况？
3. 阅读 FIPA ACL 消息结构（http://www.fipa.org/specs/fipa00037/）第 4.1-4.3 节。选择一个本课未涉及的施为句，描述其现代 JSON-RPC 类比物。
4. 阅读 Liu 等人，arXiv:2505.02279。对于 MCP、A2A、ACP、ANP，列出它们保留和丢弃的 FIPA 施为句家族。
5. 为你自己系统中 `request` 施为句的 `content` 字段设计一个最小的 JSON Schema。该模式给你提供了纯自然语言所没有的什么，代价是什么？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| 言语行为（Speech act） | "做某事的话语" | Austin/Searle：话语即行动。ACL 的理论前身。 |
| FIPA | "那个古老的 XML 东西" | IEEE 智能物理代理基金会。2000 年标准化了 ACL。 |
| ACL | "智能体通信语言" | FIPA 的信封格式：施为句 + 内容 + 元数据。 |
| 施为句（Performative） | "动词" | 消息的意图类别：`inform`、`request`、`propose`、`cfp` 等。 |
| KQML | "FIPA 的前身" | 知识查询与操作语言（1993）。更简单、更窄。 |
| 本体论（Ontology） | "共享词汇表" | 内容语言所谈论概念的形式化定义。 |
| SL0 / SL1 | "FIPA 内容语言" | 语义语言第 0 级和第 1 级——形式化内容语言家族。 |
| 合同网（Contract Net） | "任务市场" | 管理者发出 cfp；投标者提议；管理者接受。经典的交互协议。 |
| 交互协议（Interaction protocol） | "消息模式" | 具有已知正确性的施为句序列：条件请求、订阅-通知等。 |

## 延伸阅读

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — 将现代规范与 FIPA 传承联系起来的 2025 年权威综述
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 2000 年批准的信封格式
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — 完整的施为句目录
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — `request`/`query-ref` 的现代工具使用等价物
- [A2A specification](https://a2a-protocol.org/latest/specification/) — 合同网和订阅-通知的现代智能体对等等价物
