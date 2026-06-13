# 工具使用与函数调用

> Toolformer（Schick 等人，2023）开启了自监督工具标注。Berkeley Function Calling Leaderboard V4（Patil 等人，2025）设定了 2026 年的标准：40% 智能体、30% 多轮、10% 实时、10% 非实时、10% 幻觉。单轮调用已经解决。记忆、动态决策和长周期工具链还没有。

**类型：** 构建
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环）、第 13 阶段 · 01（函数调用深入）
**所需时间：** 约60分钟

## 学习目标

- 解释 Toolformer 的自监督训练信号：仅当执行降低下一个 token 的损失时才保留工具标注。
- 说出 BFCL V4 的五个评估类别及其各自的衡量内容。
- 用标准库实现一个带 schema 验证、参数强制转换和执行沙箱的工具注册表。
- 诊断 2026 年的三个开放问题：长周期工具链、动态决策和记忆。

## 问题所在

早期工具使用问的是：模型能预测正确的函数调用吗？现代工具使用问的是：模型能在 40 步中链式调用工具、带记忆、带部分可观察性、从工具失败中恢复、且不幻觉出不存在的工具吗？

Toolformer 建立了基线：模型可以通过自监督学习何时调用工具。BFCL V4 定义了 2026 年的评估目标。两者之间的差距就是生产智能体所处的空间。

## 概念说明

### Toolformer（Schick 等人，NeurIPS 2023）

思路：让模型在自己的预训练语料上标注候选 API 调用。对每个候选，执行它。仅当包含工具结果能降低下一个 token 的损失时才保留标注。在过滤后的语料上微调。

涵盖的工具：计算器、QA 系统、搜索引擎、翻译器、日历。自监督信号完全关于工具是否有助于预测文本——无需人工标注。

规模结果：工具使用在规模上涌现。小模型因工具标注受损；大模型受益。这就是为什么 2026 年的前沿模型内置了强大的工具使用能力，而大多数 7B 模型需要显式的工具使用微调才能可靠。

### Berkeley Function Calling Leaderboard V4（Patil 等人，ICML 2025）

BFCL 是 2026 年的事实标准评估。V4 组成：

- **智能体（40%）**——完整的智能体轨迹：记忆、多轮、动态决策。
- **多轮（30%）**——带工具链的交互式对话。
- **实时（10%）**——用户提交的真实提示（更难的分布）。
- **非实时（10%）**——合成测试用例。
- **幻觉（10%）**——检测何时不应调用工具。

V3 引入了基于状态的评估：在工具序列之后，检查 API 的实际状态（如"文件是否创建了？"）而非匹配工具调用的 AST。V4 添加了网页搜索、记忆和格式敏感性类别。

2026 年关键发现：单轮函数调用接近解决。失败集中在记忆（跨轮传递上下文）、动态决策（基于先前结果选择工具）、长周期链（20+ 步后漂移）和幻觉检测（拒绝在没有合适工具时调用）。

### 工具 schema

每个提供商都有一个 schema。细节不同但形状相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接使用 `input_schema`。OpenAI 使用 `function.parameters`。两者都接受 JSON Schema。描述是承重的——模型读取它们来选择正确的工具。糟糕的工具描述是选错工具的头号根本原因。

### 参数验证

不要信任任何工具调用。验证：

1. **类型强制转换。** 模型可能返回字符串 "5" 而 schema 要求 int。如果无歧义则强制转换；否则拒绝。
2. **枚举验证。** 如果 schema 规定 `status in {"open", "closed"}` 而模型发出 `"in_progress"`，以描述性错误拒绝。
3. **必填字段。** 缺少必填字段 -> 立即返回错误观察给模型，而非崩溃。
4. **格式验证。** 日期、邮箱、URL——用具体解析器验证，而非正则表达式。

每个验证失败都应返回结构化观察，以便模型用正确的形状重试。

### 并行工具调用

现代提供商支持在一个助手轮次中并行工具调用。循环：

1. 模型发出 3 个带不同 `tool_use_id` 的工具调用。
2. 运行时执行它们（如果独立则并行）。
3. 每个结果作为 `tool_result` 块通过 `tool_use_id` 关联回去。

工程规则：将关联 ID 视为承重的。交换它们你会得到错误工具对应错误结果的路由。

### 沙箱

工具执行是沙箱边界。详见第 09 课。简版：每个工具应指定读/写范围、网络访问、超时、内存上限。通用的 `run_shell(cmd)` 是红旗；具体的 `git_status()` 更安全。

```figure
tool-routing
```

## 开始构建

`code/main.py` 实现了一个生产形态的工具注册表：

- JSON Schema 子集验证器（仅标准库）。
- 带描述、输入 schema、超时和执行器的工具注册。
- 参数强制转换和枚举验证。
- 带关联 ID 的并行工具分派。
- 结构化字符串形式的错误观察。

运行它：

```
python3 code/main.py
```

轨迹显示一个小型智能体在一轮中调用三个工具，其中一个故意格式错误的调用被拒绝，返回了模型可以据此行动的描述性错误。

## 使用它

每个提供商都有自己的工具 schema——Anthropic、OpenAI、Gemini、Bedrock。如果需要多提供商支持，使用翻译层（OpenAI Agents SDK、Vercel AI SDK、LangChain 工具适配器）。BFCL 是参考基准——如果工具使用是产品核心，在发布前针对你的智能体运行它。

## 交付它

`outputs/skill-tool-registry.md` 为给定任务领域生成工具目录、schema 和注册表。包含描述质量检查（每个工具的描述是否告诉模型何时使用它？）。

## 练习

1. 添加一个"空操作"工具，让模型显式拒绝使用任何其他工具。在类似 BFCL 的幻觉测试上衡量。
2. 为 int-as-string 和 float-as-string 实现参数强制转换。强制转换从哪里开始隐藏真正的 bug？
3. 添加每个工具的超时和断路器（连续 3 次失败后拒绝工具 60 秒）。这对模型恢复能力有什么改变？
4. 阅读 BFCL V4 描述。选择一个类别（如"多轮"）在你的智能体上运行 10 个示例提示。报告通过率。
5. 将标准库验证器移植到 Pydantic 或 Zod。Pydantic/Zod 捕获了玩具实现遗漏的什么？

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| 函数调用（Function calling） | "工具使用" | 带验证 schema 的结构化输出工具调用 |
| Toolformer | "自监督工具标注" | Schick 2023 年——保留结果降低下一个 token 损失的工具调用 |
| BFCL | "Berkeley Function Calling Leaderboard" | 2026 年基准：40% 智能体、30% 多轮、10% 实时、10% 非实时、10% 幻觉 |
| 工具 schema（Tool schema） | "模型的函数签名" | 名称、描述、参数的 JSON Schema |
| tool_use_id | "关联 ID" | 将工具调用与其结果关联；对并行分派至关重要 |
| 幻觉检测（Hallucination detection） | "知道何时不调用" | V4 类别：在没有合适工具时拒绝调用 |
| 参数强制转换（Argument coercion） | "字符串转 int 修复" | 对可预测 schema 不匹配的窄修复；有歧义则拒绝 |
| 沙箱（Sandboxing） | "工具执行边界" | 每个工具的读/写范围、网络、超时、内存上限 |

## 延伸阅读

- [Schick 等人，Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761)——自监督工具标注
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html)——2026 年评估基准
- [Anthropic，Tool use 文档](https://platform.claude.com/docs/en/agent-sdk/overview)——Claude Agent SDK 中的生产工具 schema
- [OpenAI Agents SDK 文档](https://openai.github.io/openai-agents-python/)——function 工具类型和 Guardrails
