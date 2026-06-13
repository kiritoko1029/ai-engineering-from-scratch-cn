# 结构化输出 -- JSON Schema、Pydantic、Zod、约束解码

> "好好跟模型说让它返回 JSON"的失败率在前沿模型上仍有 5% 到 15%。结构化输出通过约束解码来弥合这个差距：模型在解码层面被阻止输出违反模式的 token。OpenAI 的严格模式、Anthropic 的模式类型化工具使用、Gemini 的 `responseSchema`、Pydantic AI 的 `output_type` 和 Zod 的 `.parse` 是同一个思想的五种表面形式。本课构建学习者在每个生产提取管道中都会用到的模式验证器和严格模式契约。

**类型：** 构建
**语言：** Python（标准库，JSON Schema 2020-12 子集）
**前置要求：** 第 13 阶段第 02 课（函数调用深入解析）
**所需时间：** 约75分钟

## 学习目标

- 使用正确的约束（enum、min/max、required、pattern）为提取目标编写 JSON Schema 2020-12。
- 解释严格模式和约束解码与"生成后验证"相比提供了什么不同的保证。
- 区分三种失败模式：解析错误、模式违规、模型拒绝。
- 发布一个带有类型化修复和类型化拒绝处理的提取管道。

## 问题所在

一个读取采购订单邮件的智能体需要将自由文本转换为 `{customer, line_items, total_usd}`。有三种方法。

**方法一：提示模型输出 JSON。** "以 JSON 格式回复，包含字段 customer、line_items、total_usd。"在前沿模型上 85% 到 95% 的情况下有效。六种失败方式：缺少花括号、多余逗号、类型错误、幻觉字段、在 token 限制处截断、泄露散文如"Here is your JSON:"。

**方法二：生成后验证。** 自由生成，解析，根据模式验证，失败时重试。可靠但昂贵 -- 每次重试都要花钱，截断 bug 每次额外消耗一轮。

**方法三：约束解码。** 供应商在解码时强制执行模式。无效 token 从采样分布中被屏蔽。输出保证可解析且保证可验证。失败收束为一种模式：拒绝（模型判断输入不符合模式）。

2026 年每个前沿供应商都发布了某种形式的方法三。

- **OpenAI。** `response_format: {type: "json_schema", strict: true}`，如果模型拒绝则响应中包含 `refusal`。
- **Anthropic。** 对 `tool_use` 输入强制执行模式；`stop_reason: "refusal"` 并不存在，但 `end_turn` 不带工具调用就是信号。
- **Gemini。** 请求级别的 `responseSchema`；2026 年 Gemini 为选定类型发布了 token 级别的语法约束。
- **Pydantic AI。** `output_type=InvoiceModel` 输出一个类型化为 `InvoiceModel` 的结构化 `RunResult`。
- **Zod（TypeScript）。** 运行时解析器，根据 Zod 模式验证供应商输出；与 OpenAI 的 `beta.chat.completions.parse` 配合使用。

共同点：声明一次模式，端到端强制执行。

## 概念说明

### JSON Schema 2020-12 -- 通用语言

每个供应商都接受 JSON Schema 2020-12。你最常用的构造：

- `type`：`object`、`array`、`string`、`number`、`integer`、`boolean`、`null` 之一。
- `properties`：字段名到子模式的映射。
- `required`：必须出现的字段名列表。
- `enum`：允许值的封闭集合。
- `minimum` / `maximum`（数字），`minLength` / `maxLength` / `pattern`（字符串）。
- `items`：应用于每个数组元素的子模式。
- `additionalProperties`：`false` 禁止额外字段（默认行为因模式而异）。

OpenAI 严格模式增加了三个要求：每个属性都必须列在 `required` 中，所有地方都必须 `additionalProperties: false`，不能有未解析的 `$ref`。如果违反这些，API 在请求时返回 400。

### Pydantic，Python 绑定

Pydantic v2 通过 `model_json_schema()` 从数据类形式的模型生成 JSON Schema。Pydantic AI 封装了这一功能，你只需编写：

```python
class Invoice(BaseModel):
    customer: str
    line_items: list[LineItem]
    total_usd: Decimal
```

智能体框架就会在边界层将模式转换为 OpenAI 严格模式、Anthropic `input_schema` 或 Gemini `responseSchema`。模型的输出以类型化的 `Invoice` 实例返回。验证错误抛出 `ValidationError`，带有类型化的错误路径。

### Zod，TypeScript 绑定

Zod（`z.object({customer: z.string(), ...})`）是 TypeScript 的等价物。OpenAI 的 Node SDK 暴露了 `zodResponseFormat(Invoice)`，将其转换为 API 的 JSON Schema 载荷。

### 拒绝

严格模式无法强制模型回答。如果输入无法适配模式（"邮件是一首诗，不是发票"），模型会输出一个包含原因的 `refusal` 字段。你的代码必须将其作为一等结果来处理，而不是失败。拒绝也可用作安全信号：一个被要求从受保护内容邮件中提取信用卡号的模型会返回带有安全原因的拒绝。

### 开源中的约束解码

开源权重实现使用三种技术。

1. **基于语法的解码**（`outlines`、`guidance`、`lm-format-enforcer`）：从模式构建确定性有限自动机；在每一步，屏蔽会违反 FSM 的 token 的 logits。
2. **使用 JSON 解析器进行 logit 屏蔽**：与模型同步运行流式 JSON 解析器；在每一步，计算有效下一个 token 集合。
3. **带验证器的投机解码**：廉价草稿模型提出 token，验证器强制执行模式。

商业供应商在幕后选择其中一种。2026 年的技术水平对短结构化输出比普通生成更快，对长输出大致相同。

### 三种失败模式

1. **解析错误。** 输出不是有效的 JSON。严格模式下不可能发生。在非严格供应商上仍然可能发生。
2. **模式违规。** 输出可解析但违反模式。严格模式下不可能发生。在严格模式外很常见。
3. **拒绝。** 模型拒绝。必须作为类型化结果处理。

### 重试策略

当你在严格模式之外（Anthropic 工具使用、非严格 OpenAI、旧版 Gemini）时，恢复模式是：

```
generate -> parse -> validate -> if fail, inject error and retry, max 3x
```

一次重试通常足够。三次重试捕获弱模型的偶发失败。超过三次说明模式有问题：模型对某些输入无法满足它，提示或模式需要修复。

### 小模型支持

约束解码在小模型上有效。一个带语法强制的 30 亿参数开源模型在结构化任务上的表现优于带原始提示的 700 亿参数模型。这是结构化输出对生产环境很重要的主要原因：它将可靠性与模型大小解耦。

## 开始构建

`code/main.py` 在标准库中发布了一个最小的 JSON Schema 2020-12 验证器（类型、required、enum、min/max、pattern、items、additionalProperties）。它封装了一个 `Invoice` 模式，通过验证器运行模拟的 LLM 输出，演示解析错误、模式违规和拒绝路径。在生产中用任何供应商的真实响应替换模拟输出即可。

需要关注的要点：

- 验证器返回一个类型化的 `[ValidationError]` 列表，包含路径和消息。这就是你希望在重试提示中展示的形式。
- 拒绝分支不会重试。它记录并返回类型化的拒绝。第 14 阶段第 09 课将拒绝用作安全信号。
- `additionalProperties: false` 检查在对抗性测试输入上触发，展示了为什么严格模式能堵住幻觉字段。

## 发布成果

本课生成 `outputs/skill-structured-output-designer.md`。给定一个自由文本提取目标（发票、支持工单、简历等），该技能生成一个兼容严格模式的 JSON Schema 2020-12 和一个镜像它的 Pydantic 模型，并提供了类型化拒绝和重试处理的存根代码。

## 练习

1. 运行 `code/main.py`。添加一个 `total_usd` 为负数的第四个测试用例。确认验证器通过 `minimum` 约束路径拒绝它。

2. 扩展验证器以支持带鉴别器的 `oneOf`。常见场景：`line_item` 是产品或服务，由 `kind` 标记。严格模式在这里有微妙的规则；查看 OpenAI 的结构化输出指南。

3. 将同一个 Invoice 模式编写为 Pydantic BaseModel，比较 `model_json_schema()` 输出与你手写的模式。找出 Pydantic 默认设置而手写版本遗漏的一个字段。

4. 测量拒绝率。构造十个不应可提取的输入（歌词、数学证明、空白邮件），在严格模式下通过真实供应商运行它们。统计拒绝与幻觉输出的数量。这是拒绝感知重试的基础真值。

5. 从头到尾阅读 OpenAI 的结构化输出指南。找出它在严格模式下明确禁止而普通 JSON Schema 允许的一个构造。然后设计一个非必要地使用该禁止构造的模式，并将其重构为严格兼容的版本。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|-----------|---------|
| JSON Schema 2020-12 | "模式规范" | IETF 草案模式方言，每个现代供应商都支持 |
| 严格模式（Strict mode） | "保证模式" | OpenAI 标志，通过约束解码强制执行模式 |
| 约束解码（Constrained decoding） | "Logit 屏蔽" | 解码时强制，屏蔽无效的下一个 token |
| 拒绝（Refusal） | "模型拒绝" | 输入无法适配模式时的类型化结果 |
| 解析错误（Parse error） | "无效 JSON" | 输出未能解析为 JSON；严格模式下不可能 |
| 模式违规（Schema violation） | "形式错误" | 可解析但违反类型 / required / enum / 范围 |
| `additionalProperties: false` | "不允许额外字段" | 禁止未知字段；OpenAI 严格模式必需 |
| Pydantic BaseModel | "类型化输出" | 生成和验证 JSON Schema 的 Python 类 |
| Zod schema | "TypeScript 输出类型" | 用于供应商输出验证的 TS 运行时模式 |
| 语法强制（Grammar enforcement） | "开源约束解码" | 基于 FSM 的 logit 屏蔽，如 outlines / guidance |

## 延伸阅读

- [OpenAI -- Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) -- 严格模式、拒绝和模式要求
- [OpenAI -- Introducing structured outputs](https://openai.com/index/introducing-structured-outputs-in-the-api/) -- 2024 年 8 月发布文章，解释解码保证
- [Pydantic AI -- Output](https://ai.pydantic.dev/output/) -- 类型化 output_type 绑定，序列化到每个供应商
- [JSON Schema -- 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) -- 权威规范
- [Microsoft -- Structured outputs in Azure OpenAI](https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/structured-outputs) -- 企业部署说明和严格模式注意事项
