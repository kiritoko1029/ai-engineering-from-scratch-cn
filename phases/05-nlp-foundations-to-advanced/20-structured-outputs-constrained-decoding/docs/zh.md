# 结构化输出与受限解码

> 向大语言模型请求 JSON 数据。大多数情况下都能获取到 JSON。但在生产环境中，“大多数”这一情况本身就构成了问题。通过在对样本进行采样之前编辑对数概率值，可实现受限解码，从而将“大多数”变为“始终”。  

**类型：** 构建  
**编程语言：** Python  
**先修要求：** 第 5 阶段 · 17（聊天机器人），第 5 阶段 · 19（子词分词）  
**耗时：** 约 60 分钟

## 问题所在

分类器会向大语言模型发出指令：“返回 {positive, negative, neutral} 中的一个值。”但模型却回复道：“情感为正面——这条评论极其积极，因为用户明确表示他们……”。此时解析器会崩溃，分类器的 F1 分数也会降至 0.0。

自由形式的生成结果并非具有约束力的契约，而仅是一种建议。生产环境中的系统则需要明确的契约。

2026 年时存在三层解决方案：

1. **提示工程。**以恰当的方式提出请求，例如“仅返回 JSON 对象”。在前沿模型上成功率约为 80%，在较小模型上的成功率则较低。
2. **原生结构化输出 API。**如 OpenAI 的 `response_format`、Anthropic 提供的工具以及 Gemini 的 JSON 模式。在支持相应数据结构的场景下表现稳定，但存在供应商锁定问题。
3. **受限解码技术。**在每次生成步骤中修改模型的对数概率值，从而确保模型无法输出无效的标记。从设计上即可保证 100% 的有效性，适用于任何本地部署的模型。

本课程将帮助学习者建立对这三种方法的直观理解，并明确在不同场景下应选择哪种方法。

## 概念概述

![在每一步中对无效标记进行受限解码屏蔽](../assets/constrained-decoding.svg)

**受限解码的工作原理。** 在每个生成步骤中，大语言模型会为整个词汇表（约10万个标记）生成一个对数几率向量。位于模型与采样器之间的*对数几率处理器*会根据目标语法结构——如JSON Schema、正则表达式或上下文无关文法——判断哪些标记是有效的，并将所有无效标记的对数几率设为负无穷大。随后通过对剩余有效对数几率应用softmax函数，即可使概率仅集中在合法的后续内容上。

2026年的实现方案包括：

- **Outlines。** 将JSON Schema或正则表达式编译为有限状态机。每个标记都可以进行O(1时间复杂度的有效下一个标记查询）。由于基于有限状态机，因此递归结构的Schema需要先被扁平化处理。
- **XGrammar / llguidance。** 上下文无关文法引擎，能够处理递归式的JSON Schema，解码开销几乎为零。OpenAI在2025年的结构化输出实现中引用了llguidance的技术。
- **vLLM引导式解码。** 通过Outlines、XGrammar或lm-format-enforcer后端，提供内置的`guided_json`、`guided_regex`、`guided_choice`、`guided_grammar`等功能。
- **Instructor。** 基于Pydantic的通用大语言模型封装工具，在验证失败时自动重试。支持跨不同提供商使用，但不会修改对数几率——它依靠重试机制以及针对结构化输出的提示词来实现目标。

### 这一违反直觉的结果

受限解码通常比无约束生成*更快*。原因有二：首先，它缩小了下一个标记的搜索空间；其次，巧妙的实现方式会完全跳过强制标记的生成过程（例如像`{"name": "}`这样的骨架结构——每个字节都已确定）。

### 会让你付出代价的陷阱

字段顺序至关重要。需将 `answer` 放在 `reasoning` 之前，这样模型会在思考之前就确定答案。JSON格式有效，但给出的答案却是错误的，且现有的验证机制无法检测到这一问题。

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

模式字段的顺序是逻辑上的排列，而非格式上的要求。

## 构建它

### 步骤 1：从零开始进行正则表达式约束的生成

独立的 FSM 实现参见 `code/main.py`。其核心思想仅用 30 行代码即可阐述：

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

状态机用于记录我们目前已满足的语法条件。`valid_tokens(state, tokenizer)` 函数用于判断哪些词汇表中的标记能够在不偏离接受路径的情况下推动状态机前进。

### 步骤 2：JSON Schema 的结构概要

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

零验证错误。永远如此。状态机确保无效输出无法产生。

### 步骤 3：适用于所有提供方的 Pydantic 教学指导

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

采用不同的机制。讲师无需直接操作对数概率值，只需将结构化模式整合到提示词中，解析输出结果，并在验证失败时自动重试（默认次数为3次）。该方案适用于任何服务提供商。多次重试会增加延迟与成本，而跨提供商的兼容性正是其核心优势所在。

### 步骤 4：原生供应商 API

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

服务器端受限解码。针对支持的架构，具备与 Outlines 相同的可靠性。不支持本地模型管理。这会限制您只能使用该供应商的产品。

## 常见陷阱

- **递归模式。** Outline 会将递归结构扁平化为固定深度。对于树形结构的输出（如嵌套注释、AST），则需要使用 XGrammar 或基于 CFG 的 llguidance。
- **大型枚举类型。** 包含 10,000 个选项的枚举类型在编译时速度会很慢，甚至会导致超时。建议改用检索器：先预测出前 k 个候选值，再从中进行筛选。
- **语法过于严格。** 若强制要求使用 `date: "YYYY-MM-DD"` 这样的正则表达式格式，模型在遇到缺失日期时将无法输出 `"unknown"`，只能自行编造一个日期。应允许使用 `null` 或特定的占位值。
- **过早确定结构。** 请注意上述字段顺序相关的陷阱。始终应先进行推理，再确定结构。
- **无模式的供应商 JSON 模式。** 纯 JSON 模式仅能保证数据为有效的 JSON 格式，并不能确保其适用于您的具体使用场景。务必提供完整的模式定义。

## 使用它

2026年技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 使用 OpenAI/Anthropic/Google 的模型，且数据结构简单 | 厂商提供的原生结构化输出 |
| 任意提供商的模型，采用 Pydantic 工作流，可容忍重试操作 | Instructor 工具 |
| 使用本地模型，要求100%的数据有效性，且数据结构为扁平型 | Outlines (FSM) |
| 使用本地模型，且数据结构为递归型 | XGrammar 或 llguidance |
| 自托管推理服务器 | vLLM 引导的解码方式 |
| 需要批量处理且可容忍重试操作 | Instructor 工具 + 成本最低的模型 |

## 发布它

保存为 `outputs/skill-structured-output-picker.md`：

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## 练习题

1. **简单。** 使用小型开源模型（例如 Llama-3.2-3B），不对解码过程进行限制，对 `Review(sentiment, confidence, evidence_span)` 进行处理。在 100 条评论中统计能解析为有效 JSON 的比例。
2. **中等。** 使用相同的语料库，但采用 Outlines JSON 模式。比较合规率、延迟以及语义准确性。
3. **困难。** 从零开始实现一个受正则表达式限制的解码器，用于处理电话号码格式（`\d{3}-\d{3}-\d{4}`）。在 1000 个样本中确保没有无效输出。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 受限解码 | 强制生成有效输出 | 在每一轮生成步骤中对无效令牌的logit值进行屏蔽。 |
| Logit处理器 | 执行限制功能的组件 | 函数形式：`(logits, state) -> masked_logits`。 |
| FSM | 有限状态机 | 编译后的语法表示形式；可实现O(1)时间复杂度的有效下一个令牌查询。 |
| CFG | 上下文无关文法 | 能够处理递归结构的语法；虽然速度较慢，但表达能力优于FSM。 |
| 模式字段顺序 | 重要吗？ | 很重要——第一个字段会决定最终结果；务必将推理过程放在答案之前。 |
| 引导解码 | vLLM对该功能的命名 | 同一概念，已集成到推理服务器中。 |
| JSON模式 | OpenAI早期的版本 | 仅保证输出为JSON格式；不保证与模式结构完全匹配。 |

## 延伸阅读

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — 该论文提出了高效引导式生成方法。
- [XGrammar 论文（2024）](https://arxiv.org/abs/2411.15100) — 基于 CFG 的快速受限解码技术。
- [vLLM — 结构化输出功能](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — 推理服务器集成方案。
- [OpenAI — 结构化输出指南](https://platform.openai.com/docs/guides/structured-outputs) — API 参考文档及注意事项。
- [Instructor 库](https://python.useinstructor.com/) — 支持 Pydantic 语法及跨服务重试机制。
- [JSONSchemaBench（2025）](https://arxiv.org/abs/2501.10868) — 对 6 种受限解码框架的基准测试。
