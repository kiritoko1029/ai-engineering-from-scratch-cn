# 对话状态跟踪

> “我想要一家位于北区的便宜餐厅……其实价格适中就好……还要有意大利菜。”经过三轮交互和三次状态更新后，DST会保持 `slot-value` 字典的同步，从而确保预订流程正常运行。

**类型：** 构建
**语言：** Python
**先决条件：** 第5阶段 · 17（聊天机器人），第5阶段 · 20（结构化输出）
**耗时：** 约75分钟

## 问题所在

在以任务为导向的对话系统中，用户的目标会被编码为一组槽值对：`{cuisine: italian, area: north, price: moderate}`。用户的每一条消息都可能新增、修改或删除某个槽位。系统必须完整读取整个对话内容，并准确输出当前的槽位状态。

只要有一个槽位的识别出现错误，系统就可能会预订错误的餐厅、安排错误的航班，或是从错误的银行卡扣款。DST（意图解析与状态转换）正是连接用户表述与后台执行动作的关键环节。

尽管有了大语言模型，为何在2026年它依然重要：

- 需要严格符合法规的行业（如银行业、医疗保健业、航空订票服务）要求的是确定的槽值，而非自由生成的文本。
- 工具型智能体在调用 API 之前仍需完成槽位解析。
- 多轮对话中的纠错比想象中更为困难：例如用户会说“其实不是，改成星期四”。

现代的处理流程为：传统的 DST 概念 + 大语言模型提取器 + 结构化输出约束机制。

## 概念概述

![DST：对话历史 → 槽值状态](../assets/dst.svg)

**任务结构。** 领域模式定义了不同的领域（餐厅、酒店、出租车）及其对应的槽位（菜系、区域、价格、人数）。每个槽位可以为空，也可以填入封闭集合中的某个值（例如价格：{便宜、中等、昂贵}），或者填写自由形式的文本（例如名称：“The Copper Kettle”）。

**两种DST实现方式。**

- **分类法。** 对于每一对（槽位，候选值），预测结果为是/否。适用于封闭词汇表的槽位。这是2020年之前的标准方法。
- **生成法。** 根据对话内容，以自由文本形式生成槽位值。适用于开放词汇表的槽位。这是目前的默认方法。

**评估指标。** 联合目标准确率（Joint Goal Accuracy, JGA）——即所有轮次中*每个*槽位均正确的比例。属于全有或全无的评估方式。在2026年的MultiWOZ 2.4竞赛中，顶尖模型的该指标得分约为83%。

**架构类型。**

1. **基于规则的方法（槽位正则表达式 + 关键词）。** 非常适合领域范围较窄的场景，且易于调试。
2. **TripPy / BERT-DST。** 采用BERT编码进行基于副本的生成。属于大语言模型出现之前的标准方法。
3. **LDST（LLaMA + LoRA）。** 经过指令微调的大语言模型，通过领域与槽位的提示引导进行生成。在MultiWOZ 2.4竞赛中能达到与ChatGPT相当的质量。
4. **无本体论的方法（2024–26年）。** 跳过领域模式，直接生成槽位名称和值。适用于开放领域的任务。
5. **提示语 + 结构化输出方法（2024–26年）。** 结合Pydantic模式的大语言模型，并采用受限解码方式。仅需5行代码即可实现，可直接用于生产环境。

### 典型的故障模式

- **轮次间的共指处理。** “我们就选第一个选项吧。” 需要明确具体指的是哪个选项。
- **覆盖操作与追加操作的区别。** 用户说“添加意大利菜”。是替换现有的菜品类别还是进行追加？
- **隐含的确认行为。** “好的，不错”——这是否表示接受了所提出的预订？
- **内容更正。** “实际上改为晚上7点。” 需要在不清除其他时段安排的情况下更新时间。
- **对系统先前发言的指代。** “是的，就是那个。” 这里的“那个”指的是哪一个？

## 构建它

### 步骤 1：基于规则的槽提取器

参见 `code/main.py`。正则表达式与同义词词典可覆盖特定领域内70%的典型语句：

```python
CUISINE_SYNONYMS = {
    "italian": ["italian", "pasta", "pizza", "italy"],
    "chinese": ["chinese", "chow mein", "noodles"],
}


def extract_cuisine(utterance):
    for canonical, synonyms in CUISINE_SYNONYMS.items():
        if any(syn in utterance.lower() for syn in synonyms):
            return canonical
    return None
```

在标准词汇表之外表现为易碎性。适用于确定性槽位确认场景。

### 步骤 2：状态更新循环

```python
def update_state(state, utterance):
    new_state = dict(state)
    for slot, extractor in SLOT_EXTRACTORS.items():
        value = extractor(utterance)
        if value is not None:
            new_state[slot] = value
    for slot in NEGATION_CLEARS:
        if is_negated(utterance, slot):
            new_state[slot] = None
    return new_state
```

三个不变量：

- 永不重置用户未触动的槽位。
- 明确的否定表达（“先不管菜品了”）必须清除该槽位内容。
- 用户的更正表达（“实际上……”）必须覆盖原有内容，而非追加。

### 步骤 3：基于大语言模型的离散时间序列预测与结构化输出生成

```python
from pydantic import BaseModel
from typing import Literal, Optional
import instructor

class RestaurantState(BaseModel):
    cuisine: Optional[Literal["italian", "chinese", "indian", "thai", "any"]] = None
    area: Optional[Literal["north", "south", "east", "west", "center"]] = None
    price: Optional[Literal["cheap", "moderate", "expensive"]] = None
    people: Optional[int] = None
    day: Optional[str] = None


def llm_dst(history, llm):
    prompt = f"""You track the slot values of a restaurant booking across turns.
Dialogue so far:
{render(history)}

Update the state based on the latest user turn. Output only the JSON state."""
    return llm(prompt, response_model=RestaurantState)
```

讲师结合 Pydantic 能够确保生成有效的状态对象。无需正则表达式，不存在模式不匹配问题，也不会出现虚构的字段。

### 步骤 4：JGA 评估

```python
def joint_goal_accuracy(predicted_states, gold_states):
    correct = sum(1 for p, g in zip(predicted_states, gold_states) if p == g)
    return correct / len(predicted_states)
```

校准：系统在所有槽位上的填充正确率是多少？对于 MultiWOZ 2.4 中排名前位的 2026 个系统，该准确率为 80-83%。您的领域专用系统应在较窄的词汇量下超越这一数值，或者其底层大语言模型的性能优于现有系统。

### 步骤 5：处理修正

```python
CORRECTION_CUES = {"actually", "no wait", "on second thought", "change that to"}


def is_correction(utterance):
    return any(cue in utterance.lower() for cue in CORRECTION_CUES)
```

在检测到修正时，应覆盖最新更新的槽位而非追加内容。若没有大语言模型的辅助，很难做到准确处理。现代的处理模式是：始终让大语言模型根据历史记录重新生成整个状态，而非进行增量更新——这样就能自然地处理各种修正。

## 常见陷阱

- **完整历史重生成成本。** 每次轮次都让大语言模型重新生成状态会导致总计 O(n²) 的令牌消耗。应限制历史记录长度或对旧轮次的对话进行摘要处理。
- **模式漂移问题。** 事后添加新的字段会破坏原有的训练数据。需要对模式版本进行管理。
- **大小写敏感性。** “Italian”、“italian”与“ITALIAN”等不同形式需要统一规范。
- **隐式继承机制。** 如果用户之前已指定“供4人使用”，则针对不同时间的新一轮请求不应清除该人数设置。必须始终传递完整的历史记录。
- **自由格式与固定集合的区别。** 名字、时间和地址属于需要自由格式输入的字段；而菜系和区域则属于固定集合。在模式设计中应同时包含这两类字段。

## 使用它

2026年技术栈：

| 场景 | 解决方案 |
|-----------|----------|
| 领域较窄（一两个意图） | 基于规则 + 正则表达式 |
| 领域广泛且有标注数据 | LDST（基于MultiWOZ风格数据的LLaMA + LoRA模型） |
| 领域广泛且无标注数据，需立即投入生产 | LLM + 指导系统 + Pydantic架构 |
| 语音输入场景 | ASR + 数据规范化工具 + LLM-DST |
| 多领域预订流程 | 基于架构引导的LLM，搭配各领域的Pydantic模型 |
| 对合规性要求极高的场景 | 以基于规则的方法为主，LLM作为备用方案，并设置确认流程 |

## 发布它

保存为 `outputs/skill-dst-designer.md`：

```markdown
---
name: dst-designer
description: Design a dialogue state tracker — schema, extractor, update policy, evaluation.
version: 1.0.0
phase: 5
lesson: 29
tags: [nlp, dialogue, task-oriented]
---

Given a use case (domain, languages, vocab openness, compliance needs), output:

1. Schema. Domain list, slots per domain, open vs closed vocabulary per slot.
2. Extractor. Rule-based / seq2seq / LLM-with-Pydantic. Reason.
3. Update policy. Regenerate-whole-state / incremental; correction handling; negation handling.
4. Evaluation. Joint Goal Accuracy on a held-out dialogue set, slot-level precision/recall, confusion on the hardest slot.
5. Confirmation flow. When to explicitly ask the user to confirm (destructive actions, low-confidence extractions).

Refuse LLM-only DST for compliance-sensitive slots without a rule-based secondary check. Refuse any DST that cannot roll back a slot on user correction. Flag schemas without version tags.
```

## 练习题

1. **简单级。** 在 `code/main.py` 中为 3 个字段（菜系、区域、价格）构建基于规则的状态跟踪器。使用 10 组手工编写的对话进行测试，并计算 JGA 值。
2. **中等级。** 使用相同的数据集，结合 Instructor 框架、Pydantic 类以及一个小型大语言模型。对比两者的 JGA 值，并分析最棘手的对话环节。
3. **高级别。** 同时实现基于规则的处理方式与大型语言模型的备用方案：当基于规则的模型以较低置信度仅输出少于 2 个字段时，自动切换到大语言模型处理。需同时衡量综合 JGA 值以及每轮对话的推理成本。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| DST | 对话状态跟踪 | 在对话轮次之间维护槽值字典。 |
| Slot | 用户意图的单元 | 后端所需的命名参数（如菜系、日期）。 |
| Domain | 任务领域 | 餐厅、酒店、出租车等——即一组槽的集合。 |
| JGA | 联合目标准确率 | 所有槽值均正确的轮次比例，采用全有或全无的判定方式。 |
| MultiWOZ | 测试基准 | 多领域WOZ数据集；用于标准DST性能评估。 |
| Ontology-free DST | 无本体结构 | 直接生成槽名和值，没有固定的列表。 |
| Correction | “实际上...” | 覆盖之前已填充槽值的轮次。 |

## 延伸阅读

- [Budzianowski 等人 (2018). MultiWOZ — 一种大规模多领域“奥兹巫师”评测基准](https://arxiv.org/abs/1810.00278) — 标准参考基准。
- [Feng 等人 (2023). 面向大语言模型驱动的对话状态跟踪（LDST）](https://arxiv.org/abs/2310.14970) — 利用 LLaMA 和 LoRA 进行指令微调以实现对话状态跟踪。
- [Heck 等人 (2020). TripPy — 一种用于价值无关型神经网络对话状态跟踪的三副本策略](https://arxiv.org/abs/2005.02877) — 基于副本技术的常用对话状态跟踪方法。
- [King, Flanigan (2024). 利用大语言模型实现无监督的端到端任务导向对话系统](https://arxiv.org/abs/2404.10753) — 基于 EM 算法的无监督任务导向对话方法。
- [MultiWOZ 排名榜](https://github.com/budzianowski/multiwoz) — 标准的对话状态跟踪结果展示平台。
