# 聊天机器人——从基于规则到神经网络，再到大语言模型智能体

> ELIZA通过模式匹配进行回复，DialogFlow负责意图映射，GPT则依据权重生成答案，而Claude则会调用工具并进行验证。每个时代都在解决前一个时代的最大缺陷。

**类型：** 学习
**语言：** Python
**先修要求：** 第5阶段·13（问答系统）、第5阶段·14（信息检索）
**时长：** 约75分钟

## 问题所在

有用户表示：“我想更改我的航班。”系统需要判断用户的真实需求、识别缺失的信息、确定获取这些信息的方式以及如何完成操作。随后用户又说：“等等，我要是取消航班呢？”系统必须记住上下文、切换任务并保持状态不变。

对于机器学习系统而言，对话处理是一项艰巨的任务。输入内容往往是开放式的，而输出需要在多轮交流中始终保持连贯性。系统可能还需要对现实世界进行操作（如更改航班信息、扣款等）。任何错误的步骤都会被用户立即察觉。

聊天机器人架构已经经历了四种不同的范式，每一种的出现都是因为前一种在实践中暴露出了明显的缺陷。本课程将按顺序介绍这些范式。2026年的实际应用环境则是最后两种范式的混合体。

## 概念概述

![聊天机器人发展历程：基于规则 → 基于检索 → 神经网络 → 智能体](../assets/chatbot.svg)

**基于规则的模型（ELIZA、AIML、DialogFlow）。** 通过人工编写的模式匹配用户输入并生成响应。意图分类器会将请求路由到预定义的处理流程中。状态机负责收集所需的参数信息。这类模型在最初设计的狭窄应用场景下表现优异，但一旦超出范围就会立即失效。尽管如此，它们仍被应用于对准确性要求极高的领域（如银行身份验证、航空订票），因为这些场景无法容忍幻觉现象。

**基于检索的模型。** 具有类似常见问题解答系统的特性。需要将每一组“用户话语，对应响应”的组合进行编码。在运行时，会对用户的消息进行编码并查找最相似的存储响应。其原理类似于 Zendesk 中的经典“相关文章”功能。相比规则驱动的方式，它能更好地处理同义表达。由于不涉及内容生成，因此不会出现幻觉。

**基于神经网络的模型（seq2seq）。** 通过对话日志训练出的编码器-解码器结构，能够从零开始生成响应。虽然表达流畅，但容易产生泛化性过强的输出（如“我不知道”）以及事实偏差。这类模型始终无法稳定地紧扣主题，这也是谷歌、Facebook 和微软在 2016 年至 2019 年间推出的聊天机器人表现不佳的原因。

**LLM 智能体。** 这是一种被封装在循环结构中的语言模型，能够进行规划、调用工具并验证结果。它并非依靠长文本提示的聊天机器人，而是一个智能体循环：规划 → 调用工具 → 观察结果 → 决定下一步行动。通过“检索优先增强”（RAG）技术可防止其产生幻觉，而工具调用功能则使其能够真正执行任务。这就是 2026 年的架构标准。

这四种范式并非依次替代的关系。2026 年的实用型聊天机器人会同时运用这四种技术：利用基于规则的方法处理身份验证及具有破坏性的操作，用基于检索的方式应对常见问题，借助神经网络生成自然流畅的文本，而对于含义模糊的开放式问题则由 LLM 智能体来处理。

## 构建它

### 步骤 1：基于规则的模式匹配

```python
import re


class RulePattern:
    def __init__(self, pattern, response_template):
        self.regex = re.compile(pattern, re.IGNORECASE)
        self.template = response_template


PATTERNS = [
    RulePattern(r"my name is (\w+)", "Nice to meet you, {0}."),
    RulePattern(r"i (need|want) (.+)", "Why do you {0} {1}?"),
    RulePattern(r"i feel (.+)", "Why do you feel {0}?"),
    RulePattern(r"(.*)", "Tell me more about that."),
]


def rule_based_respond(user_input):
    for pattern in PATTERNS:
        m = pattern.regex.match(user_input.strip())
        if m:
            return pattern.template.format(*m.groups())
    return "I don't understand."
```

20行实现的ELIZA程序。这种“反射技巧”（将输入“我感到难过”转换为“你为什么感到难过”）是Weizenbaum在1966年提出的经典心理治疗师演示案例，至今仍具有教学意义。

### 步骤 2：基于检索的（常见问题解答）

此示例代码需要执行 `pip install sentence-transformers`（该命令会同时安装 torch）。而本课程中可运行的 `code/main.py` 文件则使用了标准库中的 Jaccard 相似度算法，因此无需任何外部依赖即可运行。

```python
from sentence_transformers import SentenceTransformer
import numpy as np


FAQ = [
    ("how do i reset my password", "Go to Settings > Security > Reset Password."),
    ("how do i cancel my order", "Go to Orders, find the order, click Cancel."),
    ("what is your return policy", "30-day returns on unused items, original packaging."),
]


encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
faq_questions = [q for q, _ in FAQ]
faq_embeddings = encoder.encode(faq_questions, normalize_embeddings=True)


def faq_respond(user_input, threshold=0.5):
    q_emb = encoder.encode([user_input], normalize_embeddings=True)[0]
    sims = faq_embeddings @ q_emb
    best = int(np.argmax(sims))
    if sims[best] < threshold:
        return None
    return FAQ[best][1]
```

基于阈值的拒绝机制是核心设计选择。如果最佳匹配结果不够理想，则返回 `None`，由系统负责升级处理。

### 步骤 3：神经网络生成（基线模型）

可使用小型指令微调的编码器-解码器模型（FLAN-T5）或经过精细调优的对话模型。在2026年，这类模型单独使用无法用于生产环境（存在矛盾、偏离主题以及事实错误），但可集成于混合系统中以实现自然的语句表达。类似DialoGPT的仅解码器模型需要明确的轮次分隔符和结束标记处理才能生成连贯的回复；而FLAN-T5的文本到文本处理流程则可直接用于教学示例中。

```python
from transformers import pipeline

chatbot = pipeline("text2text-generation", model="google/flan-t5-small")

response = chatbot("Respond politely to: Hi there!", max_new_tokens=40)
print(response[0]["generated_text"])
```

### 步骤 4：LLM 智能体循环

2026年的生产形态：

```python
def agent_loop(user_message, tools, llm, max_steps=5):
    history = [{"role": "user", "content": user_message}]
    for _ in range(max_steps):
        response = llm(history, tools=tools)
        tool_call = response.get("tool_call")
        if tool_call:
            tool_name = tool_call.get("name")
            args = tool_call.get("arguments")
            if not isinstance(tool_name, str) or tool_name not in tools:
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": str(tool_name), "content": f"error: unknown tool {tool_name!r}"})
                continue
            if not isinstance(args, dict):
                history.append({"role": "assistant", "tool_call": tool_call})
                history.append({"role": "tool", "name": tool_name, "content": f"error: arguments must be a dict, got {type(args).__name__}"})
                continue
            fn = tools[tool_name]
            result = fn(**args)
            history.append({"role": "assistant", "tool_call": tool_call})
            history.append({"role": "tool", "name": tool_name, "content": result})
        else:
            return response["content"]
    return "I could not complete the task in the step budget."
```

需要命名的三个要素。工具即LLM可以调用的函数。当LLM返回最终答案而非工具调用时，循环即终止。步骤预算机制可防止在任务含义模糊时出现无限循环。

实际生产环境还会增加以下功能：检索优先的上下文获取（在每次LLM调用前注入相关文档）、行为约束（未经确认禁止执行破坏性操作）、可观测性（记录每一步骤）以及评估机制（自动检查智能体行为是否符合规范）。

### 第 5 步：混合路由

```python
def hybrid_chat(user_input):
    if is_destructive_action(user_input):
        return structured_flow(user_input)

    faq_answer = faq_respond(user_input, threshold=0.6)
    if faq_answer:
        return faq_answer

    return agent_loop(user_input, tools, llm)


def is_destructive_action(text):
    danger_words = ["delete", "cancel", "charge", "refund", "transfer"]
    return any(w in text.lower() for w in danger_words)
```

模式如下：对于任何具有破坏性的操作，采用确定性规则处理；针对标准常见问题，则使用预置的FAQ进行检索；其余所有场景则由大语言模型智能体负责处理。这便是2026年推出的客户支持系统所采用的架构。

## 使用它

2026年技术栈：

| 使用场景 | 架构设计 |
|---------|---------------|
| 预订、支付、身份验证 | 基于规则的状态机 + 时隙填充算法 |
| 客户支持常见问题解答 | 从精选答案库中检索回复 |
| 开放式帮助聊天 | 结合RAG技术与工具调用的LLM智能体 |
| 内部工具 / IDE助手 | 具备搜索、读取、写入功能的LLM智能体 |
| 陪伴型/角色化聊天机器人 | 经过微调的LLM，配备角色系统提示词及知识检索功能 |

在生产环境中始终应采用混合路由策略。没有一种单一架构能够完美处理所有请求。路由层通常为一个小型意图分类器。

## 仍会发布的故障模式

- **错误的任务执行。** 大语言模型智能体声称已完成其实际并未执行的操作。缓解措施：验证任务结果，记录工具调用日志，绝不允许大语言模型在工具未返回成功结果的情况下宣称已完成任务。
- **提示注入攻击。** 用户插入文本以覆盖系统提示词。该攻击类型位列 2025 年 OWASP 大语言模型应用十大风险榜单中的 LLM01 名位。其实现方式有两种：直接注入（将恶意文本粘贴到聊天界面中）和间接注入（将恶意内容隐藏在文档、电子邮件或智能体读取的工具输出中）。

不同场景下的攻击率各不相同。在通用工具使用及编程基准测试中，前沿大语言模型的成功攻击率通常在 0.5% 至 8.5% 之间。而在特定高风险环境中（如针对人工智能编程智能体的自适应攻击、存在漏洞的编排系统），该攻击率可高达约 84%。实际生产环境中的相关 CVE 包括 EchoLeak（CVE-2025-32711，CVSS 评分 9.3）——这是一种由攻击者控制的电子邮件触发的零点击数据泄露缺陷，存在于 Microsoft 365 Copilot 中。

缓解措施包括：在整个处理循环中将用户输入视为不可信内容；在调用工具前对输入进行净化处理；将工具输出与主提示词隔离开来；采用“计划-验证-执行”（PVE）模式，即让智能体先制定计划，随后在执行每项操作前根据该计划进行验证（以此防止工具返回的结果引入新的、未计划的操作）；对具有破坏性的操作要求用户确认；为工具赋予最小权限。

仅靠提示工程无法完全消除此风险，还需部署外部运行时防御层，如 LLM Guard、白名单验证以及语义异常检测功能。
- **任务范围扩大。** 由于某次工具调用返回了与之关联度较低的信息，智能体可能会偏离原定任务。缓解措施：明确限定工具的功能契约；保持系统提示词的重点清晰；增加对任务偏离程度的评估指标。
- **无限循环问题。** 智能体不断重复调用同一个工具。缓解措施：设置步骤预算限制、对工具调用进行去重处理，并由大语言模型自身判断“当前是否正在取得进展”。
- **上下文窗口耗尽。** 长时间的对话会导致最早的对话内容被挤出上下文范围。缓解措施：对早期对话内容进行总结，通过相似度检索相关历史对话内容，或使用支持长上下文的模型。

## 发布它

保存为 `outputs/skill-chatbot-architect.md`：

```markdown
---
name: chatbot-architect
description: Design a chatbot stack for a given use case.
version: 1.0.0
phase: 5
lesson: 17
tags: [nlp, agents, chatbot]
---

Given a product context (user need, compliance constraints, available tools, data volume), output:

1. Architecture. Rule-based, retrieval, neural, LLM agent, or hybrid (specify which paths go where).
2. LLM choice if applicable. Name the model family (Claude, GPT-4, Llama-3.1, Mixtral). Match to tool-use quality and cost.
3. Grounding strategy. RAG sources, retrieval method (see lesson 14), tool contracts.
4. Evaluation plan. Task success rate, tool-call correctness, off-task rate, hallucination rate on held-out dialogs.

Refuse to recommend a pure-LLM agent for any destructive action (payments, account deletion, data modification) without a structured confirmation flow. Refuse to skip the prompt-injection audit if the agent has write access to anything.
```

## 练习题

1. **简单级。** 为咖啡店点单机器人实现基于规则的响应机制，设定10种不同的模式。需测试边界情况：重复下单、订单修改、取消订单以及意图不明确的情况。
2. **中等级。** 构建一个“固定FAQ + 大语言模型兜底”相结合的系统。为一个SaaS产品准备50条预设的FAQ内容，并设置通过文档库检索来调用大语言模型作为兜底方案。需针对100个真实的客户支持问题，统计系统的拒绝率和响应准确率。
3. **高级别。** 使用搜索、读取用户数据以及发送邮件这三种工具来实现上述智能体循环逻辑。通过包含提示注入攻击在内的50种测试场景进行评估，并报告离题率、任务失败率以及任何提示注入成功的案例数量。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|-----------------|----------|
| Intent | 用户的需求 | 分类标签（如 book_flight、reset_password）。会被路由至对应的处理函数。 |
| Slot | 信息片段 | 机器人所需的参数（如日期、目的地）。填充槽位的过程即是一系列提问的流程。 |
| RAG | 检索增强生成 | 先检索相关文档，再以此为依据生成大语言模型的回复。 |
| Tool call | 函数调用 | 大语言模型会发出包含名称及参数的结构化调用请求，运行时系统执行该请求并返回结果。 |
| Agent loop | 计划、执行、验证 | 控制器负责交替执行大语言模型调用与工具调用，直至任务完成。 |
| Prompt injection | 提示注入攻击 | 试图篡改系统提示词的恶意输入。 |

## 延伸阅读

- [Weizenbaum (1966). ELIZA — 用于研究自然语言交流的计算机程序](https://web.stanford.edu/class/cs124/p36-weizenabaum.pdf) — 最早关于基于规则的聊天机器人的论文。
- [Thoppilan 等人 (2022). LaMDA：用于对话应用的语言模型](https://arxiv.org/abs/2201.08239) — Google 在大语言模型智能体兴起之前的神经网络聊天机器人相关论文。
- [Yao 等人 (2022). ReAct：在语言模型中实现推理与行动的协同](https://arxiv.org/abs/2210.03629) — 提出智能体循环模式的论文。
- [Anthropic 关于构建高效智能体的指南](https://www.anthropic.com/research/building-effective-agents) — 2024 年发布的生产环境指导，其在 2026 年依然具有参考价值。
- [Greshake 等人 (2023). 并非您预期的那样：利用间接提示注入破坏现实世界中的大语言模型集成应用](https://arxiv.org/abs/2302.12173) — 关于提示注入的论文。
- [2025 年 OWASP 大语言模型应用十大风险 — LLM01 提示注入](https://genai.owasp.org/llmrisk/llm01-prompt-injection/) — 将提示注入列为首要安全风险的排名列表。
- [AWS — 保护 Amazon Bedrock 智能体免受间接提示注入攻击](https://aws.amazon.com/blogs/machine-learning/securing-amazon-bedrock-agents-a-guide-to-safeguarding-against-indirect-prompt-injections/) — 包含“计划-验证-执行”机制及用户确认流程等实用的编排层防御方案。
- [EchoLeak (CVE-2025-32711)](https://www.vectra.ai/topics/prompt-injection) — 由间接提示注入引发的典型零点击数据泄露 CVE，是说明具备写权限的智能体为何需要运行时防护的参考案例。
