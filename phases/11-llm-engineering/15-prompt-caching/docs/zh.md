# 提示词缓存与上下文缓存

> 你的系统提示词是 4,000 个 token。你的 RAG 上下文是 20,000 个 token。你在每次请求时都把两者一起发出去，而且每一次都要为它们付费。提示词缓存让供应商在他们那一侧把这段前缀保持「热」状态，并在复用时只按正常费率的 10% 向你计费。用得好，它能把推理成本削减 50–90%，把首个 token 的延迟降低 40–85%。

**类型：** 构建
**语言：** Python
**前置要求：** Phase 11 · 01（Prompt Engineering）、Phase 11 · 05（Context Engineering）、Phase 11 · 11（Caching and Cost）
**所需时间：** 约 60 分钟

## 问题所在

一个编码 agent 在一段对话的每一轮都向 Claude 发送同样的 15,000 个 token 的系统提示词。二十轮，按每百万输入 token 收费 3 美元算，光是输入成本就有 0.90 美元——这还没算上用户实际发的任何消息。乘以每天 10,000 段对话，账单就会冲到每天 9,000 美元，而这些都是为永不变化的文本付的钱。

你没法在不损害质量的前提下缩短提示词。你也没法不发送它——模型每一轮都需要它。唯一可行的办法，就是别再为一段供应商早已见过的前缀付全价。

这个办法就是提示词缓存。Anthropic 在 2024 年 8 月推出了它（并在 2025 年加入了 1 小时的延长 TTL 变体），OpenAI 在当年晚些时候将其自动化，Google 则随 Gemini 1.5 一同推出了显式的上下文缓存，如今这三家都把它作为各自前沿模型上的一项一等公民特性提供。

## 概念说明

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**运作机制。** 当一次请求的前缀与近期某次请求相匹配时，供应商会直接复用上一次运行的 KV-cache，而不是重新对这些 token 进行编码。你第一次要付一笔小额的写入溢价，此后每一次都能享受大幅的读取折扣。

**2026 年的三种供应商风味。**

| Provider | API style | Hit discount | Write premium | Default TTL | Min cacheable |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在内容块上显式标记 `cache_control` | 输入费用 90% 折扣 | 25% 附加费 | 5 分钟（可延长至 1 小时） | 1,024 token（Sonnet/Opus），2,048（Haiku） |
| OpenAI | 自动前缀检测 | 输入费用 50% 折扣 | 无 | 最长 1 小时（尽力而为） | 1,024 token |
| Google（Gemini） | 显式 `CachedContent` API | 按存储计费；读取约为正常费率的 25% | 按 token·小时收取存储费 | 用户设定（默认 1 小时） | 4,096 token（Flash），32,768（Pro） |

**不变法则。** 这三家都只缓存前缀。如果两次请求之间有任何一个 token 不同，那么从第一个不同的 token 之后的所有内容都会未命中。把*稳定*的部分放在顶部，把*可变*的部分放在底部。

### 缓存友好的布局

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

只要违背这个顺序——把用户消息放在系统提示词上面，或者在 few-shot 之间穿插动态检索内容——缓存就永远不会命中。

### 盈亏平衡的计算

Anthropic 的 25% 写入溢价意味着，一个缓存块至少要被读取两次才能净省钱。1 次写入 + 1 次读取，平均每次请求的成本是 0.675 倍（节省 32%）；1 次写入 + 10 次读取，平均是 0.205 倍（节省 80%）。经验法则：凡是你预期在 TTL 内至少复用 3 次的内容，都值得缓存。

## 开始构建

### 第 1 步：用显式标记实现 Anthropic 提示词缓存

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 标记告诉 Anthropic 把这个块存储 5 分钟。在这个窗口内复用会命中；过期后再复用则会再次写入。

**响应中的用量字段：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

在 CI 中同时检查这两个字段——如果 `cache_read_input_tokens` 在多次请求间始终保持为零，说明你的缓存键正在漂移。

### 第 2 步：1 小时延长 TTL

对于长时间运行的批处理作业，5 分钟的默认 TTL 会在作业之间过期。设置 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小时 TTL 的写入溢价是 2 倍（超出基准 50% 而非 25%），但只要某个批处理作业复用该前缀超过 5 次，它就能很快回本。

### 第 3 步：OpenAI 自动缓存

OpenAI 不给你任何可配置的东西。任何超过 1,024 token 且与近期请求匹配的前缀，都会自动获得 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同样的缓存友好布局规则在此适用。有两件事会摧毁 OpenAI 的缓存，却不会摧毁 Anthropic 的：更改 `user` 字段（它被用作缓存键的一个组成部分）以及重新排列工具。

### 第 4 步：Gemini 显式上下文缓存

Gemini 把缓存当作一个由你创建并命名的一等对象：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3-pro",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3-pro",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

只要缓存存在，Gemini 就会按 token·小时收取存储费，并以约 25% 的正常输入费率进行读取。当你需要在数天里跨多个会话复用同一段巨大的提示词时，这种形态正合适。

### 第 5 步：在生产环境中度量命中率

参见 `code/main.py`，那里有一个模拟三家供应商的「记账员」，它会跟踪写入/读取/未命中的计数，并计算每 1K 次请求的混合成本。把部署的放行条件设定在一个目标命中率上——大多数生产环境的 Anthropic 配置在预热后应该能看到 >80% 的读取占比。

## 在 2026 年仍会出现的陷阱

- **把动态时间戳放在顶部。** 在系统提示词顶部放 `"Current time: 2026-04-22 15:30:02"`。每一次请求都会未命中。把时间戳移到缓存断点下方。
- **重新排列工具。** 以稳定的顺序序列化工具——两次部署之间字典的一次重排会破坏每一次命中。
- **自由文本的近似重复。** "You are helpful." 对比 "You are a helpful assistant."——一个字节的差异 = 完全未命中。
- **块太小。** Anthropic 强制要求 1,024 token 的下限（Haiku 为 2,048）。更小的块会悄无声息地不被缓存。
- **盲目的成本仪表盘。** 把「输入 token」拆分为已缓存与未缓存。否则一次流量下降看起来会像是一次缓存的胜利。

## 实际运用

2026 年的缓存技术栈：

| Situation | Pick |
|-----------|------|
| 拥有稳定的 10k+ 系统提示词、多轮交互的 agent | Anthropic `cache_control`，5 分钟 TTL |
| 在 30 分钟以上时间内复用某个前缀的批处理作业 | Anthropic，配 `ttl: "1h"` |
| 运行在 GPT-5 上、无自定义基础设施的 serverless 端点 | OpenAI 自动缓存（只需让你的前缀稳定且足够长） |
| 跨多日复用一份巨大的代码/文档语料 | Gemini 显式 `CachedContent` |
| 跨供应商的回退方案 | 让可缓存的前缀布局在各供应商间保持完全一致，这样任何一处命中都能生效 |

与语义缓存（Phase 11 · 11）结合用于用户消息这一层：提示词缓存处理*token 完全一致*的复用，语义缓存处理*语义一致*的复用。

## 交付成果

保存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 练习

1. **简单。** 拿一段 10 轮的对话，配一个 5,000 token 的系统提示词，对 Claude 运行。先在不带 `cache_control` 的情况下运行一次，再带上它运行一次。报告每种情况下的输入 token 账单。
2. **中等。** 编写一个测试框架，给定一个提示词模板和一份请求日志，计算每家供应商（Anthropic 5m、Anthropic 1h、OpenAI 自动、Gemini 显式）的预期命中率和美元节省额。
3. **困难。** 构建一个布局优化器：给定一个提示词和一份标记了 `stable=True/False` 的字段列表，在不丢失信息的前提下重写提示词，把单个缓存断点放在最大限度缓存友好的位置上。在一个真实的 Anthropic 端点上验证。

## 关键术语

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Prompt caching | 「让长提示词变便宜」 | 为匹配的前缀复用供应商一侧的 KV-cache；对重复的输入 token 给予 50-90% 折扣。 |
| `cache_control` | 「Anthropic 的那个标记」 | 一个内容块属性，声明「到此为止的所有内容都可缓存」；`{"type": "ephemeral"}`。 |
| Cache write | 「付那笔溢价」 | 填充缓存的第一次请求；在 Anthropic 上按约 1.25 倍输入费率计费，在 OpenAI 上免费。 |
| Cache read | 「那个折扣」 | 后续匹配该前缀的请求；按 10%（Anthropic）、50%（OpenAI）、约 25%（Gemini）计费。 |
| TTL | 「它能活多久」 | 缓存保持「热」状态的秒数；Anthropic 默认 5m（可延长至 1h），OpenAI 尽力而为最长 1h，Gemini 由用户设定。 |
| Extended TTL | 「1 小时的 Anthropic 缓存」 | `{"type": "ephemeral", "ttl": "1h"}`；写入溢价为 2 倍，但对批处理复用而言值得。 |
| Prefix match | 「我的缓存为什么没命中」 | 只有当从开头到断点的每一个 token 都逐字节一致时，缓存才会命中。 |
| Context caching (Gemini) | 「显式的那种」 | Google 的具名、按存储计费的缓存对象；最适合对大型语料的多日复用。 |

## 延伸阅读

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) —— `cache_control`、1h TTL、盈亏平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) —— 自动前缀匹配。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) —— `CachedContent` API 与存储定价。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) —— 最初的发布文章，附带延迟数据。
- Phase 11 · 05（Context Engineering）—— 该在哪里切分提示词，缓存才能落地。
- Phase 11 · 11（Caching and Cost）—— 把提示词缓存与作用于用户消息的语义缓存配对使用。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) —— 提示词缓存向用户暴露的那套 KV-cache 内存模型；解释了为什么重读一段缓存的前缀比重新计算要便宜约 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) —— prefill 正是提示词缓存所抄的近路那个阶段；这篇论文解释了为什么缓存命中时 TTFT 会大幅下降而 TPOT 不受影响。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) —— 提示词缓存与 speculative decoding、Flash Attention 以及 MQA/GQA 并列，都是扭转推理成本曲线的杠杆；想了解另外三个就读这篇。
