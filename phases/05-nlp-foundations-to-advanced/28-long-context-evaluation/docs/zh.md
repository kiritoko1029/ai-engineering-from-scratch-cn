# 长上下文评估——NIAH、RULER、LongBench、MRCR

> Gemini 3 Pro 宣称其上下文容量为 1000 万个令牌。在上下文量为 100 万个令牌时，8 针 MRCR 指标降至 26.3%。宣传数值与实际可用性能并不一定一致。通过长上下文评估才能了解您部署的模型真实容量。

**类型：** 学习
**语言：** Python
**先修课程：** 第 5 阶段 · 13（问答系统）、第 5 阶段 · 23（分块策略）
**时长：** 约 60 分钟

## 问题所在

你有一份共200页的合同。该模型宣称其上下文容量为100万个标记。你将合同内容输入并询问：“终止条款是什么？”模型虽然给出了回答，但实际引用的是封面页的内容，因为终止条款位于约12万标记深处，超出了模型实际能够处理的范围。

这就是2026年存在的上下文容量差距。产品规格表上标注的数值为100万或1000万，但实际上只有其中的60%到70%是可用的，而且“可用性”还取决于具体任务类型。

- **检索任务（大海捞针）：** 在前沿模型中，性能几乎能达到宣传的最大值。
- **多跳/聚合任务：** 在大多数模型中，一旦超过约128万标记，性能就会急剧下降。
- **基于分散事实的推理任务：** 是最先出现性能问题的任务类型。

长上下文评估就是通过这些维度来衡量模型的表现。本课程将介绍相关的基准测试、它们实际测量的内容，以及如何针对你的特定领域构建自定义的测试方法。

## 概念概述

![NIAH基准测试、RULER多任务测试、LongBench综合测试](../assets/long-context-eval.svg)

**Needle-in-a-Haystack（NIAH，2023年）。** 在长上下文中将一个事实（“魔法词是菠萝”）放置在特定深度处，要求模型检索出该事实。通过遍历不同的深度与长度组合进行测试。这是最初的長上下文基准测试。目前的顶尖模型已能在该测试中达到饱和状态；它是一个必要但非充分的基准。

**RULER（Nvidia，2024年）。** 包含4个类别下的13种任务类型：检索任务（单键/多键/多值）、多跳追踪任务（变量跟踪）、聚合任务（常见词频率统计）以及问答任务。上下文长度可配置，范围从4k到128k以上。该测试能够识别出那些在NIAH测试中表现优异但在多跳追踪任务上表现不佳的模型。在2024年的版本中，在声称能处理32k以上上下文的17个模型中，仅有半数能在32k长度下保持良好性能。

**LongBench v2（2024年）。** 包含503道选择题，上下文长度为8k至2M字，涵盖六类任务：单文档问答、多文档问答、长上下文学习、长对话、代码仓库处理以及长结构化数据处理。它是衡量模型实际长上下文处理能力的生产级基准测试。

**MRCR（多轮共指消解）。** 用于大规模的多轮共指关系识别。提供8针、24针和100针三种版本。该测试可揭示模型在注意力机制性能下降之前能够同时处理多少事实。

**NoLiMa。** “非词汇型针状目标”。针状目标与查询语句之间没有字面重叠，因此检索需要经过一步语义推理。其难度高于NIAH测试。

**HELMET。** 将多个文档拼接在一起，然后从其中任意一个文档中提出问题，以此测试模型的选择性注意力能力。

**BABILong。** 在无关的上下文“干草堆”中嵌入bAbI推理链，旨在测试模型在干草堆中的推理能力，而不仅仅是检索能力。

### 实际应报告的内容

- **宣称的上下文窗口长度。** 规格说明书中的数值。
- **实际检索长度。** 在某一阈值（例如 90%）下通过 NIAH 测试的长度。
- **实际推理长度。** 在该阈值下进行多跳推理或聚合操作时的长度。
- **性能下降曲线。** 按任务类型绘制的准确率与上下文长度之间的关系图。

规格说明书需列出两个数值：实际检索长度和实际推理长度。通常，实际推理长度仅为宣称窗口长度的 25% 至 50%。

## 构建它

### 步骤 1：为你的领域创建自定义的 NIAH

参见 `code/main.py`。其框架结构如下：

```python
def build_haystack(filler_text, needle, depth_ratio, total_tokens):
    if not (0.0 <= depth_ratio <= 1.0):
        raise ValueError(f"depth_ratio must be in [0, 1], got {depth_ratio}")
    if total_tokens <= 0:
        raise ValueError(f"total_tokens must be positive, got {total_tokens}")

    filler_tokens = tokenize(filler_text)
    needle_tokens = tokenize(needle)
    if not filler_tokens:
        raise ValueError("filler_text produced no tokens")

    # Repeat filler until long enough to fill the haystack body.
    body_len = max(total_tokens - len(needle_tokens), 0)
    while len(filler_tokens) < body_len:
        filler_tokens = filler_tokens + filler_tokens
    filler_tokens = filler_tokens[:body_len]

    insert_at = min(int(body_len * depth_ratio), body_len)
    haystack = filler_tokens[:insert_at] + needle_tokens + filler_tokens[insert_at:]
    return " ".join(haystack)


def score_niah(model, haystack, question, expected):
    answer = model.complete(f"Context: {haystack}\nQ: {question}\nA:", max_tokens=50)
    return 1 if expected.lower() in answer.lower() else 0
```

遍历 `depth_ratio` ∈ {0, 0.25, 0.5, 0.75, 1.0} 与 `total_tokens` ∈ {1k, 4k, 16k, 64k} 的所有组合，绘制热力图。该热力图即为目标模型的 NIAH 卡。

### 步骤 2：多针型变体

```python
def build_multi_needle(filler, needles, total_tokens):
    depths = [0.1, 0.4, 0.7]
    chunks = [filler[:int(total_tokens * 0.1)]]
    for depth, needle in zip(depths, needles):
        chunks.append(needle)
        next_chunk = filler[int(total_tokens * depth): int(total_tokens * (depth + 0.3))]
        chunks.append(next_chunk)
    return " ".join(chunks)
```

诸如“三个魔法词汇是什么？”之类的问题需要检索出全部三个。单针操作的成功并不能预示多针操作的成败。

### 步骤 3：多跳变量追踪（RULER 风格）

```python
haystack = """X1 = 42. ... (filler) ... X2 = X1 + 10. ... (filler) ... X3 = X2 * 2."""
question = "What is X3?"
```

该问题的解答需要依次执行三次赋值操作。参数量为128k的前沿模型在此处的准确率通常会降至50%-70%。

### 步骤 4：在您的技术栈上运行 LongBench v2

```python
from datasets import load_dataset
longbench = load_dataset("THUDM/LongBench-v2")

def eval_model_on_longbench(model, subset="single-doc-qa"):
    tasks = [x for x in longbench["test"] if x["task"] == subset]
    correct = 0
    for x in tasks:
        answer = model.complete(x["context"] + "\n\nQ: " + x["question"], max_tokens=20)
        if normalize(answer) == normalize(x["answer"]):
            correct += 1
    return correct / len(tasks)
```

按类别报告准确率。聚合后的分数会掩盖任务层面的显著差异。

## 常见陷阱

- **仅针对 NIAH 的评估。** 在 1 百万个标记的测试中通过 NIAH 并不能说明模型在多跳推理方面的能力。务必运行 RULER 或自定义的多跳测试。
- **统一的深度采样策略。** 许多实现仅测试 depth=0.5 的情况。应同时测试 depth=0、0.25、0.5、0.75、1.0 —— “中间丢失”现象确实存在。
- **与填充词的字典重叠问题。** 如果检索针与填充词共享关键词，检索工作将变得极其简单。建议使用类似 NoLiMa 的非重叠检索针。
- **忽视延迟因素。** 1 百万个标记的提示语需要 30 到 120 秒的时间进行预填充处理。在评估准确率的同时，还需测量生成第一个标记所需的时间。
- **厂商自行报告的数值。** OpenAI、Google、Anthropic 均会发布各自的评分数据。务必针对自身的应用场景重新进行独立测试。

## 使用它

2026年技术栈：

| 场景 | 基准测试 |
|-----------|-----------|
| 快速合理性检查 | 长度为3、深度为3的自定义NIAH模型 |
| 生产环境模型选择 | 目标长度下的RULER模型（13项任务） |
| 真实场景质量评估 | LongBench v2中的单文档QA子集 |
| 多跳推理能力 | BABILong模型或自定义的变量追踪机制 |
| 对话系统性能 | 目标长度下的MRCR 8-needle模型 |
| 模型升级后的回归检测 | 使用内部固定的NIAH模型及RULER测试框架，对每个新模型进行测试 |

生产环境通用准则：在确保拥有目标长度下的NIAH模型以及1项推理任务之前，切勿依赖上下文窗口。

## 发布它

保存为 `outputs/skill-long-context-eval.md`：

```markdown
---
name: long-context-eval
description: Design a long-context evaluation battery for a given model and use case.
version: 1.0.0
phase: 5
lesson: 28
tags: [nlp, long-context, evaluation]
---

Given a target model, target context length, and use case, output:

1. Tests. NIAH depth × length grid; RULER multi-hop; custom domain task.
2. Sampling. Depths 0, 0.25, 0.5, 0.75, 1.0 at each length.
3. Metrics. Retrieval pass rate; reasoning pass rate; time-to-first-token; cost-per-query.
4. Cutoff. Effective retrieval length (90% pass) and effective reasoning length (70% pass). Report both.
5. Regression. Fixed harness, rerun on every model upgrade, surface deltas.

Refuse to trust a context window from the model card alone. Refuse NIAH-only evaluation for any multi-hop workload. Refuse vendor self-reported long-context scores as independent evidence.
```

## 练习题

1. **简单级。** 构建一个具有 3 种深度（0.25、0.5、0.75）和 3 种长度（1k、4k、16k）的 NIAH。可在任意模型上运行，并将通过率以 3×3 热图的形式展示。
2. **中等级。** 增加一种三针型变体。在每种长度下测量所有 3 种针型的检索效果，并与相同长度下的单针型通过率进行对比。
3. **高级别。** 在 64k 的填充数据中嵌入一个包含 3 次跳转的变量追踪任务（X1 → X2 → X3）。在 3 种前沿模型上测量准确率，并报告每种模型的有效推理长度。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| NIAH | 海量信息中的针 | 在冗余文本中植入一个事实，要求模型检索出该事实。 |
| RULER | 强化版的NIAH | 包含检索、多跳推理、聚合与问答共13种任务类型。 |
| 有效上下文长度 | 模型的真实处理能力 | 准确率仍能保持在阈值以上的文本长度。 |
| 中间内容被忽略 | 深度偏差 | 模型对长输入中间的内容关注不足。 |
| 多针检索 | 同时处理多个事实 | 提出多个查询；测试的是模型的注意力调度能力，而不仅仅是检索能力。 |
| MRCR | 多轮共指识别 | 8针、24针或100针的共指任务；可揭示注意力的饱和现象。 |
| NoLiMa | 非词汇级针查询 | 查询与事实之间没有共享的实词；需要依赖推理能力。 |

## 延伸阅读

- [Kamradt (2023). Needle in a Haystack 分析](https://github.com/gkamradt/LLMTest_NeedleInAHaystack) —— 原始的 NIAH 代码库。
- [Hsieh 等人 (2024). RULER：你的长上下文大语言模型的真实上下文长度是多少？](https://arxiv.org/abs/2404.06654) —— 多任务基准测试。
- [Bai 等人 (2024). LongBench v2](https://arxiv.org/abs/2412.15204) —— 真实场景下的长上下文评估工具。
- [Modarressi 等人 (2024). NoLiMa：非词汇类“针状测试样本”](https://arxiv.org/abs/2404.06666) —— 更具挑战性的测试样本。
- [Kuratov 等人 (2024). BABILong](https://arxiv.org/abs/2406.10149) —— 针对堆叠数据中的推理能力测试。
- [Liu 等人 (2024). 迷失在中间：语言模型如何利用长上下文](https://arxiv.org/abs/2307.03172) —— 关于深度偏见的论文。
