# 大语言模型评估——RAGAS、DeepEval、G-Eval

> 精确匹配与 F1 分数在语义等价性方面存在差异。人工审核无法实现规模化处理。以大语言模型作为评判标准是实际应用中的解决方案——只要经过充分的校准，其输出数值即可被信赖。

**类型：** 构建
**编程语言：** Python
**先修课程：** 第 5 阶段 · 13（问答系统）、第 5 阶段 · 14（信息检索）
**耗时：** 约 75 分钟

## 问题所在

您的 RAG 系统给出的回答为：“2007年6月29日。”
而黄金参考答案为：“June 29, 2007.”
精确匹配得分仅为 0，F1 分数约为 75%，而人类评分则为 100%。

现在将此数值乘以 10,000 个测试用例，再乘上检索器、分块方式、提示词或模型发生的每一次变更。因此，我们需要一种能够理解语义、可在大规模场景下低成本运行、不会虚报性能退化情况，并能准确揭示故障模式的评估工具。

2026 年已有三种框架致力于解决这一问题。

- **RAGAS**：检索增强生成评估工具。包含四项 RAG 指标（准确性、答案相关性、上下文精确度、上下文召回率），并配有自然语言理解模块与大型语言模型评判后端。该工具基于研究结果，且体积轻量。
- **DeepEval**：专为大型语言模型设计的 Pytest 工具。提供 G-Eval 指标以及任务完成度、幻觉现象、偏见检测等评估指标，可直接集成到 CI/CD 流水线中。
- **G-Eval**：既是一种评估方法，也是 DeepEval 的一项指标。它采用大型语言模型作为评判者，结合思维链机制与自定义标准，输出 0-1 分数的评估结果。

这三种框架均依赖于“大型语言模型作为评判者”的理念。本课程旨在帮助学习者建立对该方法的直观理解，并了解其背后的信任构建机制。

## 概念概述

![四个评估维度，LLM作为评判器的架构](../assets/llm-evaluation.svg)

**LLM作为评判器。** 用能够根据评分标准对输出结果进行打分的LLM替代静态指标。给定`(query, context, answer)`后，向评判型LLM发起如下提示：“请对答案的忠实度给出0-1之间的分数。”该LLM将返回对应的分数。

其有效原因在于：LLM能以极低的成本模拟人类的判断能力。GPT-4o-mini每处理一个样本的成本约为0.003美元，因此花费不到5美元即可进行1000个样本的回归评估。

但其潜在缺陷往往表现为无声无息：

1. **评判器偏差。** 评判器更倾向于接受较长的答案、出自自身模型系列的答案，以及风格与提示文相符的答案。
2. **JSON解析错误。** 错误的JSON格式会导致分数为NaN，进而被悄悄排除在整体统计之外。RAGAS的用户对此痛点十分熟悉，应通过try/except机制并明确处理失败情况来加以规避。
3. **模型版本更迭带来的漂移。** 升级评判器模型会改变所有评估指标的数值。因此应冻结评判器模型及其版本。

**RAG四项核心指标。**

| 指标 | 评估问题 | 后端实现方式 |
|------|----------|--------------|
| 忠实度 | 答案中的每一条陈述是否均源自检索到的上下文？ | 基于NLI的蕴含关系判断 |
| 答案相关性 | 答案是否针对提问内容？ | 从答案中生成假设性问题，再与真实问题进行比对 |
| 上下文精确度 | 在所有检索到的片段中，有多少比例是相关的？ | 通过LLM作为评判器进行判定 |
| 上下文召回率 | 检索结果是否包含了所有必要的信息？ | 由LLM作为评判器，与标准答案进行对比 |

**G-Eval。** 可定义自定义评估标准，例如“答案是否引用了正确的来源？”该框架会自动扩展为思维链式评估步骤，随后给出0-1之间的分数。这对于RAGAS尚未覆盖的特定领域质量维度非常适用。

**校准。** 在获得与人工标注结果的相关性数据之前，切勿直接信任评判器的原始分数。应先处理100个手工标注的样本，绘制评判器分数与人工分数的对比图，并计算斯皮尔曼相关系数。若该系数小于0.7，则说明当前的评判标准需要优化。

## 构建它

### 步骤 1：采用 RAGAS 风格的 NLI 保真度处理

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

将答案分解为独立的命题。针对每个命题，使用检索到的上下文进行自然语言推理验证。忠实度等于得到支持的命题比例。

### 步骤 2：回答相关性

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

如果回答所涉及的问题与提问者实际提出的问题不同，则相关性会降低。

### 步骤 3：定义 G-Eval 自定义指标

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

评估步骤即评分标准。明确的步骤比隐含的“0-1分”提示更为稳定。

### 步骤 4：CI 阶段审核

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

以 pytest 文件的形式提交。在每个 Pull Request 中运行该测试，若检测到回归问题则阻止合并。

### 第 5 步：从零实现玩具评估模型

参见 `code/main.py`。仅使用标准库实现的忠实度（答案内容与上下文的重叠程度）及相关性（答案词元与问题词元的重叠程度）估算方法。非生产环境可用，用于展示数据结构。

## 常见陷阱

- **无需校准。** 相关性仅为 0.3 的评估模型属于噪声，上架前必须进行校准测试。
- **自我评估问题。** 使用同一款大语言模型同时负责生成内容与评分会导致分数虚高 10-20%，应使用不同系列的大语言模型作为评估模型。
- **成对评估中的位置偏差。** 评估模型倾向于更偏好最先呈现的选项，务必随机排序并分别进行两次评估。
- **原始汇总数据会掩盖缺陷。** 平均分高达 0.85 时也可能存在 5% 的严重错误，必须检查最低分位数数据。
- **黄金数据集的退化问题。** 未版本控制的评估数据集会随时间发生变化，从而破坏纵向对比分析，每次修改都需为数据集添加标签。
- **大语言模型的成本问题。** 在大规模应用中，模型调用次数是主要成本来源，应选用能达到校准阈值的最低成本模型，例如 GPT-4o-mini、Claude Haiku、Mistral-small。

## 使用它

2026年技术栈：

| 使用场景 | 框架 |
|---------|-----------|
| RAG质量监控 | RAGAS（4项指标） |
| CI/CD回归检测 | DeepEval + pytest |
| 自定义领域评估标准 | DeepEval中的G-Eval |
| 在线实时流量监控 | 无参考模式的RAGAS |
| 人工介入的抽查 | 带有标注界面的LangSmith或Phoenix |
| 红队测试/安全性评估 | Promptfoo + DeepEval |

典型技术栈：使用RAGAS进行监控，DeepEval用于CI流程，G-Eval用于评估新维度。同时运行这三个工具，它们之间的差异能提供有价值的参考信息。

## 发布它

保存为 `outputs/skill-eval-architect.md`：

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## 练习题

1. **简单。** 使用 RAGAS 对 10 个存在已知幻觉问题的 RAG 示例进行测试，验证忠实度指标能否检测出所有问题。
2. **中等。** 手动为 50 条问答答案标注 0-1 的正确性分数，并使用 G-Eval 进行评分；同时计算评测人员评分与人工评分之间的斯皮尔曼 rho 值。
3. **困难。** 构建基于 DeepEval 的 pytest CI 阶段门控机制，故意让检索器出现性能退化，验证该阶段门控能否成功触发失败判定；此外，通过对最低 10% 的结果进行阈值检测来实现最低分位数警报功能。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| LLM-as-judge | 使用LLM进行评分 | 提示判定模型根据评分标准为输出结果赋予0-1的分数。 |
| RAGAS | RAG指标库 | 一个开源的评估框架，包含4种无需参考文献的RAG指标。 |
| Faithfulness | 答案是否有依据？ | 回答内容中由检索到的上下文所支撑的比例。 |
| Context precision | 检索的片段是否相关？ | 属于Top-K检索片段中真正有用的比例。 |
| Context recall | 检索是否覆盖了所有内容？ | 金标准答案中的主张有多少能被检索到的片段所支持。 |
| G-Eval | 自定义LLM判定器 | 结合评分标准、思维链评估步骤以及0-1的分数输出。 |
| Calibration | 相信但需验证 | 判定模型的分数与人类评分之间的斯皮尔曼相关系数。 |

## 延伸阅读

- [Es 等人 (2023). RAGAS：检索增强生成的自动化评估](https://arxiv.org/abs/2309.15217) —— RAGAS 论文。
- [Liu 等人 (2023). G-Eval：利用 GPT-4 进行更符合人类偏好的自然语言生成评估](https://arxiv.org/abs/2303.16634) —— G-Eval 论文。
- [DeepEval 文档](https://deepeval.com/docs/metrics-introduction) —— 开源的生产级工具栈。
- [Zheng 等人 (2023). 利用 MT-Bench 和 Chatbot Arena 评估作为评判者的 LLM](https://arxiv.org/abs/2306.05685) —— 偏见、校准及局限性分析。
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) —— 整合 RAGAS、DeepEval 和 Phoenix 的统一框架。
