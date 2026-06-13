# 评估：基准测试、Evals、LM Harness

> 古德哈特定律（Goodhart's Law）：当一项指标本身成为目标时，它就不再是一个好的指标。每一家前沿实验室都在「刷」基准测试。MMLU 分数节节攀升，可模型连「strawberry」里有几个 R 都数不准。唯一真正重要的 eval 是「你的」eval——在「你的」任务上，用「你的」数据。

**类型：** 构建
**语言：** Python
**前置要求：** 第 10 阶段，第 01-05 课（从零构建 LLM）
**所需时间：** 约90分钟

## 学习目标

- 构建一套自定义评估框架（harness），针对语言模型运行选择题与开放式基准测试
- 解释为什么标准基准测试（MMLU、HumanEval）会饱和，无法区分前沿模型
- 实现带有恰当指标的任务专属 eval：精确匹配（exact match）、F1、BLEU 以及 LLM-as-judge 评分
- 设计一套针对你具体用例的自定义评估套件，而不是单纯依赖公开排行榜

## 问题所在

MMLU 于 2020 年发布，涵盖 57 个学科、共 15,908 道题目。三年之内，前沿模型就把它刷到了饱和。GPT-4 得分 86.4%，Claude 3 Opus 得分 86.8%，Llama 3 405B 得分 88.6%。整个排行榜被压缩进 3 个百分点的区间，其中的差异是统计噪声，而非真实的能力差距。

与此同时，这些模型却在一个 10 岁小孩想都不用想就能完成的任务上翻车。在 MMLU 上拿到 88.7% 的 Claude 3.5 Sonnet，起初连「strawberry」里有几个字母都数不出来——这个任务不需要任何世界知识、不需要任何推理，只需要逐字符地遍历。HumanEval 用 164 道题考查代码生成。模型在它上面得分超过 90%，但写出的代码仍会在任何初级开发者都能想到的边界情况上崩溃。

基准测试表现与现实可靠性之间的鸿沟，正是 LLM 评估的核心问题。基准测试告诉你的是模型在该基准上表现如何，却几乎无法说明这个模型在你的具体任务上、用你的具体数据、在你的具体失败模式下会有怎样的表现。如果你在构建一个客服机器人，MMLU 毫不相关。如果你在构建一个代码助手，HumanEval 也只覆盖函数级别的生成——它对调试、重构、跨文件解释代码只字未提。

你需要自定义 eval。这并不是说基准测试毫无用处——它们对粗略的模型选型很有用——而是因为最终评估必须与你的部署条件精确吻合。

## 概念说明

### Eval 全景

评估分为三大类，各自的成本与信号质量都不同。

**基准测试（Benchmarks）** 是标准化的测试套件。MMLU、HumanEval、SWE-bench、MATH、ARC、HellaSwag。你让模型跑一遍基准，得到一个分数。优点：大家用的是同一套测试，因此可以横向比较模型。缺点：模型与训练数据越来越多地「污染」了这些基准。实验室在包含基准题目的数据上训练，分数上去了，能力却未必。

**自定义 eval（Custom evals）** 是你为自己具体用例构建的测试套件。你定义输入、预期输出和评分函数。法律文档摘要器要在法律文档上评估，SQL 生成器要在你的数据库 schema 上评估。它们构建成本高昂，但却是唯一能够预测生产表现的评估。

**人工 eval（Human evals）** 雇用付费标注员，从有用性、正确性、流畅度和安全性等维度判断模型输出。对于自动评分失效的开放式任务，这是黄金标准。Chatbot Arena 已在 100 多个模型上收集了超过 200 万次人类偏好投票。缺点：成本（每次判断 $0.10-$2.00）和速度（数小时到数天）。

```mermaid
graph TD
    subgraph Eval["Evaluation Landscape"]
        direction LR
        B["Benchmarks\n(MMLU, HumanEval)\nCheap, standardized\nGameable, stale"]
        C["Custom Evals\nYour task, your data\nHighest signal\nExpensive to build"]
        H["Human Evals\n(Chatbot Arena)\nGold standard\nSlow, costly"]
    end

    B -->|"rough model selection"| C
    C -->|"ambiguous cases"| H

    style B fill:#1a1a2e,stroke:#ffa500,color:#fff
    style C fill:#1a1a2e,stroke:#51cf66,color:#fff
    style H fill:#1a1a2e,stroke:#e94560,color:#fff
```

### 为什么基准测试会失效

有三种机制会让基准分数不再反映真实能力。

**数据污染（Data contamination）。** 训练语料是从互联网抓取的。基准题目就在互联网上。模型在训练时见过答案。这并非传统意义上的作弊——实验室并非有意纳入基准数据。但网络级别的抓取使得几乎不可能将这些数据排除在外。

**应试训练（Teaching to the test）。** 实验室会针对基准表现优化训练数据配比。如果训练混合中有 5% 是 MMLU 风格的选择题，模型就会学到题目的格式和答案分布。MMLU 是四选一的选择题。模型会学到答案大致在 A/B/C/D 之间均匀分布，这一点即便在模型不知道答案时也能起到帮助。

**饱和（Saturation）。** 当每个前沿模型在某个基准上都拿到 85-90% 时，这个基准就失去了区分度。剩下那 10-15% 的题目可能本身就含糊不清、标注错误，或需要冷门的领域知识。MMLU 从 87% 提升到 89% 也许只意味着模型多背下了两道偏题，而非变得更聪明。

### 困惑度（Perplexity）：一次快速体检

困惑度衡量模型对一串 token 序列有多「意外」。形式上，它是平均负对数似然取指数的结果：

```
PPL = exp(-1/N * sum(log P(token_i | context)))
```

困惑度为 10 意味着模型平均而言，在每个 token 位置上就像在 10 个选项中均匀随机选择一样不确定。越低越好。GPT-2 在 WikiText-103 上的困惑度约为 30，GPT-3 约为 20，Llama 3 8B 约为 7。

困惑度对在同一测试集上比较模型很有用，但它有盲区。一个模型可能因为擅长预测常见模式而获得低困惑度，却在罕见但重要的模式上表现糟糕。它对指令遵循、推理或事实准确性也只字未提。把它当作一次合理性检查（sanity check），而非最终裁决。

### LLM-as-Judge

用一个强模型来评估一个弱模型的输出。思路很简单：让 GPT-4o 或 Claude Sonnet 在正确性、有用性和安全性上给一段回答打 1-5 分。用 GPT-4o-mini 时每次判断成本约 $0.01，而且与人类判断的相关性出奇地高——在大多数任务上约 80% 的一致率。

评分提示词比模型本身更重要。模糊的提示词（「给这段回答打分」）会产生噪声很大的分数。带有评分细则的结构化提示词（「若答案事实正确且引用了来源则打 5 分，正确但无来源打 4 分，部分正确打 3 分……」）则会产生一致、可复现的分数。

失败模式：评审模型存在位置偏好（在两两比较中偏好第一个回答）、冗长偏好（偏好更长的回答）和自我偏好（GPT-4 给 GPT-4 的输出打的分高于同等水平的 Claude 输出）。缓解办法：随机化顺序、按长度做归一化、使用与被评估模型不同的评审模型。

### 从两两比较中得出 ELO 评分

这是 Chatbot Arena 的做法。对同一提示词展示来自不同模型的两段回答，由人类（或 LLM 评审）选出更好的一个。从成千上万次这样的比较中，为每个模型计算一个 ELO 评分——和国际象棋用的是同一套体系。

ELO 的优势：相对排名比绝对评分更可靠，能优雅地处理平局，而且比逐个独立评分每条输出收敛得更快。截至 2026 年初，Chatbot Arena 的排名显示 GPT-4o、Claude 3.5 Sonnet 和 Gemini 1.5 Pro 在榜首位置相互之间相差不到 20 个 ELO 点。

```mermaid
graph LR
    subgraph ELO["ELO Rating Pipeline"]
        direction TB
        P["Prompt"] --> MA["Model A Output"]
        P --> MB["Model B Output"]
        MA --> J["Judge\n(Human or LLM)"]
        MB --> J
        J --> W["A Wins / B Wins / Tie"]
        W --> E["ELO Update\nK=32"]
    end

    style P fill:#1a1a2e,stroke:#0f3460,color:#fff
    style J fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#51cf66,color:#fff
```

### Eval 框架

**lm-evaluation-harness**（EleutherAI）：标准的开源 eval 框架。支持 200 多个基准。用一条命令就能让任意 Hugging Face 模型跑 MMLU、HellaSwag、ARC 等。Open LLM Leaderboard 即采用它。

**RAGAS**：专为 RAG 流水线打造的评估框架。衡量忠实度（faithfulness，答案是否与检索到的上下文一致）、相关性（relevance，检索到的上下文是否与问题相关）以及答案正确性。

**promptfoo**：面向提示词工程的、由配置驱动的 eval。在 YAML 中定义测试用例，针对多个模型运行，得到一份通过/失败报告。它对提示词的回归测试很有用——确保一次提示词改动不会破坏已有的测试用例。

### 构建自定义 Eval

这是唯一对生产环境重要的 eval。流程如下：

1. **定义任务。** 模型到底应该做什么？要精确。「回答问题」太含糊。「给定一封客户投诉邮件，提取产品名称、问题类别和情绪」才是一个你能够评估的任务。

2. **创建测试用例。** 原型 eval 至少 50 个，生产环境 200 个以上。每个测试用例都是一对 (input, expected_output)。要包含边界情况：空输入、对抗性输入、含糊不清的输入、其他语言的输入。

3. **定义评分。** 结构化输出用精确匹配，文本相似度用 BLEU/ROUGE，开放式质量用 LLM-as-judge，提取任务用 F1。用权重把多个指标组合起来。

4. **自动化。** 每个 eval 都用一条命令运行。不要有手动步骤。把结果以一种便于跨时间比较的格式存储下来。

5. **持续追踪。** 孤立的一个 eval 分数毫无意义。你需要的是趋势线。上次提示词改动后分数提升了吗？切换模型后回退了吗？把你的 eval 和提示词一起做版本管理。

| Eval 类型 | 每次判断成本 | 与人类的一致率 | 最适合 |
|-----------|------------------|----------------------|----------|
| 精确匹配 | ~$0 | 100%（适用时） | 结构化输出、分类 |
| BLEU/ROUGE | ~$0 | ~60% | 翻译、摘要 |
| LLM-as-judge | ~$0.01 | ~80% | 开放式生成 |
| 人工 eval | $0.10-$2.00 | 不适用（本身即真值） | 含糊、高风险任务 |

```figure
perplexity-loss
```

## 开始构建

### 第 1 步：一个极简的 Eval 框架

定义核心抽象。一个 eval 用例包含一个输入、一个预期输出以及一个可选的元数据字典。一个评分器接收一个预测和一个参考答案，返回一个 0 到 1 之间的分数。

```python
import json
from collections import Counter

class EvalCase:
    def __init__(self, input_text, expected, metadata=None):
        self.input_text = input_text
        self.expected = expected
        self.metadata = metadata or {}

class EvalSuite:
    def __init__(self, name, cases, scorers):
        self.name = name
        self.cases = cases
        self.scorers = scorers

    def run(self, model_fn):
        results = []
        for case in self.cases:
            prediction = model_fn(case.input_text)
            scores = {}
            for scorer_name, scorer_fn in self.scorers.items():
                scores[scorer_name] = scorer_fn(prediction, case.expected)
            results.append({
                "input": case.input_text,
                "expected": case.expected,
                "prediction": prediction,
                "scores": scores,
            })
        return results
```

### 第 2 步：评分函数

构建精确匹配、token F1，以及一个模拟的 LLM-as-judge 评分器。

```python
def exact_match(prediction, expected):
    return 1.0 if prediction.strip().lower() == expected.strip().lower() else 0.0

def token_f1(prediction, expected):
    pred_tokens = set(prediction.lower().split())
    exp_tokens = set(expected.lower().split())
    if not pred_tokens or not exp_tokens:
        return 0.0
    common = pred_tokens & exp_tokens
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(exp_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def llm_judge_simulated(prediction, expected):
    pred_words = set(prediction.lower().split())
    exp_words = set(expected.lower().split())
    if not exp_words:
        return 0.0
    overlap = len(pred_words & exp_words) / len(exp_words)
    length_penalty = min(1.0, len(prediction) / max(len(expected), 1))
    return round(overlap * 0.7 + length_penalty * 0.3, 3)
```

### 第 3 步：ELO 评分系统

实现带 ELO 更新的两两比较。这正是 Chatbot Arena 用来给模型排名的系统。

```python
class ELOTracker:
    def __init__(self, k=32, initial_rating=1500):
        self.ratings = {}
        self.k = k
        self.initial_rating = initial_rating
        self.history = []

    def _ensure_player(self, name):
        if name not in self.ratings:
            self.ratings[name] = self.initial_rating

    def expected_score(self, rating_a, rating_b):
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def record_match(self, player_a, player_b, outcome):
        self._ensure_player(player_a)
        self._ensure_player(player_b)

        ea = self.expected_score(self.ratings[player_a], self.ratings[player_b])
        eb = 1 - ea

        if outcome == "a":
            sa, sb = 1.0, 0.0
        elif outcome == "b":
            sa, sb = 0.0, 1.0
        else:
            sa, sb = 0.5, 0.5

        self.ratings[player_a] += self.k * (sa - ea)
        self.ratings[player_b] += self.k * (sb - eb)

        self.history.append({
            "a": player_a, "b": player_b,
            "outcome": outcome,
            "rating_a": round(self.ratings[player_a], 1),
            "rating_b": round(self.ratings[player_b], 1),
        })

    def leaderboard(self):
        return sorted(self.ratings.items(), key=lambda x: -x[1])
```

### 第 4 步：困惑度计算

用 token 概率计算困惑度。实践中你会从模型的 logits 得到这些概率。这里我们用一个概率分布来模拟。

```python
import numpy as np

def perplexity(log_probs):
    if not log_probs:
        return float("inf")
    avg_neg_log_prob = -np.mean(log_probs)
    return float(np.exp(avg_neg_log_prob))

def token_log_probs_simulated(text, model_quality=0.8):
    np.random.seed(hash(text) % 2**31)
    tokens = text.split()
    log_probs = []
    for i, token in enumerate(tokens):
        base_prob = model_quality
        if len(token) > 8:
            base_prob *= 0.6
        if i == 0:
            base_prob *= 0.7
        prob = np.clip(base_prob + np.random.normal(0, 0.1), 0.01, 0.99)
        log_probs.append(float(np.log(prob)))
    return log_probs
```

### 第 5 步：汇总结果

在一次 eval 运行中计算汇总统计：均值、中位数、达到阈值的通过率，以及按指标的细分。

```python
def summarize_results(results, threshold=0.8):
    all_scores = {}
    for r in results:
        for metric, score in r["scores"].items():
            all_scores.setdefault(metric, []).append(score)

    summary = {}
    for metric, scores in all_scores.items():
        arr = np.array(scores)
        summary[metric] = {
            "mean": round(float(np.mean(arr)), 3),
            "median": round(float(np.median(arr)), 3),
            "std": round(float(np.std(arr)), 3),
            "min": round(float(np.min(arr)), 3),
            "max": round(float(np.max(arr)), 3),
            "pass_rate": round(float(np.mean(arr >= threshold)), 3),
            "n": len(scores),
        }
    return summary

def print_summary(summary, suite_name="Eval"):
    print(f"\n{'=' * 60}")
    print(f"  {suite_name} Summary")
    print(f"{'=' * 60}")
    for metric, stats in summary.items():
        print(f"\n  {metric}:")
        print(f"    Mean:      {stats['mean']:.3f}")
        print(f"    Median:    {stats['median']:.3f}")
        print(f"    Std:       {stats['std']:.3f}")
        print(f"    Range:     [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"    Pass rate: {stats['pass_rate']:.1%} (threshold >= 0.8)")
        print(f"    N:         {stats['n']}")
```

### 第 6 步：运行完整流水线

把所有部分串联起来。定义一个任务、创建测试用例、模拟两个模型、运行 eval、从两两比较中计算 ELO，并打印排行榜。

```python
def demo_model_good(prompt):
    responses = {
        "What is the capital of France?": "Paris",
        "What is 2 + 2?": "4",
        "Who wrote Hamlet?": "William Shakespeare",
        "What language is PyTorch written in?": "Python and C++",
        "What is the boiling point of water?": "100 degrees Celsius",
    }
    return responses.get(prompt, "I don't know")

def demo_model_bad(prompt):
    responses = {
        "What is the capital of France?": "Paris is the capital city of France",
        "What is 2 + 2?": "The answer is four",
        "Who wrote Hamlet?": "Shakespeare",
        "What language is PyTorch written in?": "Python",
        "What is the boiling point of water?": "212 Fahrenheit",
    }
    return responses.get(prompt, "Unknown")

cases = [
    EvalCase("What is the capital of France?", "Paris"),
    EvalCase("What is 2 + 2?", "4"),
    EvalCase("Who wrote Hamlet?", "William Shakespeare"),
    EvalCase("What language is PyTorch written in?", "Python and C++"),
    EvalCase("What is the boiling point of water?", "100 degrees Celsius"),
]

suite = EvalSuite(
    name="General Knowledge",
    cases=cases,
    scorers={
        "exact_match": exact_match,
        "token_f1": token_f1,
        "llm_judge": llm_judge_simulated,
    },
)

results_good = suite.run(demo_model_good)
results_bad = suite.run(demo_model_bad)

print_summary(summarize_results(results_good), "Model A (concise)")
print_summary(summarize_results(results_bad), "Model B (verbose)")
```

「好」模型给出精确的答案。「坏」模型给出冗长的改述。精确匹配会严厉惩罚那个冗长的模型。Token F1 和 LLM-as-judge 则更宽容。这说明了为什么指标选择很重要：同一个模型，根据你的评分方式不同，看起来可以很出色，也可以很糟糕。

### 第 7 步：ELO 锦标赛

在多轮中对模型进行两两比较。

```python
elo = ELOTracker(k=32)

for case in cases:
    pred_a = demo_model_good(case.input_text)
    pred_b = demo_model_bad(case.input_text)

    score_a = token_f1(pred_a, case.expected)
    score_b = token_f1(pred_b, case.expected)

    if score_a > score_b:
        outcome = "a"
    elif score_b > score_a:
        outcome = "b"
    else:
        outcome = "tie"

    elo.record_match("model_a_concise", "model_b_verbose", outcome)

print("\nELO Leaderboard:")
for name, rating in elo.leaderboard():
    print(f"  {name}: {rating:.0f}")
```

### 第 8 步：困惑度比较

比较不同质量等级的「模型」之间的困惑度。

```python
test_text = "The quick brown fox jumps over the lazy dog in the garden"

for quality, label in [(0.9, "Strong model"), (0.7, "Medium model"), (0.4, "Weak model")]:
    log_probs = token_log_probs_simulated(test_text, model_quality=quality)
    ppl = perplexity(log_probs)
    print(f"  {label} (quality={quality}): perplexity = {ppl:.2f}")
```

## 实际运用

### lm-evaluation-harness（EleutherAI）

在任意模型上运行基准测试的标准工具。

```python
# pip install lm-eval
# Command line:
# lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B --tasks mmlu --batch_size 8

# Python API:
# import lm_eval
# results = lm_eval.simple_evaluate(
#     model="hf",
#     model_args="pretrained=meta-llama/Llama-3.1-8B",
#     tasks=["mmlu", "hellaswag", "arc_easy"],
#     batch_size=8,
# )
# print(results["results"])
```

### promptfoo

面向提示词工程的、由配置驱动的 eval。在 YAML 中定义测试，针对多个提供商运行。

```yaml
# promptfoo.yaml
providers:
  - openai:gpt-4o-mini
  - anthropic:claude-3-haiku

prompts:
  - "Answer in one word: {{question}}"

tests:
  - vars:
      question: "What is the capital of France?"
    assert:
      - type: contains
        value: "Paris"
  - vars:
      question: "What is 2 + 2?"
    assert:
      - type: equals
        value: "4"
```

### 用 RAGAS 做 RAG 评估

```python
# pip install ragas
# from ragas import evaluate
# from ragas.metrics import faithfulness, answer_relevancy, context_precision
#
# result = evaluate(
#     dataset,
#     metrics=[faithfulness, answer_relevancy, context_precision],
# )
# print(result)
```

RAGAS 衡量的正是通用 eval 所遗漏的：模型的答案是否扎根于检索到的上下文，而不仅仅是答案在抽象意义上是否「正确」。

## 交付成果

本课会产出 `outputs/prompt-eval-designer.md`——一个可复用的提示词，能为任何任务设计自定义 eval 套件。给它一段任务描述，它就会生成测试用例、评分函数以及一个通过/失败阈值建议。

它还会产出 `outputs/skill-llm-evaluation.md`——一个决策框架，帮你根据任务类型、预算和延迟要求选择正确的评估策略。

## 练习

1. 添加一个「一致性」评分器，把同一个输入跑 5 遍模型，衡量输出有多频繁地相互一致。对确定性输入给出不一致的答案，说明提示词脆弱或 temperature 设置过高。

2. 扩展 ELO 追踪器，使其支持多个评审函数（精确匹配、F1、LLM-as-judge）并为它们加权。比较一下：当你给精确匹配赋予很高权重，与给 F1 赋予很高权重时，排行榜会如何变化。

3. 为一个具体任务构建 eval 套件：把邮件分类到 5 个类别中。创建 100 个测试用例，包含多样化的样例以及边界情况（可能同时属于多个类别的邮件、空邮件、其他语言的邮件）。衡量不同「模型」（基于规则的、关键词匹配的、模拟 LLM 的）的表现。

4. 实现污染检测：给定一组 eval 题目和一份训练语料，检查有多大比例的 eval 题目（或相近的改述）出现在训练数据中。研究人员正是这样审计基准有效性的。

5. 构建一个「模型 diff」工具。给定两个模型版本的 eval 结果，高亮出哪些具体测试用例改进了、哪些回退了、哪些保持不变。这相当于 eval 版的代码 diff——对于理解一次改动是有益还是有害至关重要。

## 关键术语

| 术语 | 通常说法 | 真实含义 |
|------|----------------|----------------------|
| MMLU | 「那个基准」 | Massive Multitask Language Understanding——涵盖 57 个学科的 15,908 道选择题，到 2025 年已被刷到 88% 以上而饱和 |
| HumanEval | 「代码 eval」 | 来自 OpenAI 的 164 道 Python 函数补全题，只考查孤立的函数生成 |
| SWE-bench | 「真正的编码 eval」 | 来自 12 个 Python 仓库的 2,294 个 GitHub issue，衡量端到端的缺陷修复，包括测试生成 |
| 困惑度 | 「模型有多困惑」 | exp(-avg(log P(token_i given context)))——越低意味着模型给实际 token 赋予了越高的概率 |
| ELO 评分 | 「模型的国际象棋排名」 | 由两两胜负记录计算出的相对实力评分，Chatbot Arena 用它给 100 多个模型排名 |
| LLM-as-judge | 「用 AI 给 AI 打分」 | 一个强模型对照评分细则给弱模型的输出打分，与人类评审约 80% 一致，每次判断约 $0.01 |
| 数据污染 | 「模型见过测试」 | 训练数据包含基准题目，在不提升真实能力的情况下抬高分数 |
| Eval 套件 | 「一堆测试」 | 一组带版本管理的 (input, expected_output, scorer) 三元组，衡量某项具体能力 |
| 通过率 | 「答对的百分比」 | 评分高于某阈值的 eval 用例所占的比例——比均值分数更可操作，因为它衡量的是可靠性 |
| Chatbot Arena | 「模型排名网站」 | LMSYS 平台，拥有 200 万以上人类偏好投票，通过 ELO 评分产出最受信赖的 LLM 排行榜 |

## 延伸阅读

- [Hendrycks et al., 2021 -- "Measuring Massive Multitask Language Understanding"](https://arxiv.org/abs/2009.03300) -- MMLU 论文，尽管已经饱和，仍是被引用最多的 LLM 基准
- [Chen et al., 2021 -- "Evaluating Large Language Models Trained on Code"](https://arxiv.org/abs/2107.03374) -- 来自 OpenAI 的 HumanEval 论文，确立了代码生成的评估方法论
- [Zheng et al., 2023 -- "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685) -- 对用 LLM 评估 LLM 的系统性分析，包括位置偏好和冗长偏好的发现
- [LMSYS Chatbot Arena](https://chat.lmsys.org/) -- 拥有 200 万以上投票的众包模型比较平台，最受信赖的真实世界 LLM 排名
