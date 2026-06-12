# 机器翻译

> 翻译工作是过去三十年资助自然语言处理研究的动力，至今依然如此。

**类型：** 构建
**编程语言：** Python
**先修知识：** 第 5 阶段 · 10（注意力机制）、第 5 阶段 · 04（GloVe、FastText、子词分解）
**耗时：** 约 75 分钟

## 问题所在

模型能够读取一种语言的句子，并生成另一种语言的句子。句子长度各异，词序也有所不同。某些源语言词汇可能对应多个目标语言词汇，反之亦然。习语更无法实现一一映射。法语中“我思念你”对应的表达是“tu me manques”，字面意思为“你在我这里缺失了”。此类情况下根本不存在词级对齐。

机器翻译这一任务促使自然语言处理领域诞生了编码器-解码器、注意力机制、Transformer架构，最终形成了整个大语言模型范式。每一次技术进步都源于翻译质量的可量化衡量，以及人类与机器之间始终存在的巨大差距。

本课程将跳过历史背景介绍，直接讲解2026年的实际应用流程：预训练的多语言编码器-解码器（如NLLB-200或mBART）、子词分词、束搜索算法、BLEU和chrF评估指标，以及那些仍会流入生产环境却未被发现的少数故障模式。

## 概念概述

![MT流程：分词 → 编码 → 带注意力机制的解码 → 反分词](../assets/mt-pipeline.svg)

现代机器翻译系统是基于平行文本训练的Transformer编码器-解码器。编码器以目标语言的分词方式读取源语言文本。解码器则通过交叉注意力机制利用编码器的输出，逐个子词生成目标语言文本（参见第10课）。为避免贪婪解码带来的问题，解码过程会采用束搜索算法。最终生成的文本需经过反分词、去大小写处理，并与参考译文进行对比评分。

实际应用中，有三种关键因素决定了机器翻译的质量。

- **分词器**：基于混合语言语料库训练的SentencePiece BPE分词器。跨语言共享的词汇表使得零样本对齐在NLLB任务中成为可能。
- **模型规模**：精简后的600M参数版NLLB-200可运行在笔记本电脑上；官方推荐的生产环境默认模型为3.3B参数版的NLLB-200，而54.5B参数版则是当前研究领域的最高上限。
- **解码策略**：一般内容场景下，束宽通常设置为4-5；为防止输出过短，会引入长度惩罚机制；在需要术语一致性的场景中，则采用受限解码方式。

```figure
seq2seq-alignment
```

## 构建它

### 步骤 1：调用预训练的机器翻译模型

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

此处有三点需要注意。`src_lang` 用于告知分词器应采用何种脚本及分割规则；`forced_bos_token_id` 则用于指示解码器生成哪种语言的文本。这两者均为 NLLB 系统特有的技巧；mBART 和 M2M-100 使用各自的规范，无法相互替代。

### 步骤 2：BLEU 与 chrF

BLEU用于衡量输出文本与参考文本之间的n-gram重叠程度。它包含四个参考n-gram长度参数（1-4），通过精度的几何平均值得分，并会对过短的输出施加简洁性惩罚。该分数范围在[0, 100]之间，应用十分广泛。但其解读较为复杂：30分的BLEU可视为“可用”；40分为“良好”；50分为“优秀”；而分数差异小于1分时通常可视为无显著差别。

chrF则用于衡量字符级别的F分数。对于形态变化丰富的语言，它比BLEU更能准确反映文本匹配程度，因此常与BLEU一同使用。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

始终使用 `sacrebleu`。它能对分词结果进行标准化处理，从而使不同论文的评分具有可比性。若自行实现 BLEU 计算方法，就会导致基准测试结果出现误导性。

### 三级评估体系（2026版）

现代机器翻译评估主要采用三类互补的指标体系，实际应用中至少需选用两类。

- **启发式指标**（BLEU、chrF）。这类指标计算速度快，基于参考译文，结果易于解释，且对意译方式不敏感。适用于传统系统的对比分析及性能退化检测。
- **学习型指标**（COMET、BLEURT、BERTScore）。这类指标通过神经网络模型结合人类评分标准进行训练，用于比较译文与源语言文本及参考译文之间的语义相似度。自2023年起，COMET在机器翻译研究领域的相关性最高，并将成为2026年对质量要求较高的生产环境中的默认评估指标。
- **以大语言模型作为评判者**（无参考译文）。通过向大型语言模型发送提示词，让其从流畅度、完整性、语气及文化适配性等方面对译文进行评分。当评分标准设计合理时，GPT-4-as-judge的评分与人类评分的一致率可达约80%。此类方法适用于没有参考译文的开放式内容评估。

2026年的实际应用方案为：使用 `sacrebleu` 计算BLEU和chrF指标，使用 `unbabel-comet` 计算COMET指标，最后再通过提示大型语言模型生成面向最终用户的评分结果。在将这些指标应用于生产数据之前，必须先用50至100个由人类标注的样本对它们进行校准。

无参考译文指标（COMET-QE、BLEURT-QE、LLM-as-judge）允许在没有参考译文的情况下对译文进行评估，这对于那些缺乏参考译文的冷门语言对尤为重要。

### 步骤 3：生产环境中会出现哪些故障

上述工作流程在80%的情况下能够流畅完成翻译，而在剩余的20%情况下则会无声地失败。常见的故障模式如下：

- **幻觉现象**：模型编造了源文本中不存在的内容。在不熟悉的领域词汇中尤为常见。症状为输出虽然通顺，但声称的事实并非源文本所提及。缓解措施包括对领域术语进行约束解码、对受监管内容进行人工审核，以及监控输出长度是否远超过输入长度。
- **目标语言错误**：模型将文本翻译成了错误的语言。在罕见的语种对中，NLLB模型极易出现此问题。缓解措施包括验证`forced_bos_token_id`的值，并始终使用带有语言ID检查的模型进行解码。
- **术语一致性偏差**：同一文档中的“Sign up”在某处被译为“s'inscrire”，而在另一处又被译为“créer un compte”。对于用户界面文本和面向用户的字符串而言，一致性比单纯的翻译质量更为重要。缓解措施包括采用受词汇表约束的解码方式或使用后编辑词典。
- **正式程度不匹配**：法语中的“tu”与“vous”，以及日语中的敬语等级。模型会选择在训练数据中出现频率更高的形式。对于面向客户的内容而言，这种选择通常是不正确的。缓解措施包括在提示词前添加表示正式程度的标记（如果模型支持的话），或者使用仅包含正式用语的语料对小型模型进行微调。
- **短输入导致的翻译过长**：非常短的输入句子往往会产生过长的译文，因为当源文本token数低于约5个时，长度惩罚会急剧上升。缓解措施是对翻译结果设置与源文本长度成比例的硬性最大长度限制。

### 步骤 4：针对特定领域的微调

预训练模型属于通用型模型。在法律、医疗或游戏对话等领域的翻译中，若能使用该领域内的平行数据对其进行微调，效果将显著提升。其实现方法并不复杂：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

几千个高质量的同构示例，远胜于几十万个包含噪声的网页爬取数据。训练数据的质量是提升系统性能的最关键因素。

## 使用它

2026年机器翻译的生产环境技术栈：

| 应用场景 | 推荐的初始模型 |
|---------|---------------------------|
| 任意语言对互译，支持200种语言 | `facebook/nllb-200-distilled-600M`（笔记本端）或 `nllb-200-3.3B`（生产环境） |
| 以英语为主，追求高质量，支持50种语言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 小批量处理，需要低成本推理，支持英法/德西互译 | Helsinki-NLP / Marian模型 |
| 对延迟要求极高的浏览器端应用 | 已量化的ONNX格式Marian模型（约50 MB） |
| 需要最高质量且愿意承担更高成本 | 带有翻译提示的GPT-4 / Claude / Gemini模型 |

截至2026年，对于多种语言对而言，大型语言模型在处理惯用表达及长上下文任务时的表现已优于专用机器翻译模型。其代价在于每个标记的处理成本与延迟。当上下文长度、风格一致性或通过提示进行领域适配比处理吞吐量更为重要时，应选择大型语言模型。

## 发布它

保存为 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 练习题

1. **简单级。** 使用 `nllb-200-distilled-600M` 将一段5句话的英文段落先翻译成法文，然后再译回英文。测量该往返翻译结果与原文的相似度。预期结果应保持语义一致性，但用词可能会出现偏差。
2. **中等级。** 利用 `fasttext lid.176` 或 `langdetect` 对翻译输出进行语言识别检查。将该功能集成到机器翻译调用流程中，以便在返回结果之前捕获错误翻译。
3. **高级别。** 选择任意一个包含5,000对文本的领域语料库，对 `nllb-200-distilled-600M` 模型进行微调。在微调前后，使用保留的测试集计算BLEU分数，并分析哪些类型的句子表现有所提升，哪些出现了退化。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| BLEU | 翻译得分 | 带有简洁性惩罚的n-gram精确度。取值范围为[0, 100]。 |
| chrF | 字符F分数 | 基于字符级别的F分数。对形态丰富的语言更为敏感。 |
| NMT | 神经机器翻译 | 在平行文本上训练的Transformer编码器-解码器模型。2017年及之后的默认选择。 |
| NLLB | 不让任何语言掉队 | Meta推出的包含200种语言的机器翻译模型系列。 |
| Constrained decoding | 受限解码 | 强制要求特定标记或n-gram出现在输出中或不出现在输出中。 |
| Hallucination | 虚构内容 | 模型生成的、在源文本中不存在的内容。 |

## 延伸阅读

- [Costa-jussà 等人 (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 论文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 为何 `sacrebleu` 是报告 BLEU 分数的唯一正确方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 论文。
- [Hugging Face 机器翻译指南](https://huggingface.co/docs/transformers/tasks/translation) — 实用的微调操作指南。
