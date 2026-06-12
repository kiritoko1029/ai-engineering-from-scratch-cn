# 子词分词技术 —— BPE、WordPiece、Unigram、SentencePiece

> 单词分词器在遇到未见过的单词时会卡住。字符分词器会导致序列长度急剧增加。子词分词器则能在两者之间取得平衡。所有现代大型语言模型都采用了其中一种分词方式。

**类型：** 学习
**语言：** Python
**先修课程：** 第5阶段 · 01（文本处理）、第5阶段 · 04（GloVe / FastText / 子词）
**时长：** 约60分钟

## 问题所在

您的词汇量包含50,000个单词。当用户输入“untokenizable”时，分词器会将其返回为`[UNK]`。此时模型便无法获取该单词的任何特征信息。更糟糕的是，您语料库中第90百分位的文档包含了40个稀有单词，这意味着每份文档会有40比特的信息丢失。

子词分词技术可以解决这一问题。常见单词仍保持为单个标记，而稀有单词则会被拆解为有意义的片段：`untokenizable` → `un`、`token`、`izable`。由于任何字符串归根结底都是字节序列，因此训练数据能够覆盖所有情况。

2026年推出的每一款前沿大语言模型都会采用BPE、Unigram或WordPiece这三种算法中的一种，并通过tiktoken、SentencePiece或HF Tokenizers这三种库之一进行封装。若不选择其中任何一种，便无法部署语言模型。

## 概念概述

![BPE、Unigram与WordPiece的逐字符对比](../assets/subword-tokenization.svg)

**BPE（字节对编码）。** 从字符级词汇表开始。统计所有相邻字符对，将出现频率最高的字符对合并为一个新标记。重复此过程直至达到目标词汇量。常见应用算法包括：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**字节级BPE。** 算法相同，但以原始字节（共256个基础标记）而非Unicode字符作为处理单位。可确保不存在`[UNK]`标记——任何字节序列均可被编码。GPT-2使用了50,257个标记（256个字节 + 50,000次合并 + 1个特殊标记）。

**Unigram。** 从庞大的词汇表开始，为每个标记分配一个单字符概率值。通过迭代方式移除那些移除后对语料库对数似然影响最小的标记。在推理阶段具有概率特性：可生成多种标记化结果（有助于通过子词规则进行数据增强）。被T5、mBART、ALBERT、XLNet、Gemma等模型采用。

**WordPiece。** 优先合并那些能最大程度提升训练语料库似然度的字符对，而非仅依据原始出现频率。被BERT、DistilBERT、ELECTRA等模型使用。

**SentencePiece与tiktoken的对比。** SentencePiece是一个可直接在原始Unicode文本上训练词汇表（BPE或Unigram）的库，它将空白字符编码为`▁`。tiktoken则是OpenAI开发的快速*编码器*，用于基于预构建的词汇表进行编码，本身不负责训练。

经验法则：

- **训练新词汇表：** 使用SentencePiece（支持多语言，无需预先分词）或HF Tokenizers。
- **针对GPT词汇表的快速推理：** 使用tiktoken（cl100k_base、o200k_base版本）。
- **兼顾训练与推理：** 选择HF Tokenizers——同一个库即可完成训练和部署任务。

```figure
bpe-merge
```

## 构建它

### 步骤 1：从零实现 BPE

参见 `code/main.py`。该循环如下：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

该算法编码了三个关键事实。`</w>` 用于标记单词结尾，从而确保后缀“low”与前缀“lower”能够被区分开来。频率加权机制使得高频出现的词对能更早地胜出。合并列表是有序的——推理过程会按照训练时的顺序进行合并操作。

### 步骤 2：使用学到的合并规则进行编码

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

朴素算法的时间复杂度为 O(n·|merges|)。实际生产环境中的实现（如 tiktoken、HF Tokenizers）采用基于优先队列的合并等级查找机制，可实现近乎线性的运行时间。

### 步骤 3：SentencePiece 的实际应用

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：无需进行预分词处理，空格将编码为`▁`；`character_coverage`参数用于控制在保留罕见字符与将其映射为`<unk>`之间采取的策略强度。

### 步骤 4：为兼容 OpenAI 的词汇集配置 tiktoken

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

仅编码模式。速度极快（基于 Rust 后端）。采用与 GPT-4/5 相同的分词方式，可精确计算字节数、成本预估以及上下文窗口容量。

## 2026年仍会存在的缺陷与隐患

- **分词器漂移。** 使用词汇表 A 进行训练，但实际部署时使用的是词汇表 B。由于 Token ID 不同，模型输出结果会变得无意义。请在持续集成流程中检查 `tokenizer.json` 的哈希值。
- **空白字符的歧义性。** BPE 分词算法会将 “hello” 和 “ hello” 视为不同的 Token。务必明确指定 `add_special_tokens` 和 `add_prefix_space` 参数。
- **多语言数据训练不足。** 以英语为主的语料库会导致词汇表规模大幅增加，从而使非拉丁字母的字符被拆分为更多 Token（数量可能增加 5 到 10 倍）。在 GPT-3.5 上，处理日语或阿拉伯语时的相同提示词所需成本也会高出 5 到 10 倍。o200k_base 模型在一定程度上解决了这一问题。
- **表情符号的拆分问题。** 单个表情符号可能会被拆分为 5 个 Token。在规划上下文长度预算时，需注意模型对表情符号的处理方式。

## 使用它

2026年推荐技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 从零开始训练单语模型 | HF Tokenizers（BPE） |
| 训练多语言模型 | SentencePiece（Unigram，`character_coverage=0.9995`） |
| 提供兼容OpenAI的API服务 | tiktoken（GPT-4+推荐使用`o200k_base`） |
| 需要领域专用词汇表（代码、数学、蛋白质等） | 在该领域语料上训练自定义BPE，再与基础词汇表合并 |
| 边缘设备推理及小型模型场景 | Unigram（较小的词汇表表现更佳） |

词汇表大小属于规模调整参数，并非固定值。大致经验法则如下：参数量<10亿时使用32k，10亿至100亿时使用50-100k，多语言或前沿模型则需200k以上。

## 发布它

保存为 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 练习题

1. **简单。** 使用 `code/main.py` 中的小型语料库训练一个包含500个合并规则的BPE模型。对三个保留的单词进行编码，统计其中生成恰好1个标记与生成多于1个标记的分别有多少。
2. **中等。** 比较 `cl100k_base`、`o200k_base` 以及使用词汇表大小为32k自行训练的SentencePiece BPE模型在100句英文维基百科文本上的标记数量。报告每种模型的压缩比。
3. **困难。** 使用BPE、Unigram和WordPiece在同一语料库上分别进行训练。在小型情感分类器上测试使用这些模型时的下游准确率。这种选择能否使F1分数的提升超过1分？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| BPE | 字节对编码 | 通过贪婪地合并出现频率最高的字符对，直至达到目标词汇表大小。 |
| 字节级 BPE | 永不出现未知标记 | 在原始的 256 个字节上进行 BPE 处理；GPT-2 / Llama 即采用此方法。 |
| Unigram | 概率分词器 | 基于对数似然值从大量候选项中筛选出最优结果；T5、Gemma 等模型使用该方式。 |
| SentencePiece | 处理空白字符的分词器 | 一种可在原始文本上训练 BPE/Unigram 的库；空格会被编码为 `▁`。 |
| tiktoken | 高速分词器 | OpenAI 开发的基于 Rust 的 BPE 编码器，适用于预构建的词汇表，无需训练过程。 |
| 合并列表 | 魔法数字序列 | 一个按顺序排列的 `(a, b) → ab` 合并规则列表；推理时将按照该顺序应用这些规则。 |
| 字符覆盖率 | 多少才算过于罕见？ | 分词器必须覆盖的训练语料库中字符的比例；通常要求在 0.9995 左右。 |

## 延伸阅读

- [Sennrich, Haddow, Birch (2015). 基于子词单元的稀有词神经机器翻译](https://arxiv.org/abs/1508.07909) —— BPE相关论文。
- [Kudo (2018). 利用单语模型进行子词正则化](https://arxiv.org/abs/1804.10959) —— Unigram相关论文。
- [Kudo, Richardson (2018). SentencePiece：一种简单且与语言无关的子词分词器](https://arxiv.org/abs/1808.06226) —— 对应的库。
- [Hugging Face — 分词器概要](https://huggingface.co/docs/transformers/tokenizer_summary) —— 简洁的参考资料。
- [OpenAI tiktoken仓库](https://github.com/openai/tiktoken) —— 实用指南及编码列表。
