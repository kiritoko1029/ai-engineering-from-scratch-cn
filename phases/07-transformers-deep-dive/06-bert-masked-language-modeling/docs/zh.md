# BERT — 掩码语言建模

> GPT用于预测下一个单词，而BERT则用于预测缺失的单词。仅这一句区别，便造就了半世纪以来以嵌入模型为核心的所有技术发展。

**类型：** 构建
**语言：** Python
**先修课程：** 第7阶段·05课（完整Transformer模型），第5阶段·02课（文本表示）
**耗时：** 约45分钟

## 问题所在

2018年时，各类NLP任务——情感分析、命名实体识别、问答系统、蕴含关系判断——都需要在各自的标注数据上从零开始训练专属模型。当时并不存在可供微调的预训练“理解英语”模型检查点。ELMo（2018年）证明了可以使用双向LSTM来预训练上下文嵌入，虽然有所帮助，但缺乏泛化能力。

BERT（Devlin等人，2018年）提出了这样一个设想：如果使用Transformer编码器，在互联网上的所有句子上进行预训练，并强制其根据两侧的上下文预测缺失的单词会怎样？之后再针对具体的下游任务对某个模块进行微调。参数效率的提升堪称一项重大突破。

其结果便是：在18个月内，BERT及其衍生模型（RoBERTa、ALBERT、ELECTRA）便占据了所有现有的NLP排行榜的榜首。到2020年，全球所有的搜索引擎、内容审核流程以及语义搜索系统都已采用了BERT技术。

截至2026年，仅包含编码器的模型依然是分类、检索和结构化信息提取任务的理想选择——它们的处理速度比带有解码器的模型快5至10倍，且其生成的嵌入向量是所有现代检索系统的核心基础。ModernBERT（2024年12月）通过引入Flash Attention、RoPE以及GeGLU技术，将该架构的上下文长度提升到了8K。

## 概念概述

![掩码语言建模：选择标记、对其进行掩码处理，然后预测原始内容](../assets/bert-mlm.svg)

### 训练信号

取一句句子：`the quick brown fox jumps over the lazy dog`。

随机屏蔽15%的标记：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

训练模型以预测被遮蔽位置处的原始标记。由于编码器是双向的，因此在第1位预测 `[MASK]` 时，可以利用第2位及之后的“brown fox jumps”作为参考。而这正是GPT所无法做到的。

### BERT 隐藏规则

在用于预测的 15% 的标记中：

- 80% 被替换为 `[MASK]`。
- 10% 被替换为一个随机标记。
- 10% 保持不变。

为何不始终使用 `[MASK]` 呢？因为推理阶段根本不会出现 `[MASK]`。如果让模型在所有被遮蔽的位置都预期出现 `[MASK]`，将会导致预训练与微调之间的分布偏移。保留 10% 的随机标记和 10% 的不变标记，有助于保持模型的真实性。

### 下一句预测（NSP）——及其被移除的原因

原始的 BERT 也在 NSP 数据集上进行了训练：给定两个句子 A 和 B，预测 B 是否紧跟在 A 之后。RoBERTa（2019）通过实验发现去除该模块后性能下降，表明 NSP 对模型并无帮助，反而有害。现代编码器已不再使用该模块。

### 2026年的变化：ModernBERT

2024年的ModernBERT论文使用2026个基本组件重构了该架构：

| 组件 | 原版BERT（2018） | ModernBERT（2024） |
|-----------|----------------------|-------------------|
| 位置编码 | 学习得到的绝对值 | RoPE |
| 激活函数 | GELU | GeGLU |
| 归一化层 | LayerNorm | 预归一化的RMSNorm |
| 注意力机制 | 全密集连接 | 交替的局部（128）+全局注意力 |
| 上下文长度 | 512 | 8192 |
| 分词器 | WordPiece | BPE |

与2018年的架构不同，ModernBERT专为Flash-Attention设计。在序列长度为8K时，其推理速度比DeBERTa-v3快2–3倍，同时GLUE评测分数也更高。

### 2026年仍需使用编码器的应用场景

| 任务 | 为何编码器优于解码器 |
|------|---------------------------|
| 检索/语义搜索嵌入 | 双向上下文 = 更高的每个标记的嵌入质量 |
| 分类（情感、意图、毒性） | 单次前向传播；无需生成开销 |
| 名词实体识别/标记标注 | 逐位置输出，天然支持双向处理 |
| 零样本蕴含推理（NLI） | 在编码器之上添加分类头 |
| RAG系统的重排器 | 跨编码器评分，速度比LLM重排器快10倍 |

```figure
transformer-residual
```

## 构建它

### 步骤 1：掩码逻辑

请参阅 `code/main.py`。函数 `create_mlm_batch` 接收一个令牌 ID 列表、词汇表大小以及掩码概率作为参数。该函数返回应用了掩码处理的输入 ID 以及标签（仅在被掩码的位置有值，其余位置为 -100 —— 这是 PyTorch 的忽略索引约定）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 步骤 2：在小型语料库上运行 MLM 预测

在包含20个词汇和200个句子的训练集上，训练一个由2层编码器与MLM头组成的模型。由于不使用梯度，因此仅通过前向传播来进行合理性检查。完整的训练过程需要PyTorch框架。

### 步骤 3：比较掩码类型

展示三向规则如何确保模型在存在 `[MASK]` 的情况下仍能正常使用。分别对未遮蔽的句子和已遮蔽的句子进行预测，由于模型在训练过程中见过这两种模式，两种情况的词元分布都应保持合理。

### 步骤 4：微调头部模型

在一个简单的情绪分析数据集上，将 MLM 头替换为分类头。仅对分类头进行训练，而编码器保持冻结状态。这是所有 BERT 应用所遵循的通用模式。

## 使用它

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**嵌入模型是经过微调的 BERT。** 如 `all-MiniLM-L6-v2` 这样的 `sentence-transformers` 模型，是通过对比损失训练得到的 BERT 变体。其编码器结构保持不变，仅有损失函数有所不同。

**跨编码器重排序模型同样属于微调后的 BERT。** 它们基于 `[CLS] 查询词 [SEP] 文档词 [SEP]` 的格式进行成对分类。查询词与文档词之间的双向注意力机制，正是让跨编码器在性能上优于双编码器的关键所在。

**2026 年何时不应选择 BERT。** 任何生成式任务都不适用，因为其编码器缺乏自回归生成标记的合理方式。此外，参数量低于 10 亿的情况也不适合使用 BERT，因为较小的解码器即可以更高的灵活性达到相似的性能水平（如 Phi-3-Mini、Qwen2-1.5B）。

## 发布它

请参阅 `outputs/skill-bert-finetuner.md`。该技能定义了针对新的分类或抽取任务对 BERT 模型进行微调的各个参数（包括骨干网络选择、头部结构规格、训练数据、评估方式以及停止条件）。

## 练习题

1. **简单级。** 运行 `code/main.py`，输出 10,000 个标记的掩码分布情况。确认约有 15% 的标记被选中，且其中约 80% 被替换为 `[MASK]`。
2. **中等级。** 实现整词掩码机制：若某个单词被拆分为多个子词，则将所有子词一起进行掩码处理，或全部不进行掩码处理。在包含 500 句句子的语料库上测试该机制是否能提升 MLM 模型的准确率。
3. **高级别。** 使用公共数据集中的 10,000 句句子训练一个小型（2 层结构，隐藏层维度 d=64）的 BERT 模型。对该模型中的 `[CLS]` 标记进行 SST-2 情感分析任务的微调。在参数设置相同的情况下，将其与仅包含解码器的基线模型进行对比——哪种模型表现更好？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| MLM | “掩码语言建模” | 训练信号：随机将15%的标记替换为`[MASK]`，然后预测原始标记。 |
| Bidirectional | “双向处理” | 编码器注意力机制没有因果屏蔽——每个位置都可以看到其他所有位置的信息。 |
| `[CLS]` | “聚合标记” | 添加在每个序列开头的特殊标记；其最终的嵌入向量被用作句子级别的表示。 |
| `[SEP]` | “分词分隔符” | 用于分隔成对的序列（例如查询/文档、句子A/B）。 |
| NSP | “下一句预测” | BERT的第二个预训练任务；在RoBERTa中已被证明无用，2019年后被移除。 |
| Fine-tuning | “针对任务进行微调” | 保持编码器大部分参数不变，在其顶部训练一个小型头部网络以处理下游任务。 |
| Cross-encoder | “重排器” | 一种将查询和文档作为输入、输出相关性分数的BERT模型。 |
| ModernBERT | “2024版更新版本” | 使用RoPE、RMSNorm、GeGLU以及交替的局部/全局注意力机制重新构建的编码器，支持8K上下文长度。 |

## 延伸阅读

- [Devlin 等人 (2018). BERT：用于语言理解的深度双向变换器预训练](https://arxiv.org/abs/1810.04805) —— 原始论文。
- [Liu 等人 (2019). RoBERTa：一种经过优化以提升稳定性的 BERT 预训练方法](https://arxiv.org/abs/1907.11692) —— 如何正确训练 BERT；可有效解决 NSP 问题。
- [Clark 等人 (2020). ELECTRA：将文本编码器作为判别器而非生成器进行预训练](https://arxiv.org/abs/2003.10555) —— 替换词检测在相同计算资源下性能优于 MLM。
- [Warner 等人 (2024). 更智能、更优秀、更快、更持久：一种现代双向编码器](https://arxiv.org/abs/2412.13663) —— ModernBERT 论文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) —— 标准编码器参考实现。
