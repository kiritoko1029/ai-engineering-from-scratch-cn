# Whisper — 架构与微调

> Whisper 是一种基于 30 秒时间窗口的Transformer编解码器，通过在 68 万小时的多种语言弱监督音频-文本对数据上训练而成。同一架构可支持多种任务，在 99 种语言中均表现出良好的性能。它是 2026 年的参考级 ASR 系统。

**类型：** 构建
**编程语言：** Python
**先修课程：** 第 6 阶段 · 04（ASR）、第 5 阶段 · 10（注意力机制）、第 7 阶段 · 05（完整 Transformer 模型）
**所需时间：** 约 75 分钟

## 问题所在

由 OpenAI 于 2022 年 9 月发布的 Whisper 是首个作为通用产品推出的 ASR 模型：只需粘贴音频即可获取文本，支持 99 种语言，具备较强的抗噪能力，且可在笔记本电脑上运行。到 2024 年，OpenAI 又推出了 Large-v3 和 Turbo 版本；到了 2026 年，Whisper 已成为从播客转录、语音助手到 YouTube 字幕等所有应用中的默认基准模型。

不过，Whisper 并非可以永远当作黑箱来使用的处理流程。领域差异会严重影响其性能——技术术语、说话者的口音、专有名词、短片段音频以及静默时段都会对其造成挑战。因此你需要了解：

1. 它的内部架构究竟是怎样的。
2. 如何正确地以分块、流式或长格式提供音频数据。
3. 何时以及如何进行微调。

## 概念概述

![Whisper 编码器-解码器、任务类型、分块推理与微调](../assets/whisper.svg)

**架构。** 标准的 Transformer 编码器-解码器结构。

- 输入：30 秒长的对数梅尔频谱图，共 80 个梅尔频率点，采样间隔为 10 毫秒 → 共 3000 帧。较短的片段会进行零填充，较长的片段则会被分块处理。
- 编码器：卷积下采样（步长为 2）+ `N` 个 Transformer 块。对于 Large-v3 版本，其结构包含 32 层，每层维度为 1280，拥有 20 个注意力头。
- 解码器：`N` 个 Transformer 块，采用因果自注意力机制以及针对编码器输出的交叉注意力机制。解码器的规模与编码器相同。
- 输出：基于包含 51,865 个词条的词汇表生成的 BPE 令牌。

Large-v3 版本的参数量为 15.5 亿。Turbo 版本则采用了仅含 4 层的解码器结构（原版本为 32 层），在将延迟降低 8 倍的同时，词错误率仍保持在 1% 以下。

**提示语格式。** Whisper 是一个多任务模型，其运行方式由解码器提示语中的特殊令牌来控制：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — 语言标签；用于强制指定翻译或转录模式。
- `<|transcribe|>` 或 `<|translate|>` — 将任意语言输入的英文内容进行翻译，或保持原文不变。
- `<|notimestamps|>` — 跳过单词级别的时间戳（提升速度）。

提示词决定了模型能够执行的任务类型。将 `<|en|>` 更改为 `<|fr|>` 即可使其执行法语转录功能。

**30秒的时间窗口。** 所有处理均限制在30秒内完成。较长的音频片段需要拆分处理，较短的片段则需补充内容。系统不支持直接流式处理该时间窗口——这也是 WhisperX、Whisper-Streaming 以及 faster-whisper 等工具诞生的原因。

**对数梅尔归一化。** 公式为 `(log_mel - mean) / std`，其中统计参数取自 Whisper 自身的训练语料库。必须使用 Whisper 提供的预处理函数（`whisper.audio.log_mel_spectrogram`），而不能使用 `librosa.feature.melspectrogram`。

### 2026年的变体版本

| 变体 | 参数量 | 延迟（A100） | WER（LibriSpeech-clean） |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 实时 1× | 5.4% |
| Base | 74M | 实时 1× | 4.1% |
| Small | 244M | 实时 1× | 3.0% |
| Medium | 769M | 实时 1× | 2.7% |
| Large-v3 | 1.55B | 流式处理 2× | 1.8% |
| Large-v3-turbo | 809M | 流式处理 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | 流式处理 | 2.0% |

### 微调

2026年的标准工作流程：

1. 收集10至100小时的目标领域音频，并附带对应的文字转录。
2. 使用`transformers.Seq2SeqTrainer`，并配置`generate_with_loss`回调函数进行训练。
3. 参数高效优化：在注意力层的`q_proj`、`k_proj`和`v_proj`上应用LoRA技术，可使GPU内存占用降低4倍，同时保持WER值在0.3以下。
4. 若可用音频时长不足10小时，则冻结编码器部分，仅对解码器进行微调。
5. 始终使用Whisper自带的分词器和提示格式，严禁更换其他分词器。

社区实验结果：使用20小时的医疗语音数据对Medium模型进行微调后，其在医疗领域的WER值可从12%降至4.5%。使用4小时的冰岛语语音数据对Turbo模型进行微调后，其WER值可从18%降至6%。

## 构建它

### 步骤 1：直接运行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

务必覆盖的关键默认值：`temperature=0.0`（采样概率的默认范围为 0.0 → 0.2 → 0.4……，遵循回退序列），`condition_on_previous_text=False`（用于防止级联幻觉问题），以及 `no_speech_threshold=0.6`（用于静音检测）。

### 步骤 2：长文本分块处理

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX新增了以下功能：(1)基于Silero的VAD门控机制，(2)通过wav2vec 2.0实现的词级对齐，以及(3)使用`pyannote.audio`进行的对话分割。它是2026年用于生产环境转录的理想工具。

### 步骤 3：使用 LoRA 进行微调

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

接着是标准的训练器循环。每 1000 步保存一次检查点，并使用 WER 指标对保留的测试集进行评估。

### 步骤 4：检查每一层学到了什么

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

使用热图进行可视化展示——您将看到解码器步骤遍历编码器帧时的对角线排列。该对角线即 Whisper 所定义的词时间戳。

## 使用它

2026年的技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 普通英语，离线环境 | 通过 `whisperx` 调用 Large-v3-turbo |
| 移动端/边缘设备 | 量化后的 Whisper-Tiny（int8）或 Moonshine |
| 多语言长文本处理 | 通过 `whisperx` 调用 Large-v3 并结合语音分段功能 |
| 资源匮乏的语言 | 使用 LoRA 对 Medium 或 Turbo 模型进行微调 |
| 流式处理（2秒延迟） | Whisper-Streaming 或 Parakeet-TDT |
| 单词级时间戳生成 | WhisperX（通过 wav2vec 2.0 强制对齐） |

`faster-whisper`（基于 CTranslate2 后端）是2026年最快的 CPU+GPU 推理引擎——在输出质量相同的情况下，其速度是普通版本的4倍。

## 2026年仍会存在的缺陷与隐患

- **静音时的幻觉文本问题。** 基于字幕训练的 Whisper 会生成“感谢观看！”、“订阅！”以及歌曲歌词等内容。在调用相关功能前务必先进行 VAD 检测。
- **`condition_on_previous_text` 的级联效应。** 一次幻觉内容会影响后续的时间窗口。除非需要保证各片段之间的连贯性，否则请将该参数设置为 `False`。
- **短片段的填充问题。** 将 2 秒长的片段填充至 30 秒后，其在尾部的静音区域可能会出现幻觉内容。此时应使用 `pad=False` 或再次进行 VAD 检测。
- **错误的梅尔频谱统计值问题。** 若使用 librosa 的梅尔频谱函数而非 Whisper 自带的函数，会导致输出近乎随机。请使用 `whisper.audio.log_mel_spectrogram`。

## 发布它

将文件保存为 `outputs/skill-whisper-tuner.md`。针对特定领域设计 Whisper 的微调或推理流程。

## 练习题

1. **简单级。** 运行 `code/main.py`。该脚本会对 Whisper 风格的提示词进行分词处理，计算解码后的形状预算，并输出一段 10 分钟视频的切片计划。
2. **中等级。** 安装 `faster-whisper`，对一段 10 分钟的播客进行转录，将其词错误率与人工转写的结果进行对比。尝试使用 `language="auto"` 自动识别语言，以及强制指定 `language="en"` 的情况。
3. **高级别。** 使用 HF 的 `datasets` 库选择一种 Whisper 处理困难的语言（例如乌尔都语），通过 LoRA 对中等级模型进行 2 个时代的微调训练，耗时约 2 小时，并报告词错误率的变化幅度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 30秒时间窗口 | Whisper的限制 | 硬性的输入长度上限；需将更长的音频分割成多个片段。 |
| SOT | 转录起始标记 | `<\|startoftranscript\|>`用于标识解码器提示的开始位置。 |
| 时间戳令牌 | 时间对齐机制 | 在51k词汇量的词典中，每0.02秒的间隔都对应一个特殊令牌。 |
| Turbo | 快速版本 | 拥有4层解码器结构，处理速度提升8倍，WER下降幅度小于1%。 |
| WhisperX | 长文本处理框架 | 结合了VAD、Whisper、wav2vec对齐技术以及对话分段功能。 |
| LoRA微调 | 高效训练方法 | 通过向注意力机制中添加低秩适配器来实现高效训练，仅需训练约0.3%的参数。 |
| 幻觉现象 | 沉默中的错误 | Whisper能够在噪声或静音环境下生成流畅的英文内容。 |

## 延伸阅读

- [Radford 等人 (2022). Whisper 论文](https://arxiv.org/abs/2212.04356) —— 原始架构与训练方案。
- [OpenAI (2024). Whisper Large-v3-turbo 版本发布](https://github.com/openai/whisper/discussions/2363) —— 4 层解码器，速度提升 8 倍。
- [Bain 等人 (2023). WhisperX](https://arxiv.org/abs/2303.00747) —— 支持长文本、词对齐及对话记录功能。
- [Systran —— faster-whisper 仓库](https://github.com/SYSTRAN/faster-whisper) —— 基于 CTranslate2，速度提升 4 倍。
- [HuggingFace —— Whisper 微调教程](https://huggingface.co/blog/fine-tune-whisper) —— 关于 LoRA / 完整微调的标准化操作指南。
