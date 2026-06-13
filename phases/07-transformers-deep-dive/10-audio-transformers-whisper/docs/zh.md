# 音频变换器——Whisper 架构

> 音频本质上是随时间变化的频率图像。Whisper 是一种能够输入梅尔频谱图并生成语音输出的 ViT 模型。

**类型：** 学习
**语言：** Python
**先修知识：** 第 7 阶段 · 05（完整 Transformer）、第 7 阶段 · 08（编码器-解码器）、第 7 阶段 · 09（ViT）
**时长：** 约 45 分钟

## 问题所在

在 Whisper（OpenAI，Radford 等人，2022 年）出现之前，最先进的自动语音识别（ASR）技术为 wav2vec 2.0 和 HuBERT——这类技术依赖自监督特征提取器，并需对后续处理模块进行微调。虽然其性能优异，但数据管道成本高昂，且对特定领域的数据敏感度极高。多语言语音识别则需要为每个语系单独构建模型。

Whisper 则采取了三项创新策略：

1. **利用海量数据进行训练**：从互联网上收集了涵盖 97 种语言的 680,000 小时的弱标注音频数据，既没有结构完善的学术语料库，也没有音素标注。
2. **单模型多任务处理**：通过任务标记，让同一个解码器同时负责文本转写、翻译、语音活动检测、语言识别以及时间戳生成等任务。
3. **采用标准的编码器-解码器 Transformer 结构**：编码器输入对数梅尔频谱图，解码器则通过自回归方式生成文本标记；该模型不依赖声码器、CTC 算法或 HMM 模型。

其结果便是 Whisper large-v3 在面对不同口音、噪声以及完全缺乏标注数据的语言时仍能保持稳定性能。截至 2026 年，它已成为所有开源语音助手以及大多数商业语音助手的默认语音处理前端。

## 概念概述

![Whisper 处理流程：音频 → Mel 频谱 → 编码器 → 解码器 → 文本](../assets/whisper.svg)

### 步骤 1 — 重采样 + 窗口处理

音频采样率为 16 kHz。将音频截取或填充至 30 秒长度。计算对数梅尔频谱图：使用 80 个梅尔频段，步长为 10 毫秒 → 约 3,000 帧 × 80 个特征值。这就是 Whisper 所接收的“输入数据”。

### 步骤 2 — 卷积词干提取

两个核大小为3、步长为2的Conv1D层将3,000帧减少到1,500帧。在不会大幅增加参数数量的情况下，即可将序列长度减半。

### 步骤 3 —— 编码器

一个包含24层（适用于大型模型）的Transformer编码器，处理长度为1,500个时间步的数据。采用正弦位置编码、自注意力机制以及GELU函数作为前馈网络。最终可生成1,500 × 1,280个隐藏状态。

### 第 4 步 —— 解码器

一个包含24层解码器的Transformer模型。它基于比GPT-2词汇表更大的BPE词汇表进行自回归生成，该词汇表还包含一些针对音频的特殊标记。

### 步骤 5 — 任务令牌

解码器提示以控制令牌开头，这些令牌用于告知模型执行的具体操作：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

该模型是按照此规范进行训练的。您可以通过前缀来控制任务类型。这相当于2026年版本的指令微调技术，但应用于语音处理领域。

### 步骤 6 — 输出

带对数概率阈值的束搜索（宽度为5）。当不存在`<|notimestamps|>`标记时，每0.02秒的音频时长会预测一次时间戳。

### Whisper 尺寸

| 模型 | 参数量 | 层数 | d_model | 输出头数量 | VRAM（fp16） |
|-------|--------|------|---------|-----------|-------------|
| Tiny | 39M | 4 | 384 | 6 | 约 1 GB |
| Base | 74M | 6 | 512 | 8 | 约 1 GB |
| Small | 244M | 12 | 768 | 12 | 约 2 GB |
| Medium | 769M | 24 | 1024 | 16 | 约 5 GB |
| Large | 1550M | 32 | 1280 | 20 | 约 10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | 约 10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | 约 6 GB（4层解码器） |

Large-v3-turbo（2024版）将解码器的层数从32层缩减至4层。在WER值几乎不受影响的情况下，解码速度提升了8倍。正是这一解码速度的提升，使得Whisper-turbo成为2026年实时语音智能体的默认选择。

### Whisper 无法实现的功能

- 不支持语音分帧（即无法识别谁在说话），如需此功能需结合 pyannote 使用。
- 本身不支持实时流式处理——时间窗口固定为 30 秒。现代封装库（如 `faster-whisper`、`WhisperX`）通过 VAD 和重叠技术实现了流式处理功能。
- 若无外部分块处理，无法获取超过 30 秒的长期上下文。但在实际应用中表现良好，因为人类语音在转录时很少需要长距离上下文。

### 2026年行业格局

| 任务 | 模型 | 备注 |
|------|-------|------|
| 英语 ASR | Whisper-turbo、Moonshine | Moonshine 在边缘设备上的处理速度是前者的 4 倍 |
| 多语言 ASR | Whisper-large-v3 | 支持 97 种语言 |
| 流式 ASR | faster-whisper + VAD | 可实现 150 毫秒的延迟目标 |
| TTS | Piper、XTTS-v2、Kokoro | 采用编解码器架构，但结构类似 Whisper |
| 音频与语言处理 | AudioLM、SeamlessM4T | 在同一个 Transformer 中同时处理文本令牌和音频令牌 |

## 构建它

请参阅 `code/main.py`。我们不会训练 Whisper 模型——而是构建对数梅尔频谱图处理流程以及任务令牌提示格式化器。这些才是你在实际生产环境中需要操作的组件。

### 步骤 1：合成音频

生成一个频率为 440 Hz、持续时间为 1 秒的正弦波，采样率为 16 kHz，共包含 16,000 个样本。

### 步骤 2：对数梅尔频谱图（简化版）

完整的梅尔频谱图需要使用 FFT。我们提供了一种简化的帧结构处理方式以及逐帧能量计算版本，无需依赖 `librosa` 即可展示相关处理流程：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

帧间隔为 25 毫秒，跳变间隔为 10 毫秒。该设置与 Whisper 的窗口化机制一致。出于教学目的，每帧能量值用于替代 mel 分箱。

### 步骤 3：将时长补足至 30 秒

Whisper 始终以 30 秒为粒度处理数据。需将频谱图填充（或截取）至 3,000 帧。

### 步骤 4：构建提示词标记

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

这就是完整的任务控制界面。由4个标记组成的前缀。

## 使用它

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快，兼容 OpenAI：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026年何时选择Whisper：**

- 仅需一个模型即可实现多语言语音识别。
- 能够高效转录嘈杂且来源多样的音频。
- 用于研究或原型开发的语音识别——最快的起步方案。

**何时应选择其他方案：**

- 需要在边缘设备上实现超低延迟流式传输——在同等质量下，Moonshine优于Whisper。
- 需要实时对话型AI且响应时间需低于200毫秒——应使用专用流式语音识别技术。
- 需要进行说话人分离——Whisper不支持此功能，需额外集成pyannote。

## 发布它

请参阅 `outputs/skill-asr-configurator.md`。该技能会为新的语音应用选择 ASR 模型、解码参数以及预处理流程。

## 练习题

1. **简单。** 运行 `code/main.py`。确认在 16 kHz 频率、10 ms 跳变间隔下，1 秒长的信号大约包含 100 帧数据；30 秒长则约为 3,000 帧。
2. **中等难度。** 使用 `numpy.fft` 函数生成完整的对数梅尔频谱图。验证 80 个梅尔频段的数量是否与 `librosa.feature.melspectrogram(n_mels=80)` 的输出在数值误差范围内一致。
3. **高难度。** 实现流式推理功能：将音频分割为长度为 10 秒、重叠部分为 2 秒的块，对每个块使用 Whisper 进行处理，并合并生成的文本转录结果。以一段 5 分钟长的播客样本为测试数据，测量其词错误率与单次处理方式的差异。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Mel频谱图 | “音频图像” | 一种二维表示形式：一条轴为频率区间，另一条轴为时间帧；每个单元格的数值为对数缩放后的能量值。 |
| 对数Mel频谱 | “Whisper所使用的格式” | 经过对数处理后的Mel频谱图；能够近似模拟人类对音量的感知。 |
| 时间帧 | “一个时间切片” | 长度为25毫秒的样本窗口，以10毫秒的步长相互重叠。 |
| 任务标记 | “语音处理的提示前缀” | 解码器提示中的特殊标记，例如`<\|transcribe\|>` / `<\|translate\|>`。 |
| 语音活动检测（VAD） | “识别语音片段” | 一种用于在自动语音识别之前过滤静音的机制；可大幅降低处理成本。 |
| CTC | “连接主义时间分类算法” | 一种无需对齐即可进行训练的经典自动语音识别损失函数；Whisper并未采用该算法。 |
| Whisper-turbo | “小型解码器搭配完整编码器” | 采用large-v3编码器加上4层解码器结构；解码速度提升8倍。 |
| Faster-whisper | “正式发布的封装版本” | 基于CTranslate2重新实现的版本；采用int8量化技术；其运行速度是OpenAI参考版本的4倍。 |

## 延伸阅读

- [Radford 等人（2022）。基于大规模弱监督的鲁棒语音识别](https://arxiv.org/abs/2212.04356) — Whisper 论文。
- [OpenAI Whisper 代码仓库](https://github.com/openai/whisper) — 参考代码及模型权重。阅读 `whisper/model.py` 即可查看约 400 行代码中自上而下的 Conv1D 骨干结构、编码器与解码器实现。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 第 5–6 步中描述的束搜索及任务令牌逻辑位于此处；代码共 500 行，可完整阅读。
- [Baevski 等人（2020）。wav2vec 2.0：一种用于语音表示的自监督学习框架](https://arxiv.org/abs/2006.11477) — 其前版本；在某些场景下仍具备当前最优性能。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 用于生产环境的封装工具，速度比参考实现快 4 倍。
- [Jia 等人（2024）。Moonshine：适用于实时转录与语音指令的语音识别系统](https://arxiv.org/abs/2410.15608) — 2024 年推出的适合边缘设备的 ASR 系统，架构类似 Whisper 但规模更小。
- [HuggingFace 博客 — “使用 🤗 Transformers 对 Whisper 进行多语言 ASR 微调”](https://huggingface.co/blog/fine-tune-whisper) — 包含梅尔频谱图预处理器及令牌时间戳处理在内的标准微调流程。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整实现代码（包含编码器、解码器、交叉注意力机制及生成模块），其架构与课程中的示意图一致。
