# 文本转语音（TTS）——从 Tacotron 到 F5 与 Kokoro

> ASR负责将语音转换为文本；TTS则负责将文本转换为语音。2026年的技术架构由三部分组成：文本 → 令牌，令牌 → Mel频谱，Mel频谱 → 波形。每一部分都配有可运行在笔记本电脑上的默认模型。

**类型：** 构建
**语言：** Python
**先修知识：** 第6阶段 · 02（频谱图与Mel频谱），第5阶段 · 09（Seq2Seq模型），第7阶段 · 05（完整Transformer架构）
**耗时：** 约75分钟

## 问题所在

你有一段文本：“Please remind me to water the plants at 6 pm.”你需要为实时语音助手生成一段时长为3秒的音频片段，要求听起来自然、韵律正确（包含适当的停顿和重音），“plants”一词的元音发音准确，并且在CPU上的处理时间需控制在300毫秒以内。此外，该系统还需能切换不同的语音、处理混合语码的输入（如“remind me at 6 pm, daijoubu?”），并且在涉及人名时不会出错。

现代文本转语音技术流程如下：

1. **文本预处理层**：对文本进行规范化处理（包括日期、数字、电子邮件等），将其转换为音素或子词标记，同时预测韵律特征。
2. **声学模型层**：将文本转换为梅尔频谱图。常用的模型包括Tacotron 2（2017年）、FastSpeech 2（2020年）、VITS（2021年）、F5-TTS（2024年）以及Kokoro（2024年）。
3. **声码器层**：将梅尔频谱图转换为波形。常用的技术包括WaveNet（2016年）、WaveRNN、HiFi-GAN（2020年）、BigVGAN（2022年），以及2024年及以后出现的神经编解码器声码器。

到2026年，随着端到端的扩散模型和流匹配模型的出现，声学模型与声码器之间的界限将逐渐模糊。不过，在进行调试时，人们依然习惯于采用三层结构的思维模型。

## 概念概述

![Tacotron、FastSpeech、VITS与F5/Kokoro的并列对比图](../assets/tts.svg)

**Tacotron 2（2017年）。** 采用Seq2seq架构：字符嵌入层 → 双向LSTM编码器 → 位置敏感注意力机制 → 自回归LSTM解码器，最终生成梅尔频谱帧。由于采用自回归方式，处理长文本时速度较慢且语音质量不稳定。目前仍被用作基准模型。

**FastSpeech 2（2020年）。** 非自回归架构。通过时长预测器确定每个音素对应的梅尔频谱帧数量。仅需一次遍历即可完成生成，速度是Tacotron的10倍。虽因单调对齐问题导致语音自然度有所下降，但已广泛应用于各类场景。

**VITS（2021年）。** 采用变分推断技术，端到端联合训练编码器、基于流的时长预测模块以及HiFi-GAN声码器。该模型能够生成高质量语音，且仅需一个模型即可实现功能。在2022年至2024年间成为最主流的开源文本转语音技术。其衍生版本包括：YourTTS（支持多说话人零样本克隆）、XTTS v2（2024年发布，由Coqui开发）。

**F5-TTS（2024年）。** 基于流匹配的扩散变换模型。具备自然的语调表现能力，仅需5秒的参考音频即可实现零样本语音克隆。在2026年的开源文本转语音性能排行榜中位居榜首，模型参数量为3.35亿。

**Kokoro（2024年）。** 模型规模较小（8200万参数），可在CPU上运行，是实时应用场景下表现最佳的英语文本转语音模型。仅支持封闭词汇表的英语内容，采用apache-2.0许可协议。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 这些均为商业领域的顶尖文本转语音技术。其中，ElevenLabs v2.5具备情绪标签功能（如“[whispered]”、“[laughing]”）以及多种角色声音选项，在2026年的有声书制作领域占据主导地位。

### 语音编码器的演进

| 发展时期 | 语音编码器 | 延迟 | 音质 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 仅离线模式 | 发布时处于行业最先进水平 |
| 2018 | WaveRNN | 接近实时 | 良好 |
| 2020 | HiFi-GAN | 实时速度的100倍 | 几乎接近人类发音 |
| 2022 | BigVGAN | 实时速度的50倍 | 能跨不同说话者及语言进行泛化 |
| 2024 | SNAC、DAC（神经编解码器） | 与AR模型集成 | 使用离散令牌，具有较高的比特效率 |

到2026年，大多数“文本转语音”模型均为从文本到波形的端到端架构；梅尔频谱图仅作为其内部表示形式使用。

### 评估

- **MOS（平均主观评分）**。采用1–5分制，基于众包数据。仍是行业黄金标准，但处理速度极慢。
- **CMOS（对比式MOS）**。用于评估A模型相对于B模型的偏好程度，每条标注的置信区间更窄。
- **UTMOS、DNSMOS**。无需参考数据的神经网络MOS预测模型，常用于排行榜生成。
- **通过ASR计算的CER（字符错误率）**。将文本转语音的输出结果送入Whisper模型，再计算其与原始文本的CER值，以此作为可懂度的替代指标。
- **SECS（说话人嵌入余弦相似度）**。用于衡量声音克隆的质量。

LibriTTS测试集上的2026年数据如下：

| 模型 | UTMOS | 通过Whisper计算的CER | 文件大小 |
|-------|-------|-------------------|----------|
| 真实参考值 | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

## 构建它

### 步骤 1：对输入进行音素化处理

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是实现通用连接的桥梁。切勿将质量低于 VITS 级别的原始文本输入至任何系统。

### 步骤 2：运行 Kokoro（2026 年款 CPU 默认配置）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

离线运行，单文件格式，包含8200万个参数。

### 步骤 3：使用语音克隆功能运行 F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

上传一段5秒的参考视频片段及其字幕；F5键可复制其韵律与音色特征。

### 第 4 步：从零实现 HiFi-GAN 语音编码器

内容过于庞大，无法放入教程脚本中，但其结构如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

训练方式：对抗训练（基于短时间窗口的判别器）+ Mel频谱图重建损失 + 特征匹配损失。该方案已实现商品化——可直接使用 `hifi-gan` 仓库或 nvidia-NeMo 提供的预训练检查点。

### 步骤 5：完整流程（伪代码）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 使用它

2026年的技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 实时英语语音助手 | Kokoro（CPU）或 XTTS v2（GPU） |
| 基于5秒参考音频的语音克隆 | F5-TTS |
| 商业角色语音 | ElevenLabs v2.5 |
| 有声书旁白 | ElevenLabs v2.5 或 XTTS v2 + 微调 |
| 资源匮乏的语言 | 使用5–20小时的目标语言数据训练VITS |
| 表情/情绪标签生成 | ElevenLabs v2.5 或经过微调的StyleTTS 2 |

截至2026年的开源领域领先者：**在质量方面首选F5-TTS，追求效率则选Kokoro**。除非你是研究历史的人，否则无需考虑Tacotron。

## 常见陷阱

- **无文本规范化功能。** “Dr. Smith” 应该被读作 “Doctor” 还是 “Drive”？“2026” 应该被读作 “twenty twenty six” 还是 “two zero two six”？请在语音合成器之前进行规范化处理。
- **词汇表外的专有名词。** “Ghumare” 应该转换为 “ghyu-mair” 吗？请为未知的标记提供一个备用的字形到音素映射模型。
- **数值截断问题。** 语音编码器的输出很少出现截断现象，但在推理过程中如果梅尔刻度不匹配，数值可能会超出 ±1.0 的范围。务必始终使用 `np.clip(wav, -1, 1)` 进行处理。
- **采样率不匹配问题。** Kokoro 的输出采样率为 24 kHz，而下游处理流程期望的采样率为 16 kHz → 需要对音频进行重采样，否则会出现混叠现象。

## 发布它

将文件保存为 `outputs/skill-tts-designer.md`。针对指定的语音、延迟要求及目标语言，设计一个文本转语音（TTS）处理流程。

## 练习题

1. **简单。** 运行 `code/main.py`。该脚本会使用一个简单的词汇表构建音素词典，估算每个音素的时长，并输出一个模拟的“mel”调度表。
2. **中等难度。** 安装 Kokoro，使用语音 `af_bella` 和 `am_adam` 合成相同的句子。比较音频的时长以及主观听感质量。
3. **高难度。** 录制一段5秒长的个人参考音频片段。使用 F5-TTS 对其进行克隆。需报告参考音频与克隆输出之间的时间差（以秒为单位）。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|-----------------|----------|
| 音素 | 声音单位 | 抽象的声学类别；英语中共有39个（ARPABet标准）。 |
| 持续时间预测器 | 每个音素的持续时间 | 非AR模型输出；表示每个音素对应的整数帧数。 |
| 语音编码器 | Mel频谱 → 波形 | 将Mel频谱特征映射为原始音频样本的神经网络。 |
| HiFi-GAN | 标准语音编码器 | 基于GAN技术；2020年至2024年间最为常用。 |
| MOS | 主观质量评分 | 由人类评估员给出的1至5分的平均意见得分。 |
| SECS | 语音克隆指标 | 目标说话者嵌入向量与输出说话者嵌入向量之间的余弦相似度。 |
| F5-TTS | 2024年的开源最先进技术 | 基于流匹配扩散模型的零样本语音克隆技术。 |
| Kokoro | CPU环境下的英语语音处理领先模型 | 参数量为8200万的模型，采用Apache 2.0许可协议。 |

## 延伸阅读

- [Shen 等人 (2017). Tacotron 2](https://arxiv.org/abs/1712.05884) — 序列到序列模型的基准模型。  
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103) — 基于流式架构的端到端模型。  
- [Chen 等人 (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 当前最先进的开源文本转语音模型。  
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646) — 2026 年仍在使用的声码器。  
- [HuggingFace 上的 Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) — 2024 年适用于 CPU 的英语文本转语音模型。
