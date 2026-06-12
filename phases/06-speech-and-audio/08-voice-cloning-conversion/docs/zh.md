# 语音克隆与语音转换

> 语音克隆技术能够以他人的声音朗读您的文本；而语音转换则能在保留原话内容的前提下，将您自己的声音转换为他人之声。这两种技术都基于同一核心原理：将说话者身份与语音内容分离。

**类型：** 构建
**语言：** Python
**先修课程：** 第6阶段 · 06（说话人识别）、第6阶段 · 07（文本转语音）
**耗时：** 约75分钟

## 问题所在

截至2026年，仅需5秒的音频片段，借助消费级GPU即可生成任何人的高质量语音克隆。ElevenLabs、F5-TTS、OpenVoice v2以及VoiceBox均支持零样本或少样本克隆功能。这项技术既是福音（如无障碍文本转语音、配音、辅助语音系统），也可能是利器（用于诈骗电话、政治深度伪造及知识产权窃取）。

与之密切相关的两项任务如下：

- **语音克隆（文本转语音方向）：** 输入文本 + 5秒的参考语音 → 生成对应语音的音频。
- **语音转换（语音处理方向）：** 输入源音频（如A说X的内容）+ B的参考语音 → 生成B说X的音频。

这两种技术均会将波形分解为“内容”、“说话人”和“韵律”三个要素，然后从不同来源提取相应元素并重新组合。

在2026年产品上线时必须遵守的关键约束是：**根据欧盟《人工智能法案》（将于2026年8月正式实施）及加利福尼亚州AB 2905法案（2025年起生效），在产品中加入水印并设置同意机制在法律上是强制性的**。因此，您的处理流程必须嵌入不可听的水印，并拒绝生成未经授权的语音克隆内容。

## 概念概述

![语音克隆与转换：分解、替换说话人、重新组合](../assets/voice-cloning.svg)

**零样本克隆。** 将5秒长的音频片段输入已在数千名不同说话人数据上训练过的模型。说话人编码器会将该片段映射为对应的说话人嵌入向量；TTS解码器则结合该嵌入向量与文本信息进行生成。

应用案例：F5-TTS（2024年）、YourTTS（2022年）、XTTS v2（2024年）、OpenVoice v2（2024年）。

**小样本微调。** 录制5至30分钟的目标语音。使用LoRA技术对基础模型进行约1小时的微调。音频质量可从“尚可”提升至“几乎无法区分”。Coqui与ElevenLabs均支持此方法；社区中也常将其与F5-TTS结合使用。

**语音转换（VC）。** 主要分为两大类：

- **识别-合成法。** 运行类似ASR的模型提取内容表征（例如软音素后验概率、PPG值），随后利用目标说话人嵌入向量进行重新合成。该方法对语言和口音具有较强鲁棒性。应用案例：KNN-VC（2023年）、Diff-HierVC（2023年）。
- **解耦法。** 训练一个自编码器，在瓶颈层的潜在空间中分离内容、说话人特征及韵律信息。在推理阶段替换说话人嵌入向量。虽然质量略低，但处理速度更快。应用案例：AutoVC（2019年）、VITS-VC系列模型。

**基于神经编解码器的克隆技术（2024年及以后）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox —— 这些技术将音频视为来自SoundStream / EnCodec的离散标记，然后在这些编码标记上训练大型自回归模型或流匹配模型。在短文本输入条件下，其生成质量可与ElevenLabs相媲美。

### 伦理部分并非事后添加的内容

**水印技术。** PerTh（Perth）与 SilentCipher（2024）能够在音频中不可察觉地嵌入约16至32位的标识符。该标识符能够经受重新编码、流式传输以及常见编辑操作而保持完整。属于可直接投入生产的开源方案。

**授权机制。** 每个克隆生成的输出都必须附带可验证的授权记录。“我，Rohit，在2026-04-22授权将此语音用于X用途。”此类授权记录需存储在防篡改日志中。

**检测技术。** AASIST、RawNet2以及Wav2Vec2-AASIST均作为检测工具提供。ASVspoof 2025竞赛公布了最先进的检测技术在面对ElevenLabs、VALL-E 2及Bark输出时的错误率，范围在0.8%至2.3%之间。

### 数字（2026）

| 模型 | 是否零样本？ | SECS（目标语音） | WER（智能评估） | 参数量 |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | 是 | 0.72 | 2.1% | 335M |
| XTTS v2 | 是 | 0.65 | 3.5% | 470M |
| OpenVoice v2 | 是 | 0.70 | 2.8% | 220M |
| VALL-E 2 | 是 | 0.77 | 2.4% | 370M |
| VoiceBox | 是 | 0.78 | 2.1% | 330M |

对于大多数听众而言，若 SECS 值大于 0.70，则该语音与目标语音几乎无法区分。

## 构建它

### 步骤 1：通过识别-合成进行分解（在 main.py 中提供仅代码的演示）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上较为简单；实际实现的工作量主要集中在 `tts_model` 和说话人编码器中。

### 步骤 2：使用 F5-TTS 进行零样本克隆

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

参考转录文本必须与音频内容完全一致；任何差异都会导致对齐失败。

### 步骤 3：使用 KNN-VC 进行语音转换

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 会运行 WavLM 以提取源语音集和目标语音集中的每帧嵌入向量，随后用该集合中距离最近的嵌入向量替换每一帧源语音。这是一种非参数方法，仅需一分钟的目标语音即可使用。

### 步骤 4：嵌入水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

约 32 位有效载荷，在 MP3 重新编码及轻微噪声之后仍可被检测到。

### 第 5 步：同意机制

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 使用它

2026年技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 5秒内零样本克隆，开源版本 | F5-TTS 或 OpenVoice v2 |
| 商业级生产环境克隆 | ElevenLabs Instant Voice Clone v2.5 |
| 语音转换（重写） | KNN-VC 或 Diff-HierVC |
| 多说话人微调 | StyleTTS 2 + 说话人适配器 |
| 跨语言克隆 | XTTS v2 或 VALL-E X |
| 深度伪造检测 | Wav2Vec2-AASIST |

## 常见陷阱

- **参考文本对齐错误。** F5-TTS 及同类模型要求参考文本与参考音频完全一致，包括标点符号。
- **混响严重的参考音频。** 回声会破坏生成的克隆语音。请使用无回声环境并靠近麦克风进行录制。
- **情感不匹配。** 若以“欢快”的情绪作为训练参考，生成的所有克隆语音都会带有欢快的风格。需根据实际应用场景匹配相应的参考情绪。
- **语言泄露问题。** 即使克隆了英语使用者的语音，再要求模型说法语，往往仍会保留口音；此时应使用跨语言模型（如 XTTS、VALL-E X）。
- **无水印。** 自 2026 年 8 月起，在欧盟地区若产品未添加水印将无法合法销售。

## 发布它

将文件保存为 `outputs/skill-voice-cloner.md`。设计一个包含同意机制、水印添加以及质量控制目标的克隆或转换流程。

## 练习题

1. **简单。** 运行 `code/main.py`。通过计算交换前后两个“说话人”向量的余弦值，演示说话人嵌入的交换过程。
2. **中等难度。** 使用 OpenVoice v2 克隆自己的声音。测量参考语音与克隆语音之间的 SECS 值，并通过 Whisper 工具检测 CER 值。
3. **高难度。** 对 20 个克隆语音应用 SilentCipher 水印，将其进行 128 kbps MP3 的编码和解码处理，然后检测水印载荷的内容，并报告比特准确率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 零样本克隆 | 5秒即可完成 | 使用预训练模型与说话人嵌入，无需额外训练。 |
| PPG | 音素后验图 | 作为与语言无关的内容表示，使用逐帧ASR后验值。 |
| KNN-VC | 最近邻转换 | 用目标池中最接近的帧替换每个源帧。 |
| 神经编解码器TTS | VALL-E风格 | 基于EnCodec/SoundStream标记的AR模型。 |
| 水印 | 听不见的签名 | 嵌入音频中的比特，可经受重新编码。 |
| SECS | 克隆保真度 | 目标说话人嵌入与克隆说话人嵌入之间的余弦值。 |
| AASIST | 深度伪造检测器 | 反欺骗模型，用于检测合成语音。 |

## 延伸阅读

- [Chen 等人 (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — 开源的当前最先进零样本克隆技术。  
- [Baevski 等人 / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 以及 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — 基于神经编解码器的文本转语音技术。  
- [Qian 等人 (2019). AutoVC](https://arxiv.org/abs/1905.05879) — 基于解耦技术的声音转换方法。  
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — 基于检索的声音转换技术。  
- [SilentCipher (2024) — 音频水印技术](https://github.com/sony/silentcipher) — 可用于实际生产的 32 位音频水印方案。  
- [ASVspoof 2025 竞赛结果](https://www.asvspoof.org/) — 检测器与合成器之间的技术竞争，数据更新至 2026 年。
