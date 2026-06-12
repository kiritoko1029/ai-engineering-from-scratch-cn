# 音频评估——WER、MOS、UTMOS、MMAU、FAD以及开放排行榜

> 若无法衡量，便无法交付。本课将明确各项音频任务的2026年评估指标：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、ASR往返处理的WER）、音频与语言相关任务（MMAU、LongAudioBench）、音乐相关任务（FAD、CLAP）以及说话人识别相关任务（EER）。此外还会介绍用于对比的排行榜。

**类型：** 学习
**编程语言：** Python
**先修课程：** 第6阶段 · 04、06、07、09、10；第2阶段 · 09（模型评估）
**时长：** 约60分钟

## 问题所在

每个音频任务都包含多项指标，每项用于衡量不同的维度。若选用错误的指标，就会导致模型在控制台显示效果极佳，但在实际生产环境中表现糟糕。2026年的标准指标列表如下：

| 任务 | 主要指标 | 次要指标 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · 首个词元延迟 |
| TTS | MOS / UTMOS | SECS · 基于ASR的往返WER · CER · TTFA |
| 语音克隆 | SECS (ECAPA余弦值) | MOS · CER |
| 说话人验证 | EER | minDCF · 工作点下的FAR / FRR |
| 语音分段 | DER | JER · 说话人混淆度 |
| 音频分类 | top-1准确率 · mAP | 宏F1分数 · 各类别召回率 |
| 音乐生成 | FAD | CLAP得分 · 听众评分MOS |
| 音频语言模型 | MMAU-Pro | LongAudioBench测试结果 · AudioCaps FENSE指标 |
| 流式端到端传输 | P50/P95延迟 | WER · MOS |

## 概念概述

![音频评估矩阵——指标、任务与2026年排行榜对比](../assets/eval-landscape.svg)

### ASR指标

**WER（词错误率）**。计算公式为 `(S + D + I) / N`。在评分前需将文本转换为小写、去除标点，并对数字进行归一化处理。可使用 `jiwer` 或 OpenAI 的 `whisper_normalizer` 工具完成。若WER低于5%，则表示语音识别准确率与人类相当。

**CER（字符错误率）**。公式相同，但计算粒度为字符级。适用于分词存在歧义的声调语言（如普通话、粤语）。

**RTFx（逆实时因子）**。指每实际秒数所能处理的音频秒数，数值越高表示性能越好。Parakeet-TDT 的该值为3380倍，而 Whisper-large-v3 的该值约为30倍。

**首个词元延迟**。从音频输入到生成首个转录词元之间的实际时间。对于流式服务而言至关重要。Deepgram Nova-3的该值约为150毫秒。

### TTS指标

**MOS（平均主观评分）。** 由人类评分员给出1-5分的评分。虽为行业金标准，但处理速度较慢。每个样本需收集20名以上听音者评分，每个模型则需要100个以上样本。

**UTMOS（2022-2026版）。** 基于机器学习的MOS预测器。在标准基准测试中与人类评分的关联度约为0.9。F5-TTS模型的UTMOS评分为3.95，而真实值则为4.08。

**SECS（说话人编码器余弦相似度）。** 用于语音克隆技术。通过计算参考语音与克隆后语音的ECAPA嵌入向量之间的余弦值来判断相似度。若该值大于0.75，则表示克隆出的语音可被识别。

**WER-on-ASR-round-trip。** 先对文本转语音生成的输出进行Whisper模型识别，再根据输入文本计算WER值。该方法可用于检测语音清晰度的下降情况。2026年的最新最佳实践标准为CER低于2%。

**TTFA（首次生成音频的时间）。** 指从指令发出到首次产生音频的墙钟时间延迟。Kokoro-82M模型的该值为约100毫秒，而F5-TTS模型的该值约为1秒。

### 语音克隆专用

**SECS + MOS + CER** 构成了一组三元指标。若克隆模型的 SECS 分数高而 MOS 分数低，意味着音色正确但听起来不自然；反之则表现为声音自然但说话者身份错误。

### 说话人验证

**EER（等错误率）**。即误报率与漏报率相等的阈值。在 VoxCeleb1-O 数据集上的 ECAPA 值为 0.87%。

**minDCF（最小检测成本）**。在选定操作点（通常为 FAR=0.01）下的加权成本。相比 EER，该指标更具实际生产应用价值。

### 语音分割

**DER（语音分割错误率）**。计算公式为 `(FA + Miss + Confusion) / total_speaker_time`，其中 `Miss`、`False Alarm` 和 `Confusion` 均以分数形式表示。在 AMI 会议场景中，合理的 DER 值通常在 10% 至 20% 之间。使用 pyannote 3.1 版本以及 Precision-2 商业版时，在录音质量良好的情况下，DER 值可低于 10%。

**JER（杰卡德错误率）**。作为 DER 的替代指标，其对短片段带来的偏差具有更强的鲁棒性。

### 音频分类

多标签任务：所有类别的**mAP（平均精度均值）**。在 AudioSet 数据集上，BEATs-iter3 的 mAP 值为 0.548。

多类互斥任务：**top-1 准确率**与**top-5 准确率**。在 Speech Commands v2 中，使用 Audio-MAE 评估方法时 top-1 准确率为 99.0%。

不平衡数据集：**宏 F1 值**与**各类别召回率**。需按类别分别报告结果——整体准确率会掩盖某些类别存在缺陷的事实。

### 音乐生成

**FAD（Fréchet音频距离）**。指真实音频与生成音频的VGGish嵌入分布之间的距离。在MusicCaps数据集上，MusicGen-small的值为4.5，MusicLM的值为4.0。数值越低越好。

**CLAP分数**。基于CLAP嵌入计算的文本-音频对齐度得分。大于0.3表示对齐程度合理。

**听众评分MOS**。仍是衡量消费级音乐质量的核心指标。在TTS Arena平台上，Suno v5的ELO评分为1293（该评分来自配对的人类用户偏好）。

### 语音语言基准测试

**MMAU（大规模多音频理解）**。包含10,000对音频与问题配对数据。

**MMAU-Pro**。包含1,800个高难度任务，分为四类：语音 / 声音 / 音乐 / 多音频。在四分类任务中随机出题的概率为25%；Gemini 2.5 Pro的整体准确率约为60%，而在所有模型中的多音频任务准确率约为22%。

**LongAudioBench**。包含带有语义查询的多分钟长度音频片段。Audio Flamingo Next的表现优于Gemini 2.5 Pro。

**AudioCaps / Clotho**。用于字幕生成的基准测试，评估指标包括SPICE、CIDEr和FENSE。

### 流式语音合成

**延迟 P50 / P95 / P99。** 指从用户结束说话到首次听到响应的实时时长。Moshi：200 毫秒；GPT-4o Realtime：300 毫秒。

输出端的 **WER / MOS** 指标。

**插话响应速度。** 指从用户中断对话到助手静音的时长。目标值应小于 150 毫秒。

### 2026年排行榜

| 排行榜 | 任务类型 | URL |
|------------|--------|-----|
| Open ASR 排行榜（HF） | 英语 + 多语言 + 长文本 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena（HF） | 英语语音合成 | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | 语音合成 + 语音识别，基于配对评分的 ELO 分数 | `artificialanalysis.ai/speech` |
| MMAU-Pro | 大语言模型推理能力测试 | `mmaubenchmark.github.io` |
| SpeakerBench / VoxSRC | 说话人识别 | `voxsrc.github.io` |
| MMAU 音乐子集 | 音乐领域的大语言模型测试 | （属于 MMAU 范围内） |
| HEAR 基准测试 | 自监督音频处理 | `hearbenchmark.com` |

## 构建它

### 步骤 1：带归一化的WER计算

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 步骤 2：TTS 往返词错误率

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步骤 3：用于语音克隆的 SECS 技术

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步骤 4：音乐生成的 FAD 方法

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步骤 5：用于说话人验证的 EER（代码与第 6 课相同）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 使用它

将每次部署与一个固定的评估工具集绑定，该工具集会在每个模型更新时运行。需遵循以下三条基本原则：

1. **在评分前进行标准化处理。** 将文本转换为小写、去除标点符号，并对数字进行扩展处理。同时需说明所采用的标准化规则。
2. **报告分布情况而非平均值。** 延迟指标应报告 P50/P95/P99 值；分类任务需报告各类别的召回率；MMAU 指标则需按类别分别报告。
3. **仅使用一个标准的公开基准测试。** 即使实际生产数据存在差异，通过 Open ASR / TTS Arena / MMAU 等标准测试进行评估，也能让评审人员实现公平对比。

## 常见陷阱

- **UTMOS 外推问题。** 该模型在 VCTK 风格的纯净语音数据上训练，对于含噪声、克隆或情感化的音频表现较差。
- **MOS 评估小组的偏差。** 20 名 Amazon Mechanical Turk 工作人员并不等同于 20 名目标用户。若重要性较高，建议使用自定义域名构建评估小组。
- **FAD 取决于参考数据集。** 需在各个模型之间使用相同的参考分布进行对比。
- **综合 WER 值。** 整体 5% 的 WER 值可能掩盖了带口音语音 30% 的高错误率，应按人口统计特征分项报告。
- **公共基准测试的饱和现象。** 大多数前沿模型在标准基准测试中的表现已接近极限，建议构建能够反映自身业务流量的内部保留数据集。

## 发布它

将文件保存为 `outputs/skill-audio-evaluator.md`。为任何音频模型的发布选择相应的指标、基准测试以及报告格式。

## 练习题

1. **简单。** 运行 `code/main.py`，使用示例输入数据计算 WER / CER / EER / SECS / FAD-ish / MMAU-ish 的得分。
2. **中等难度。** 构建一个用于 TTS 双向评估的 WER 测评工具。将您使用的 Kokoro 或 F5-TTS 生成的输出通过 Whisper 处理，对 50 条提示语进行 WER 计算，并标记出 WER 高于 10% 的提示语。
3. **高难度。** 使用 MMAU-Pro 语音数据以及多音频子集（每个子集包含 50 个样本）来评估您在第 10 课中选择的 LALM 模型。报告各类别的准确率，并与已公布的数值进行对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| WER | ASR得分 | 经过归一化处理后的单词级得分，公式为 `(S+D+I)/N`。 |
| CER | 字符WER | 用于声调语言或字符级系统的指标。 |
| MOS | 人类评分 | 1-5分的评级；基于20名以上听众对100个样本的评估。 |
| UTMOS | ML MOS预测器 | 通过机器学习训练得到的模型，其与人类评分的相关性约为0.9。 |
| SECS | 声音克隆相似度 | 参考声音与克隆声音之间的ECAPA余弦值。 |
| EER | 发言人验证得分 | 当FAR等于FRR时的阈值。 |
| DER | 分段识别得分 | `(FA + Miss + Confusion) / 总数` 的计算结果。 |
| FAD | 音乐生成质量 | 基于VGGish嵌入向量的Fréchet距离值。 |
| RTFx | 吞吐量 | 每实际秒数处理的音频秒数。 |

## 延伸阅读

- [jiwer](https://github.com/jitsi/jiwer) — 提供标准化功能的WER/CER库。
- [UTMOS (Saeki等人，2022)](https://arxiv.org/abs/2204.02152) — 基于学习的MOS预测器。
- [Fréchet音频距离 (Kilgour等人，2019)](https://arxiv.org/abs/1812.08466) — 音乐生成领域的标准指标。
- [Open ASR排行榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) — 2026年的实时排名榜单。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena) — 基于人类评分的TTS排行榜。
- [MMAU-Pro基准测试](https://mmaubenchmark.github.io/) — LALM推理能力的排行榜。
- [HEAR基准测试](https://hearbenchmark.com/) — 音频SSL领域的基准测试。
