# 语音识别（ASR）——CTC、RNN-T、注意力机制

> 语音识别实质上是在每个时间步对音频进行分类，再通过一个能够区分英语语音与静音的序列模型将这些分类结果串联起来。CTC、RNN-T 和注意力机制是实现这一目标的三种方法。请选择其中一种并理解其原理。

**类型：** 实践构建
**语言：** Python
**先修知识：** 第6阶段 · 02（频谱图与梅尔频率），第5阶段 · 08（用于文本处理的CNN与RNN），第5阶段 · 10（注意力机制）
**耗时：** 约45分钟

## 问题所在

你有一段时长为10秒、采样率为16 kHz的音频片段，需要将其转换为字符串“turn on the kitchen lights”。面临的挑战在于结构上的不匹配：音频帧与字符之间并不存在一一对应的对应关系。“okay”这个词可能对应的音频时长在200毫秒到1200毫秒之间，语句中还包含静音间隔，且某些音素的长度也各不相同。因此，输出文本的token数量无法提前确定。

目前有三种方法可以解决这一问题：

1. **CTC（连接主义时间分类模型）**：为每一帧生成包含特殊*空白*标记在内的token概率，在解码阶段合并重复出现的token和空白。该模型属于非自回归结构，处理速度快，被wav2vec 2.0和MMS所采用。

2. **RNN-T（循环神经网络转录器）**：通过一个联合网络，根据编码器输出的帧信息以及之前的token来预测下一个token。该模型支持流式处理，被Google的离线ASR系统及NVIDIA的Parakeet所使用。

3. **注意力编码器-解码器**：编码器将音频压缩为隐藏状态，解码器则通过交叉注意力机制自回归地生成token。该模型被Whisper和SeamlessM4T所采用。

截至2026年，在LibriSpeech测试集上的最佳平均词错误率（WER）分别为：NVIDIA的Parakeet-TDT-1.1B模型为1.4%，Whisper-Large-v3-turbo模型为1.58%。虽然两者之间的误差极小，但在实际部署层面的差异却十分显著。

## 概念概述

![三种ASR模型：CTC、RNN-T、注意力编码器-解码器](../assets/asr-formulations.svg)

**CTC模型的原理。** 编码器会为`V+1`个标记（V个字符 + 空白）输出`T`个帧级的概率分布。对于长度为`U < T`的目标字符串`y`，任何能够映射到`y`的帧对齐方式都会被计入统计。CTC损失函数会对所有此类对齐方式进行求和。在推理阶段：对每个帧执行argmax操作，合并重复项，并移除空白标记。

优点：非自回归结构，可流式处理，无需前瞻信息。缺点：存在*条件独立性假设*——即每一帧的预测结果相互独立，因此没有内置的语言模型。可通过束搜索或浅层融合的方式引入外部语言模型来解决这一问题。

**RNN-T模型的原理。** 它在原有基础上增加了一个*预测器*网络，用于嵌入标记历史信息；同时还有一个*连接器*网络，负责将预测器的状态与编码器的帧级输出结合，生成关于`V+1`个标记的联合概率分布（其中的`+1`代表空值/不发音）。该模型明确地建模了CTC所忽略的条件依赖关系。由于每一步仅依赖于之前的帧和标记，因此具备流式处理能力。

优点：兼具流式处理能力和内置语言模型。缺点：训练更为复杂且占用更多内存（需要构建3D损失网格）；RNN-T相关的损失核函数本身就形成了一个独立的完整库体系。

**注意力编码器-解码器模型。** 编码器由6到32层Transformer层组成，处理对数梅尔频谱帧。解码器同样由6到32层Transformer层构成，通过交叉注意力机制关注编码器的输出，从而以自回归方式生成标记。该模型没有对齐约束——注意力可以自由地聚焦在音频的任意位置。除非对注意力机制进行限制（如分块处理的Whisper-Streaming技术，2024年提出），否则无法实现流式处理。

优点：在离线ASR任务中能够达到最高质量，且可使用标准的seq2seq训练工具轻松实现训练。缺点：自回归式的处理延迟与输出长度成正比；若不进行特殊工程设计，则无法实现流式处理。

### WER：唯一的数值

**词错误率** = `(S + D + I) / N`，其中 S 表示替换数，D 表示删除数，I 表示插入数，N 表示参考词的总量。该指标在词级上对应莱文斯坦编辑距离。数值越低越好。WER 超过 20% 通常无法使用；低于 5% 则可达到与人类相当的朗读质量。标准基准测试中的 2026 年数据如下：

| 模型 | LibriSpeech test-clean | LibriSpeech test-other | 参数量 |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 11 亿参数 |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 8.09 亿参数 |
| Canary-1B Flash | 1.48% | 2.87% | 10 亿参数 |
| Seamless M4T v2 | 1.7% | 3.5% | 23 亿参数 |

以上所有模型均基于编解码器或 RNN-T 架构。纯 CTC 系统（如 wav2vec 2.0）在 test-clean 数据集上的词错误率约为 1.8–2.1%。

## 构建它

### 步骤 1：贪婪型 CTC 解码

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

两条规则：合并连续重复的字符，删除空白字符。示例：`a a _ _ a b b _ c` → `a a b c`。

### 步骤 2：beam-search CTC 算法

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

生产环境采用前缀树束搜索结合语言模型融合的技术；这就是其概念框架。

### 步骤 3：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步骤 4：使用 Whisper 进行推理

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

2026年最强大的通用ASR模型一句话总结：在24GB显存上运行，实时处理速度约为常规模型的20倍。

### 步骤 5：使用 Parakeet 或 wav2vec 2.0 进行流式处理

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

流式语音识别需要分块编码器注意力机制与状态延续功能；应使用支持该功能的库（Parakeet 使用 NeMo，`transformers` 管道则配合 `chunk_length_s` 参数）。

## 使用它

2026年技术栈：

| 场景 | 推荐模型 |
|-----------|----------|
| 英语内容，离线环境，需最高质量 | Whisper-large-v3-turbo |
| 多语言处理，需高稳定性 | SeamlessM4T v2 |
| 流式处理，要求低延迟 | Parakeet-TDT-1.1B 或 Riva |
| 边缘设备、移动端，延迟需低于500毫秒 | 量化后的Whisper-Tiny或Moonshine（2024版） |
| 长文本处理 | 基于VAD进行分块的Whisper（WhisperX版本） |
| 特定领域（医疗、法律等） | 对wav2vec 2.0进行微调并融合领域专用语言模型 |

## 2026年仍会存在的缺陷与隐患

- **无VAD功能。** 在静音环境下运行Whisper会导致幻觉现象（如“感谢观看！”）。务必配合VAD进行语音检测。
- **字符级、单词级与子词级的WER指标。** 应在完成标准化处理（转为小写并去除标点）之后，再报告单词级别的WER值。
- **语言ID识别偏差。** Whisper的自动语言识别功能可能会将含噪声的音频片段错误地识别为日语或威尔士语；若已知语言类型，请强制设置 `language="en"`。
- **未分块处理的过长音频片段。** Whisper的处理窗口长度为30秒。对于更长的音频，需使用 `chunk_length_s=30, stride=5` 参数进行分块处理。

## 发布它

将文件保存为 `outputs/skill-asr-picker.md`。针对特定的部署目标，选择模型、解码策略、分块方式以及大语言模型融合方案。

## 练习题

1. **简单。** 运行 `code/main.py`。该脚本会直接解码手工生成的CTC输出，并据此计算与参考文本的WER值。
2. **中等难度。** 正确实现步骤2中的前缀树束搜索算法（需考虑空白字符合并规则）。在包含10个样本的合成数据集上，将其结果与贪婪解码法进行对比。
3. **高难度。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 数据集上使用 `whisper-large-v3-turbo` 模型。对前100条语音记录计算WER值，并与已公布的数值进行比对。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|-----------------|----------|
| CTC | 空白标记损失 | 对所有帧到标记的对齐进行求和；不属于自回归模型。 |
| RNN-T | 流式损失 | CTC + 下一个标记预测器；能够处理词序问题。 |
| Attention enc-dec | Whisper风格 | 编码器 + 交叉注意力解码器；离线场景下的最佳质量。 |
| WER | 您报告的数值 | 在单词层面的 `(S+D+I)/N` 值。 |
| Blank | 空值 | CTC中的特殊标记，用于表示“当前帧不输出任何内容”。 |
| LM fusion | 外部语言模型 | 在束搜索过程中加入加权的语言模型对数概率。 |
| VAD | 静音检测器 | 语音活动检测器；用于剔除非语音内容。 |

## 延伸阅读

- [Graves 等人 (2006). 连通主义时间分类方法](https://www.cs.toronto.edu/~graves/icml_2006.pdf) —— CTC 方法的相关论文。  
- [Graves (2012). 基于 RNN 的序列转换技术](https://arxiv.org/abs/1211.3711) —— RNN-T 方法的相关论文。  
- [Radford 等人 / OpenAI (2022). Whisper：基于大规模弱监督的鲁棒语音识别系统](https://arxiv.org/abs/2212.04356) —— 2022 年领域的权威论文；2024 年推出了 v3-turbo 版本。  
- [NVIDIA NeMo — Parakeet-TDT 模型](https://huggingface.co/nvidia/parakeet-tdt-1.1b) —— 2026 年 Open ASR 排名榜的领先模型。  
- [Hugging Face — Open ASR 排名榜](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard) —— 涵盖 25 种以上模型的实时基准测试平台。
