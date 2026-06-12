# 音频基础——波形、采样与傅里叶变换

> 波形是原始信号，频谱图是其表示形式，而 Mel 特征则是适合机器学习处理的格式。所有现代的 ASR 和 TTS 流水线都会经历这一系列步骤，其中的第一步就是理解采样与傅里叶变换。

**类型：** 学习
**语言：** Python
**先修课程：** 第一阶段 · 06（向量与矩阵）、第一阶段 · 14（概率分布）
**时长：** 约 45 分钟

## 问题所在

麦克风会生成压力随时间变化的信号，而神经网络则需要处理张量数据。在这两者之间存在着一系列约定俗成的规范，一旦违反这些规范，就会引发“无声的错误”：模型训练正常，但词错误率却翻倍；文本转语音功能会产生杂音；或者语音克隆系统记住了麦克风的声音而非说话人的声音。

语音系统中的每一个错误都可以归结为以下三个问题之一：

1. 数据是以何种采样率录制的？模型又期望怎样的采样率？
2. 该信号是否存在混叠现象？
3. 您是在处理原始样本，还是在处理频率表示形式？

只要正确解答了这些问题，第六阶段的其余工作就能顺利推进。一旦出错，即便是 Whisper-Large-v4 这样的模型也会生成错误结果。

## 概念概述

![波形、采样、DFT以及频率频段的可视化示意图](../assets/audio-fundamentals.svg)

**波形。** 由位于 `[-1.0, 1.0]` 范围内的浮点数构成的一维数组，按样本编号排序。若需转换为秒级时间，需用采样率除以该数值：`t = n / sr`。一段时长为 10 秒、采样率为 16 kHz 的音频片段，其数据量即为 160,000 个浮点数。

**采样率（sr）。** 每秒的样本数量。2026 年常见的采样率如下：

| 采样率 | 应用场景 |
|------|---------|
| 8 kHz | 电话通信、传统 VOIP。4 kHz 处为奈奎斯特频率，会导致辅音失真，不适合用于 ASR。 |
| 16 kHz | ASR 的标准采样率。Whisper、Parakeet、SeamlessM4T v2 等工具均采用 16 kHz 采样。 |
| 22.05 kHz | 老款 TTS 模型的发声器训练用采样率。 |
| 24 kHz | 现代 TTS（如 Kokoro、F5-TTS、xTTS v2）的常用采样率。 |
| 44.1 kHz | CD 音频及普通音乐的标准采样率。 |
| 48 kHz | 电影音频、专业音频以及高保真 TTS（如 VALL-E 2、NaturalSpeech 3）的常用采样率。 |

**奈奎斯特-香农定理。** 采样率为 `sr` 时，能够无歧义地表示最高为 `sr/2` 的频率。`sr/2` 这一界限即为*奈奎斯特频率*。高于奈奎斯特频率的能量会发生*混叠现象*——即折叠到较低的频率中——从而破坏信号质量。在下采样之前务必先进行低通滤波。

**位深度。** 16 位 PCM（有符号 int16，取值范围为 ±32,767）是通用的数据交换格式。音乐文件通常使用 24 位，而内部 DSP 处理则常用 32 位浮点数。像 `soundfile` 这样的库虽然读取的是 int16 格式的数据，但会以 `[-1, 1]` 范围内的 float32 数组形式提供数据。

**傅里叶变换。** 任何有限信号都可以表示为不同频率的正弦波之和。离散傅里叶变换（DFT）针对 `N` 个样本计算出 `N` 个复数系数，每个频率频段对应一个系数。第 `k` 个频段对应的频率为 `k · sr / N` Hz。该系数的幅值表示该频率处的振幅，角度则表示相位。

**FFT。** 快速傅里叶变换：当 `N` 为 2 的幂时，可用于高效计算 DFT，其时间复杂度为 `O(N log N)`。所有音频处理库在底层均使用 FFT 算法。例如，对 1024 个样本、采样率为 16 kHz 的数据执行 FFT 后，可得到 512 个有效的频率频段，覆盖 0–8 kHz 范围，频率分辨率约为 15.6 Hz。

**分帧与加窗。** 我们不会对整个音频片段直接进行 FFT 计算。而是将其分割成相互重叠的*帧*（通常帧长为 25 毫秒，帧间跳过时间为 10 毫秒），随后用窗函数（如汉宁窗、哈明窗）乘以每个帧的数据以消除边缘的不连续性，最后再对每个帧分别进行 FFT 计算。这就是短时傅里叶变换（STFT）。第 02 课将在此基础上继续讲解。

```figure
mel-scale
```

## 构建它

### 步骤 1：读取音频片段并绘制波形图

`code/main.py` 仅使用标准库中的 `wave` 模块，以确保演示版本无需任何外部依赖。在正式生产环境中，则应使用 `soundfile` 或 `torchaudio.load`（二者均返回形如 `(waveform, sr)` 的元组）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步骤 2：从基本原理出发合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

频率为 440 Hz、音高为音乐会 A 音的正弦波，在 16 kHz 下持续 1 秒，共包含 16,000 个浮点数值。请使用 `wave.open(..., "wb")` 并采用 16 位 PCM 编码来生成该波形。

### 步骤 3：手动计算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)` — 对于 `N=256` 的场景可用于验证正确性，但用于处理真实音频时毫无用处。实际代码会调用 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步骤 4：查找主导频率

幅值峰值索引 `k_star` 对应的频率为 `k_star * sr / N`。将该公式应用于 440 Hz 正弦波时，应在频段 `440 * N / sr` 处出现峰值。

### 步骤 5：演示别名功能

在 10 kHz 的采样率下对 7 kHz的正弦波进行采样（奈奎斯特频率为 5 kHz）。由于 7 kHz的频率高于奈奎斯特频率，会发生混叠现象，其频率变为 `10 − 7 = 3 kHz`。FFT结果中的峰值将出现在 3 kHz 处。这是典型的混叠演示案例，也是为何每款DAC/ADC都会配备极低通滤波器的原因。

## 使用它

2026年实际部署的技术栈：

| 任务 | 库 | 原因 |
|------|---------|-----|
| 读取/写入 WAV/FLAC/OGG 文件 | `soundfile`（libsndfile 的封装） | 速度最快、稳定性高，返回 float32 类型数据。 |
| 重采样 | `torchaudio.transforms.Resample` 或 `librosa.resample` | 内置正确的抗混叠处理功能。 |
| STFT / Mel 分析 | `torchaudio` 或 `librosa` | 兼容 GPU，属于 PyTorch 生态系统。 |
| 实时流式处理 | `sounddevice` 或 `pyaudio` | 基于跨平台的 PortAudio 接口。 |
| 文件信息检测 | `ffprobe` 或 `soxi` | 命令行工具，响应迅速，可输出采样率、声道数及编码格式等信息。 |

决策规则：**在考虑其他任何参数之前，首先确保采样率匹配**。Whisper 要求输入为 16 kHz 的单声道 float32 数据。若输入 44.1 kHz 的双声道数据，生成的输出将出现异常，看似模型存在缺陷。

## 发布它

将文件保存为 `outputs/skill-audio-loader.md`。该技能可帮助你检查音频输入是否符合下游模型的预期，若不符合则能进行正确的重采样处理。

## 练习题

1. **简单。** 在 16 kHz 下合成一个时长为 1 秒的混合信号，包含 220 Hz、440 Hz 和 880 Hz 的频率分量。执行 DFT 计算，并确认在预期的频段位置出现三个峰值。
2. **中等难度。** 以 48 kHz 的采样率录制一段时长为 3 秒的本人语音 WAV 文件。首先使用 `torchaudio.transforms.Resample`（配合抗混叠处理）将其下采样至 16 kHz，然后再通过简单的等间隔抽取方式（每三个样本取一个）将其下采样至 16 kHz。对两种处理后的信号分别进行 FFT 计算。混叠现象出现在哪个频段？
3. **高难度。** 仅使用 `math` 模块以及第 3 步中的 DFT 算法，从零开始实现 STFT 函数。设置帧大小为 400、步长为 160，并采用汉宁窗函数。使用 `matplotlib.pyplot.imshow` 绘制幅值图。该图像即为第 02 课的频谱图。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 采样率 | 每秒的样本数 | ADC 测量信号时的频率，单位为 Hz。 |
| 奈奎斯特频率 | 可表示的最高频率 | `sr/2`；高于此频率的能量会发生混叠现象。 |
| 位深度 | 每个样本的分辨率 | `int16` = 65,536 级别；`float32` = 在 `[-1, 1]` 范围内的 24 位精度。 |
| DFT | 序列的傅里叶变换 | `N` 个样本 → `N` 个复数频率系数。 |
| FFT | 快速 DFT | 时间复杂度为 `O(N log N)` 的算法，要求 `N` 为 2 的幂。 |
| 频率bin | 频率列 | `k · sr / N` Hz；分辨率 = `sr / N`。 |
| STFT | 底层实现的频谱图 | 对时间序列进行分帧并加上窗函数后的 FFT。 |
| 混叠现象 | 异常的频率“幽灵” | 高于奈奎斯特频率的能量会反射到较低的频率bin中。 |

## 延伸阅读

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) — 奠定采样定理基础的论文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm) — 免费且权威的数字信号处理教材。
- [librosa docs — 音频入门指南](https://librosa.org/doc/latest/tutorial.html) — 带有代码示例的实际操作教程。
- [Heinrich Kuttruff — Room Acoustics (第6版)](https://www.routledge.com/Room-Acoustics/Kuttruff/p/book/9781482260434) — 阐述为何现实世界中的音频并非纯净正弦波的参考资料。
- [Steve Eddins — FFT解读笔记](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/) — 10分钟内讲清频率bin的概念。
