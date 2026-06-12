# 频谱图、梅尔频率刻度与音频特征

> 神经网络并不擅长直接处理原始波形，它们更倾向于使用频谱图作为输入，而梅尔频谱图则是更好的选择。在2026年，所有的语音识别、文本转语音以及音频分类系统，其性能成败都取决于这一项预处理决策。

**类型：** 构建
**语言：** Python
**先修课程：** 第6阶段 · 01（音频基础）
**耗时：** 约45分钟

## 问题所在

取一段时长为10秒、采样率为16 kHz的音频片段。该片段包含160,000个浮点数值，所有值均在`[-1, 1]`范围内，且与“狗叫声”或“猫”这类标签几乎完全不相关。原始波形虽然蕴含了相关信息，但其形式使得模型难以直接提取。即使两个相同的音素相隔仅100毫秒发音，其生成的原始样本也会存在显著差异。

频谱图可解决这一问题。它去除了人类感知无法识别的时间细节（如微秒级的抖动），同时保留了人类能够注意到的结构特征（即在约10–25毫秒的时间窗口内哪些频率具有能量）。

梅尔频谱图则更进一步。人类的音高感知是呈对数关系的：100 Hz与200 Hz之间的“距离感”与1000 Hz与2000 Hz之间的“距离感”相同。梅尔刻度通过对频率轴进行变形来匹配这种感知特性。从2010年到2026年，基于梅尔刻度的频谱图一直是语音机器学习中最核心的特征之一。

## 概念概述

![从波形到STFT，再到mel频谱图，最后到MFCC的流程图](../assets/mel-features.svg)

**STFT（短时傅里叶变换）。** 将波形分割为相互重叠的帧（通常：窗口长度为25毫秒，步长为10毫秒，即每帧400个样本；在16 kHz采样率下则为160个样本）。将每个帧乘以窗函数（默认使用汉宁窗；哈明窗则在不同方面有略微不同的权衡）。对每个帧进行FFT变换。将各帧的幅度谱堆叠成形状为`(n_frames, n_freq_bins)`的矩阵，这就是最终的频谱图。

**对数幅度。** 原始幅度值跨越5到6个数量级。通过计算`log(|X| + 1e-6)`或`20 * log10(|X|)`来压缩动态范围。所有实际生产流程都使用对数幅度，而非原始幅度。

**Mel刻度。** 频率`f`（单位：Hz）通过公式`m = 2595 * log10(1 + f / 700)`映射到mel值`m`。在1 kHz以下该映射关系大致呈线性，而在更高频率下则近似为对数关系。覆盖0–8 kHz范围的80个mel刻度是标准的语音识别输入格式。

**Mel滤波器组。** 这是一组在Mel刻度上等间距分布的三角形滤波器。每个滤波器都是相邻FFT频段的加权之和。将STFT幅度与滤波器组矩阵相乘，即可通过一次矩阵乘法得到mel频谱图。

**对数-Mel频谱图。** 计算`log(mel_spec + 1e-10)`得到的结果。这是Whisper、Parakeet以及SeamlessM4T等系统的输入格式，也是2026年通用的音频前端处理方式。

**MFCCs。** 首先获取对数-Mel频谱图，然后对其进行DCT（II型）变换，并保留前13个系数。这一过程能够降低特征之间的相关性并进一步压缩数据量。在大约2015年之前，MFCC一直是核心特征表示方式，直到基于原始对数-Mel数据的CNN和Transformer模型出现后其地位才受到挑战。目前它仍被用于说话人识别技术中（如x-vectors、ECAPA）。

**分辨率权衡。** FFT窗口大小越大，频率分辨率越高，但时间分辨率越差。音频机器学习领域的默认设置通常为25毫秒/10毫秒；音乐处理则采用50毫秒/12.5毫秒的设置；而用于检测瞬态现象（如鼓点、爆破音）时则使用5毫秒/2毫秒的设置。

```figure
spectrogram-window
```

## 构建它

### 步骤 1：构建波形框架

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一个时长为10秒、采样率为16 kHz的音频片段，若设置`frame_len=400, hop=160`，则会产生998帧。

### 步骤 2：Hann 窗

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在FFT运算之前进行逐元素相乘操作。此举可消除因在非零端点处截断数据而产生的频谱泄漏现象。

### 步骤 3：STFT 振幅

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

在生产环境中，通常使用 `torch.stft` 或 `librosa.stft`（基于 FFT 的向量化实现）。此处的循环仅用于教学演示，在 `code/main.py` 中会对短音频片段进行处理。

### 步骤 4：mel滤波器组

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

使用 `n_fft=400` 对 0–8 kHz 范围内的 80 个梅尔频率点进行计算，可得到一个大小为 `(80, 201)` 的矩阵。将尺寸为 `(n_frames, 201)` 的 STFT 模值矩阵与其转置相乘，即可获得尺寸为 `(n_frames, 80)` 的梅尔频谱图。

### 第 5 步：log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常见替代方案：`librosa.power_to_db`（参考归一化的分贝值），以及 `10 * log10(power + eps)`。Whisper 采用更为复杂的裁剪与归一化流程（详见 Whisper 的 `log_mel_spectrogram` 函数）。

### 步骤 6：MFCC 特征提取

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

对每个对数梅尔帧应用离散余弦变换（DCT），并保留前13个系数。这些系数即构成了MFCC矩阵。通常会舍弃第一个系数，因为它用于表示整体能量信息。

## 使用它

2026年技术栈：

| 任务 | 特性 |
|------|----------|
| ASR（Whisper、Parakeet、SeamlessM4T） | 80对数梅尔频谱，10毫秒跳点间隔，25毫秒窗口长度 |
| TTS声学模型（VITS、F5-TTS、Kokoro） | 80梅尔频谱，5–12毫秒的跳点间隔以实现精细的时间控制 |
| 音频分类（AST、PANNs、BEATs） | 128对数梅尔频谱，10毫秒跳点间隔 |
| 说话人嵌入模型（ECAPA-TDNN、WavLM） | 80对数梅尔频谱或原始波形的SSL表示 |
| 音乐处理（MusicGen、Stable Audio 2） | EnCodec离散令牌（非梅尔频谱形式） |
| 关键词检测 | 小型设备适用40个MFCC特征 |

经验法则：**若不涉及音乐处理，建议从80对数梅尔频谱开始。** 若需采用其他参数，则需提供相应的证明依据。

## 2026年仍会存在的缺陷与隐患

- **Mel 值数量不匹配。** 训练时使用 80 个 Mel 值，推理时使用 128 个 Mel 值，导致静音场景下失败。请记录训练端与推理端的特征形状。
- **上游采样率不匹配。** 以 22.05 kHz 计算的 Mel 值与 16 kHz 下的计算结果存在差异。请在特征提取之前修正采样率。
- **dB 与对数单位的问题。** Whisper 需要的是对数梅尔值，而非 dB 梅尔值。某些高频处理管道能够自动检测，但自定义代码则无法实现。
- **归一化方式不一致。** 训练时采用逐句归一化，推理时采用全局归一化。这种差异会在生产环境中引发错误，导致词错误率翻倍。
- **填充导致的泄漏效应。** 对音频片段末尾进行零填充会导致后续帧的频谱呈现平坦特征。建议采用对称填充或重复数据的方式处理。

## 发布它

将文件保存为 `outputs/skill-feature-extractor.md`。该技能会针对指定的模型目标，选择特征类型、mel计数、帧/跳变间隔以及归一化参数。

## 练习题

1. **简单。** 运行 `code/main.py`。该脚本会生成一个频率在 200 Hz 到 4000 Hz 之间扫描的啁啾信号，并输出每一帧中 argmax mel 分箱的值。可选择性绘制图表，以确认其与频率扫描曲线一致。
2. **中等难度。** 使用 `n_mels` 取 `{40, 80, 128}` 值，`frame_len` 取 `{200, 400, 800}` 值重新运行程序。测量时间轴上尖锐峰值的带宽。哪种参数组合能最清晰地分辨出该啁啾信号？
3. **高难度。** 实现 `power_to_db` 函数，并使用 (a) 原始对数 mel 频谱，(b) 参考值为最大值的 dB 形式 mel 频谱，以及 (c) MFCC-13 + delta + delta-delta 特征，对比在 AudioMNIST 数据集上运行的小型 CNN 分类器的 ASR 准确率。需报告 top-1 准确率数值。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| Frame | 一个切片 | 输入到单次 FFT 的 25 毫秒波形数据块。 |
| Hop | 步长 | 连续帧之间的样本间隔；ASR 的默认值为 10 毫秒。 |
| Window | Hann/Hamming 窗函数 | 一种逐点乘法器，用于将帧的边缘渐变为零。 |
| STFT | 频谱图生成器 | 对数据进行分帧并应用窗函数后的 FFT；可生成时间-频率矩阵。 |
| Mel | 变形频率刻度 | 基于感知的对数尺度；计算公式为 `m = 2595·log10(1 + f/700)`。 |
| Filterbank | 滤波器组 | 用于将 STFT 结果映射到 Mel 刻度上的三角滤波器阵列。 |
| Log-mel | Whisper 的输入格式 | 表达式为 `log(mel_spec + eps)`；该格式于 2026 年被标准化。 |
| MFCC | 传统特征向量 | 对 log-mel 数据进行 DCT 变换得到的结果；包含 13 个去相关的系数。 |

## 延伸阅读

- [Davis, Mermelstein (1980). 单音节词识别参数表示法的比较](https://ieeexplore.ieee.org/document/1163420) —— MFCC相关论文。
- [Stevens, Volkmann, Newman (1937). 心理音高强度的测量尺度](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/) —— 最初的mel刻度定义。
- [OpenAI — Whisper源代码，log_mel_spectrogram函数](https://github.com/openai/whisper/blob/main/whisper/audio.py) —— 查阅参考实现代码。
- [librosa特征提取文档](https://librosa.org/doc/main/feature.html) —— `mfcc`、`melspectrogram`以及hop/window参数的参考资料。
- [NVIDIA NeMo — 音频预处理功能](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers) —— 用于Parakeet + Canary模型的生产级处理流程。
