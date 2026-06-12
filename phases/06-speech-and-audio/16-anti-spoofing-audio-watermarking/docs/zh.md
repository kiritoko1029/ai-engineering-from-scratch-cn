# 语音防欺骗与音频水印技术 —— ASVspoof 5、AudioSeal、WaveVerify

> 语音克隆技术的推出速度超过了相应的防御措施。2026年的实际语音系统需要具备两项功能：一是能够区分真实语音与合成语音的检测器（如 AASIST、RawNet2），二是能够在压缩和编辑后依然保持完整性的水印技术（如 AudioSeal）。要么同时实现这两项功能，否则就不要推出语音克隆技术。

**类型：** 构建
**语言：** Python
**前置要求：** 第6阶段 · 06（说话人识别）、第6阶段 · 08（语音克隆）
**耗时：** 约75分钟

## 问题所在

三种相关的防御措施：

1. **反伪造/深度伪造检测。** 给定一段音频，判断其是合成的还是真实的？ASVspoof 基准测试（ASVspoof 2019 → 2021 → 5）被视为该领域的黄金标准。
2. **音频水印技术。** 在生成的音频中嵌入难以察觉的信号，以便后续由检测工具提取。Meta 开发的 AudioSeal 以及 WavMark 是可供选择的开源方案。
3. **经过认证的来源标识。** 对音频文件及其元数据进行加密签名。C2PA / Content Authenticity Initiative 即为此类标准。

检测技术用于应对不合作的攻击者；水印技术则用于满足合规要求——确保人工智能生成的音频能够被识别出来。这两种技术在 2026 年均属必备。

## 概念概述

![反伪造、水印技术与溯源机制——三大防御层级](../assets/spoofing-watermark.svg)

### ASVspoof 5——2024-2025年度基准测试工具

相较于以往版本的最大变化：

- **众包数据**（非工作室处理过的数据）——模拟真实环境。
- **约2000名说话者**（之前约为100人）。
- **32种攻击算法**，包括文本转语音、语音转换以及对抗性扰动。
- **两种研究方向**：独立的对抗措施检测；用于生物特征系统的抗欺骗自动语音识别系统。

ASVspoof 5的最新性能指标为EER约7.23%。而旧版本的ASVspoof 2019 LA的EER仅为0.42%。在实际应用中，针对真实环境中的音频片段，其EER预计在5%至10%之间。

### AASIST与RawNet2——检测模型系列

**AASIST**（2021年发布，更新至2026年）。基于频谱特征的图注意力机制。在ASVspoof 5对抗任务中目前处于最先进水平。

**RawNet2.** 采用原始波形作为卷积前端，并结合TDNN主干网络。结构较为简单，但经过微调后仍具备较强竞争力。

**NeXt-TDNN + SSL特征。** 2025年版本：融合ECAPA风格特征、WavLM特征以及焦点损失函数。在ASVspoof 2019 LA数据集上的EER值达到0.42%。

### AudioSeal — 2024年默认水印方案

Meta的**AudioSeal**（2024年1月发布，v0.2版本于2024年12月更新）。核心设计特点如下：

- **精准定位。** 以16 kHz的采样分辨率（即1/16000秒）对每一帧进行水印检测。
- **生成器与检测器联合训练。** 生成器负责学习如何嵌入不可听见的信号；检测器则通过数据增强技术来学习如何识别这些信号。
- **高度鲁棒。** 能够在MP3/AAC压缩、均衡处理、±10%的速度偏移以及信噪比为+10 dB的噪声环境中依然保持正常工作。
- **快速响应。** 检测器的运行速度可达实时速度的485倍，远快于WavMark。
- **大容量存储。** 每条语音数据中可嵌入16位的数据载荷，用于存储模型ID、生成时间戳及用户ID等信息。

### WavMark

AudioSeal 预处理阶段的原始基线。采用可逆神经网络，数据传输速率为 32 比特/秒。存在的问题包括：

- 同步的暴力搜索方法效率低下。
- 可通过高斯噪声或 MP3 压缩来消除该问题。
- 不适合实时应用场景。

### WaveVerify（2025年7月）

解决了AudioSeal的缺陷——尤其是时间维度上的操作（倒放、变速）。采用基于FiLM的生成器与专家混合检测器。在标准攻击场景下可与AudioSeal相媲美，并能处理时间编辑操作。

### 攻击者利用的漏洞

来自 AudioMarkBench 的数据：“在音高偏移处理后，所有水印的比特恢复准确率均低于 0.6，表明其已被几乎完全去除。”**音高偏移是一种通用的攻击手段。**没有任何一种 2026 年开发的水印能够完全抵御强烈的音高修改攻击。正因如此，在添加水印的同时还需要搭配检测技术（AASIST）。

### C2PA / 内容真实性计划

并非机器学习技术，而是一种清单格式。音频文件中包含经过加密签名的元数据，记载了创建工具、作者及生成日期等信息。Audobox/Seamless支持该格式的使用。它有助于追踪文件的来源信息；但如果恶意行为者重新编码并移除这些元数据，则将毫无作用。

## 构建它

### 步骤 1：一个简单的频谱特征检测器（示例模型）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成语音的高频能量通常异常平坦。语音生成检测器使用的是AASIST，而非此方法。不过这一直觉仍然是成立的。

### 步骤 2：AudioSeal 嵌入与检测

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### 步骤 3：评估——EER 值

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 步骤 4：生产环境集成

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每个版本都会包含：(1) 水印，(2) 已签名清单，(3) 符合保留策略的审计日志。

## 使用它

| 使用场景 | 防护措施 |
|----------|---------|
| 发送文本转语音 / 声音克隆 | 必须在每个输出中嵌入 AudioSeal（无商量余地） |
| 生物特征声音解锁 | AASIST 与 ECAPA 组合使用；需进行活体检测挑战 |
| 客服中心欺诈检测 | 对 20% 的来电样本应用 AASIST |
| 播客真实性验证 | 上传时使用 C2PA 签名，若为 AI 生成则使用 AudioSeal |
| 检测器的研究/训练 | ASVspoof 5 提供训练集、开发集和评估集 |

## 常见陷阱

- **无需运行检测器即可添加水印。** 这毫无意义，应在 CI 流程中一并集成检测器。
- **无校准的检测功能。** 基于 ASVspoof LA 数据训练的 AASIST 模型会出现过拟合现象，导致实际场景下的准确率下降；需在目标领域内进行校准。
- **音高偏移问题。** 过度的音高偏移会消除大部分水印，因此需要设置检测功能的备用方案。
- **元数据剥离与重新上传攻击。** 通过重新编码即可轻易绕过 C2PA 防护机制，必须同时采用加密防护与感知型（水印）防护措施。
- **以实时语音作为检测手段。** 可要求用户说出随机短语，此方法虽能防止重放攻击，却无法阻止实时克隆攻击。

## 发布它

将文件保存为 `outputs/skill-spoof-defender.md`。为语音生成部署选择检测模型、水印方案、来源证明清单以及运营操作手册。

## 练习题

1. **简单级。** 运行 `code/main.py`。使用简易检测器以及在合成音频上进行的简易水印嵌入/检测功能。
2. **中等级。** 安装 `audioseal`，将 16 位有效载荷嵌入到文本转语音的输出中，然后再进行解码。通过添加噪声来破坏音频质量，并测量比特恢复准确率。
3. **高级别。** 在 ASVspoof 2019 LA 数据集上对 RawNet2 或 AASIST 模型进行微调。测量 EER 值。在 F5-TTS 生成的保留测试集中进行测试，观察模型在面对未知数据时的检测性能下降情况。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| ASVspoof | 测试基准 | 每两年举办一次的挑战赛；2024年即为ASVspoof 5。 |
| CM（对策） | 检测器 | 分类器：用于区分真实语音与合成/转换后的语音。 |
| SASV | 说话人验证 + CM | 集成生物特征识别与欺骗检测功能的技术。 |
| AudioSeal | 元水印 | 局部化的16位数据载荷，其处理速度比WavMark快485倍。 |
| 比特恢复精度 | 水印存活率 | 攻击后能够恢复的载荷比特所占比例。 |
| C2PA | 出处清单 | 关于内容创建/作者身份的加密元数据。 |
| AASIST | 检测器系列 | 基于图注意力机制的最先进反欺骗技术。 |

## 延伸阅读

- [Todisco 等人 (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825) — 当前的基准测试。
- [Defossez 等人 (2024). AudioSeal](https://arxiv.org/abs/2401.17264) — 水印检测的默认方案。
- [Chen 等人 (2025). WaveVerify](https://arxiv.org/abs/2507.21150) — 用于时间攻击的 MoE 检测器。
- [Jung 等人 (2022). AASIST](https://arxiv.org/abs/2110.01200) — 当前最先进的检测框架。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf) — 稳定性评估工具。
- [C2PA 规范](https://c2pa.org/specifications/specifications/) — 来源声明格式规范。
