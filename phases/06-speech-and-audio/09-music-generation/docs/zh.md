# 音乐生成——MusicGen、Stable Audio、Suno以及版权领域的巨大变革

> 2026年音乐生成领域：Suno v5与Udio v4在商业应用中占据主导地位；MusicGen、Stable Audio Open及ACE-Step则在开源领域处于领先位置。相关技术难题已基本得到解决。而法律问题（华纳音乐5亿美元的和解协议以及环球音乐集团的和解协议）在2025至2026年间重塑了该行业格局。

**类型：** 构建
**语言：** Python
**先修课程：** 第6阶段·02课（频谱图）、第4阶段·10课（扩散模型）
**时长：** 约75分钟

## 问题所在

文本 → 长度为30秒至4分钟的音乐片段，包含歌词、人声及完整结构。具体分为三个子问题：

1. **器乐生成。** 将“带有温暖键盘音色的低频嘻哈鼓点”等文本转换为音频。相关工具包括MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（含人声与歌词）。** 根据“关于德克萨斯州雨夜的乡村歌曲”这样的描述生成完整歌曲。相关工具包括Suno、Udio、YuE、ACE-Step。
3. **条件化/可控生成。** 对现有片段进行扩展、重新生成桥段、切换音乐风格、分离音轨或进行图像修复。Udio的图像修复与音轨分离功能是2026年计划推出的对标功能。

## 概念概述

![音乐生成：基于令牌的LM与扩散模型，2026年模型格局](../assets/music-generation.svg)

### 基于神经编解码器令牌的LM令牌

Meta的**MusicGen**（2023年，MIT开发）及其众多衍生模型：以文本/旋律嵌入作为条件输入，通过自回归方式预测EnCodec令牌（采样率为32 kHz，配备4个码本），随后使用EnCodec进行解码。参数量在3亿至33亿之间。虽可作为强大的基准模型，但在处理超过30秒长度的音频时表现不佳。

**ACE-Step**（开源项目，40亿参数的XL版本于2026年4月发布）在此基础之上实现了基于整首歌曲歌词的条件生成功能。它是当前开放社区中最接近Suno性能的模型。

### 基于 MEL 或潜变量的扩散模型

**Stable Audio (2023)** 与 **Stable Audio Open (2024)**：基于压缩音频的潜在扩散模型。擅长处理循环片段、声音设计以及环境音质感，但不太适合构建结构完整的歌曲。

**AudioLDM / AudioLDM2**：通过 T2I 风格的潜在扩散技术实现文本转音频，可应用于音乐、音效及语音生成。

### 混合模式（生产环境）——Suno、Udio、Lyria

封闭式权重结构。大概率采用基于AR编解码器的语言模型，搭配基于扩散模型的声码器，并配备专门处理人声、鼓声及旋律的模块。Suno v5（2026版）在ELO评分体系中以1293分的高分位居质量榜首。Udio v4则新增了图像修复功能以及音轨分离功能（可分别下载贝斯、鼓声和人声音频）。

### 评估

- **FAD（Fréchet音频距离）**：利用VGGish或PANNs特征计算生成的音频分布与真实音频分布之间的嵌入层距离。数值越低越好。MusicGen在小数据集上的FAD值为4.5（在MusicCaps上测试），当前最先进技术的该值约为3.0。
- **音乐性（主观评价）**：由人类偏好决定。Suno v5的ELO评分为1293，表现最佳。
- **文本与音频的对齐程度**：提示词与生成输出之间的CLAP得分。
- **音乐性缺陷**：节拍错位、声乐片段漂移，以及超过30秒后结构丧失的问题。

## 2026年型号列表

| 模型名称 | 参数量 | 音频长度 | 是否支持人声 | 许可协议 |
|-------|--------|----------|--------------|---------|
| MusicGen-large | 33亿 | 30秒 | 否 | MIT |
| Stable Audio Open | 12亿 | 47秒 | 否 | Stability非商业许可 |
| ACE-Step XL（2026年4月版本） | 40亿 | >2分钟 | 是 | Apache-2.0 |
| YuE | 70亿 | >2分钟 | 是，支持多语言 | Apache-2.0 |
| Suno v5（闭源） | ? | 4分钟 | 是，ELO评分1293 | 商业用途 |
| Udio v4（闭源） | ? | 4分钟 | 是，包含独立音轨 | 商业用途 |
| Google Lyria 3（闭源） | ? | 实时生成 | 是 | 商业用途 |
| MiniMax Music 2.5 | ? | 4分钟 | 是 | 商业API |

## 法律环境（2025–2026年）

- **Warner Music 与 Suno 的和解协议**。金额达 5 亿美元。WMG 现已获得对 Suno 上的 AI 风格内容、音乐版权以及用户生成的曲目的监管权。UMG 在 Udio 方面也达成了类似的和解协议。
- **欧盟《AI 法案》** + **加利福尼亚州 SB 942 法案**：必须公开标注 AI 生成的音乐。
- **MIT 开发的 Riffusion / MusicGen** 虽无合规问题，但也不支持商业用途的人声合成。

可安全使用的模式：

1. 仅生成器乐内容（使用 MusicGen、Stable Audio Open 以及 MIT/CC0 输出的音频）。
2. 使用带有按次生成许可的商业 API（如 Suno、Udio、ElevenLabs Music）。
3. 基于自有或已获授权的音乐曲库进行训练（大多数企业采用此方式）。
4. 为生成的音频添加水印及元数据标签。

## 构建它

### 步骤 1：使用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三种尺寸：`small`（300M，速度较快）、`medium`（1.5B）和`large`（3.3B）。对于初步验证想法是否可行而言，`small`尺寸已足够。

### 步骤 2：旋律条件化

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 可以输入色度图，在保持旋律不变的同时改变音色。非常适合将某段旋律转换为弦乐四重奏的演奏版本。

### 步骤 3：FAD 评估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

计算 VGGish 嵌入距离。适用于流派级别的回归测试；不能替代人类听音者的判断。

### 步骤 4：集成至 LLM-music 工作流

结合第7课至第8课中的内容：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 使用它

| 目标 | 技术栈 |
|------|-------|
| 乐器音效设计 | Stable Audio Open |
| 游戏/自适应音乐 | Google Lyria RealTime（闭源） |
| 含人声的完整歌曲（商业用途） | 需使用带有明确许可的 Suno v5 或 Udio v4 |
| 含人声的完整歌曲（开源用途） | ACE-Step XL 或 YuE |
| 短广告铃声 | 基于哼唱参考音生成的 MusicGen 旋律 |
| 音乐视频背景音乐 | MusicGen + Stable Video Diffusion |

## 2026年仍会存在的缺陷与隐患

- **版权清洗类提示词。** “泰勒·斯威夫特风格的歌曲”——商业版 Suno/Udio 现已能过滤此类内容，而开源模型则无法做到。可自行添加过滤规则列表。
- **重复问题 / 30 秒后出现偏差。** AR 模型会循环播放。可通过交叉淡入多代生成的内容来改善，或使用 ACE-Step 以保持结构连贯性。
- **节奏偏差。** 模型的节拍可能会偏离设定值。可在提示词中添加 BPM 标签，并在后期处理时使用 librosa 的 `beat_track` 函数进行调整。
- **人声清晰度。** Suno 的表现非常出色；而开源模型的人声往往发音含糊不清。如果歌词内容很重要，建议使用商业 API 或进行微调。
- **单声道输出。** 开源模型生成的音频为单声道或伪立体声。可通过合适的立体声重建技术（如 ezst、Cartesia 的立体声扩散模型）进行升级。

## 发布它

将文件保存为 `outputs/skill-music-designer.md`。为音乐生成部署选择模型、许可策略、长度/结构规划以及披露元数据。

## 练习题

1. **简单级。** 运行 `code/main.py` 即可。该脚本会生成由 ASCII 符号构成的“生成式”和弦进行与鼓点模式——相当于一个音乐生成的卡通图。如需播放，可使用任何 MIDI 渲染器。
2. **中等级。** 安装 `audiocraft`，使用 MusicGen-small 根据 4 种不同的风格提示生成 10 秒长的音频片段，并针对参考风格集计算 FAD 值。
3. **高级别。** 使用 ACE-Step（或 MusicGen-melody），根据不同的音色提示为同一曲调生成三种变体。随后计算这些变体与原始提示的 CLAP 相似度，以验证其一致性。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| FAD | Audio FID | 真实音频与生成音频嵌入分布之间的弗雷歇距离。 |
| Chromagram | 音高形式的旋律 | 每帧12维向量；用作旋律条件化的输入。 |
| Stems | 乐器音轨 | 分离出的贝斯/鼓声/人声/旋律，以WAV格式存在。 |
| Inpainting | 修复特定片段 | 对时间窗口进行遮罩处理；模型仅对该区域进行重建。 |
| CLAP | Text-audio CLIP | 对比式音频-文本嵌入；用于评估音频与文本的对齐程度。 |
| EnCodec | 音乐编解码器 | Meta开发的神经网络编解码器，被MusicGen所使用；采样率为32 kHz，包含4个代码本。 |

## 延伸阅读

- [Copet 等人 (2023). MusicGen](https://arxiv.org/abs/2306.05284) — 开源的自回归基准模型。
- [Evans 等人 (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358) — 音效设计的默认选择。
- [ACE-Step](https://github.com/ace-step/ACE-Step) — 开源的 4B 规模完整歌曲生成器，2026 年 4 月发布。
- [Suno v5 平台文档](https://suno.com) — 具有商业级质量的领先产品。
- [AudioLDM2](https://arxiv.org/abs/2308.05734) — 用于音乐及音效的潜在扩散模型。
- [WMG-Suno 和解案相关报道](https://www.musicbusinessworldwide.com/suno-warner-music-settlement/) — 2025 年 11 月的判例参考。
