# 语音活动检测与轮询机制 —— Silero、Cobra 以及“冲刷技巧”

> 每个语音助手的成败都取决于两个决策：用户当前是否正在说话，以及他们是否已经说完。VAD用于回答第一个问题；而转场检测（结合 VAD、静音残留效应及语义端点模型）则用于回答第二个问题。一旦其中任何一个判断出错，助手要么会突然中断用户的发言，要么就会不停地喋喋不休。

**类型：** 构建
**语言：** Python
**先修要求：** 第 6 阶段 · 11（实时音频处理）、第 6 阶段 · 12（语音助手）
**耗时：** 约 45 分钟

## 问题所在

语音代理在每20毫秒的时间片段内会做出三项独立的决策：

1. **当前帧是否为语音？** —— 语音活动检测（VAD）。采用二进制判定，针对每一帧独立处理。
2. **用户是否开始了新的发言？** —— 发言起始点检测。
3. **用户是否已结束发言？** —— 发言结束点检测。

基于能量阈值的简单方法在面对任何噪声——如交通声、键盘声或人群嘈杂声时都会失效。2026年的解决方案为：Silero VAD（开源的深度学习模型）+ 发言结束点检测模型（语义级结束点识别）+ 经过VAD校准的静默过渡处理机制。

## 概念概述

![VAD级联结构：能量检测 → Silero模型 → 话轮检测器 → 结束标记识别](../assets/vad-turn-taking.svg)

### 三层VAD级联结构

**第一级：能量门限检测。** 成本最低。阈值均方根值为 -40 dBFS。可过滤明显的静音，但会对超过该阈值的任何噪声触发响应。

**第二级：Silero VAD**（2020-2026年，麻省理工学院开发）。拥有100万个参数，基于6000多种语言进行训练。在单个CPU线程上，每30毫秒处理一次数据，耗时约1毫秒。在5%的假阳性率下，真正阳性率为87.7%。为开源项目的默认选择。

**第三级：语义话轮检测器。** 可使用LiveKit的话轮检测模型（2024-2026年）或自定义的小型分类器。能够区分“说话途中的停顿”与“已结束讲话”的状态，不仅依据静音情况，还结合语言语境（语调及近期词汇）进行判断。

### 关键参数及其默认值

- **阈值。** Silero 会输出一个概率值；当概率大于 0.5（默认值）或大于 0.3（敏感模式）时即对语音进行分类。阈值越低，首词片段数量越少，但误报率越高。
- **最小语音时长。** 长度小于 250 毫秒的语音将被拒绝处理——这类语音通常为咳嗽声或椅子移动的噪音。
- **静默残留问题（结束点判定）。** 在 VAD 指标返回 0 后，需等待 500–800 毫秒才能判定对话结束。等待时间过短会导致打断用户；时间过长则会让系统响应显得迟缓。
- **预播放缓冲区。** 在 VAD 触发前保留 300–500 毫秒的音频内容，以避免将“hey”等开头词汇截断。

### “刷新技巧”（Kyutai 2025）

流式语音转文字模型存在前瞻延迟（Kyutai STT-1B为500毫秒，STT-2.6B为2.5秒）。通常需要在语音结束后的这段时间内等待才能获得文本转录结果。解决此问题的技巧是：当语音活动检测器检测到语音结束时，**向语音转文字模型发送强制刷新信号**，以促使其立即输出结果。由于该模型的处理速度约为实时速度的4倍，因此500毫秒的缓冲时间实际上仅需约125毫秒即可完成处理。

端到端总延迟：125毫秒的语音活动检测时间 + 强制刷新后的语音转文字处理时间 = 对话延迟。

### 2026年VAD技术对比

| VAD | 5%误报率下的TPR | 延迟时间 | 许可证 |
|-----|--------------|---------|---------|
| WebRTC VAD（Google，2013年） | 50.0% | 30 毫秒 | BSD |
| Silero VAD（2020–2026年） | 87.7% | 约1 毫秒 | MIT |
| Cobra VAD（Picovoice） | 98.9% | 约1 毫秒 | 商业许可 |
| pyannote segmentation | 95% | 约10 毫秒 | 类MIT许可 |

Silero是理想的默认选择。Cobra则用于提升合规性及准确率。在2026年的生产环境中，仅基于能量检测的VAD已不再适用。

## 构建它

### 步骤 1：能量门

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步骤 2：使用 Python 实现 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步骤 3：转向-结束状态机

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步骤 4：flush 技巧框架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必须支持刷新功能才能正常工作。而 Whisper 流式处理则不支持——它采用分块方式，始终需要等待数据块完成传输。

## 使用它

| 场景 | 选用的 VAD 工具 |
|-----------|----------------|
| 开源、快速、通用型 | Silero VAD |
| 商业电话中心 | Cobra VAD |
| 设备端（手机） | Silero VAD ONNX |
| 研究/语音分段 | pyannote segmentation |
| 无依赖的备用方案 | WebRTC VAD（旧版） |
| 需要高精度的语音段落结束检测 | Silero + LiveKit 语音段落检测器组合使用 |

经验法则：除非实在别无选择，否则切勿部署仅基于能量值的 VAD 工具。

## 常见陷阱

- **固定阈值问题。**在安静环境下正常工作，但在嘈杂环境中失效。需在设备端进行校准或改用 Silero。
- **静音过渡时间过短。**智能体会在句子中间中断对话。对于对话式语音，500-800 毫秒是最佳时间间隔。
- **静音过渡时间过长。**会导致响应迟缓。需与目标用户进行 A/B 测试。
- **缺乏预滚动缓冲区。**用户的音频前 200-300 毫秒内容会丢失。应始终保留动态预滚动缓冲区。
- **忽略语义端点识别。**诸如“嗯，让我想想...”这类表达包含较长的停顿。用户不喜欢在思考中途被打断。建议使用 LiveKit 的轮次检测功能或类似工具。

## 发布它

将文件保存为 `outputs/skill-vad-tuner.md`。针对特定工作负载，选择合适的VAD模型、阈值、滞后时间、预滚动时长以及转句检测策略。

## 练习题

1. **简单级。** 运行 `code/main.py`。该脚本模拟“语音 + 沉默 + 语音 + 咳嗽”的序列，并测试三种不同级别的 VAD 算法。
2. **中等级。** 安装 `silero-vad`，处理一段 5 分钟的录音，调整阈值以尽量减少首词被截取的情况以及误触发现象。需输出精确度与召回率数据。
3. **高级别。** 构建一个简易的对话轮次检测器：结合 Silero VAD 以及在最后 10 个词的嵌入向量上运行的三层 MLP（使用 sentence-transformers 库）。在经过手动标注的对话结束点数据集上进行训练，使其 F1 分数比仅使用 Silero VAD 的方案高出 10%。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| VAD | 语音检测器 | 每帧二进制判断：当前是否为语音？ |
| 话轮检测 | 结束点定位 | VAD + 语音后静默期 + 语义结束点。 |
| 语音后静默期 | 语音结束后等待时间 | 在判定话轮结束前需要等待的时间；通常为500-800毫秒。 |
| 预录音段 | 语音前缓冲区 | 在VAD触发前保留300-500毫秒的音频。 |
| 快速处理技巧 | Kyutai优化方法 | 通过VAD → 快速STT的流程，将延迟从500毫秒缩短至125毫秒。 |
| 语义结束点 | “对方是否真的想停止？” | 基于机器学习的分类器，不仅考虑静默状态，还会分析词汇内容。 |
| 在FPR为5%时的TPR | ROC曲线上的点 | 标准的VAD性能评估指标；Silero的该值为87.7%，WebRTC为50%。 |

## 延伸阅读

- [Silero VAD](https://github.com/snakers4/silero-vad) — 参考级的开源语音活动检测工具。  
- [Picovoice Cobra VAD](https://picovoice.ai/products/cobra/) — 行业领先的商业级语音活动检测方案。  
- [Kyutai — 无声段消除与数据刷新技巧](https://kyutai.org/stt) — 实现低于200毫秒处理速度的工程技巧。  
- [LiveKit — 话轮检测功能](https://docs.livekit.io/agents/logic/turns/) — 生产环境中的语义级话轮识别方案。  
- [WebRTC VAD](https://webrtc.googlesource.com/src/) — 传统的基准语音活动检测实现。  
- [pyannote 分割工具](https://github.com/pyannote/pyannote-audio) — 达到语音转写级别精度的分割功能。
