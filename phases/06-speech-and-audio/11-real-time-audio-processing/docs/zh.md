# 实时音频处理

> 批处理管道用于处理单个文件。而实时管道则会在下一个数据到达前的 20 毫秒内处理当前数据。所有的对话式 AI、广播演播室系统以及电话机器人，其性能优劣皆取决于这一延迟预算。

**类型：** 构建
**语言：** Python
**先修课程：** 第 6 阶段 · 02（频谱图）、第 6 阶段 · 04（语音识别）、第 6 阶段 · 07（文本转语音）
**时长：** 约 75 分钟

## 问题所在

您希望拥有一种具有“生命力”的语音助手。人类对话中的轮转延迟通常在约230毫秒左右（从静默到回应）。若延迟超过500毫秒，会给人以机械感；而超过1500毫秒则意味着系统出现故障。2026年实现完整的**监听 → 理解 → 响应 → 说话**循环的预算分配如下：

| 阶段 | 预算 |
|-------|------|
| 麦克风 → 缓冲区 | 20 毫秒 |
| VAD | 10 毫秒 |
| 流式ASR | 150 毫秒 |
| LLM（首个token生成） | 100 毫秒 |
| TTS（首个语音片段生成） | 100 毫秒 |
| 合成结果 → 扬声器 | 20 毫秒 |
| **总计** | **约400毫秒** |

Moshi（Kyutai，2024年发布）的全双工延迟为200毫秒。GPT-4o-realtime（2024年版本）的延迟约为320毫秒。2022年推出的级联式处理流程的延迟则为2500毫秒。实现10倍性能提升主要得益于三种技术：(1) 全流程流式处理，(2) 基于部分结果的异步流水线处理，(3) 可中断的生成机制。

## 概念概述

![带有环形缓冲区、VAD门控及中断功能的流式音频处理流程图，来源：../assets/real-time.svg]

**帧 / 数据块 / 窗口。** 实时音频以固定大小的块形式传输。常见选择为 20 毫秒（16 kHz 下的 320 个采样值）。所有下游组件都必须保持与此节奏一致。

**环形缓冲区。** 一种固定大小的循环缓冲区。生产者线程负责写入新帧，消费者线程负责读取。这可避免在高频处理路径中进行内存分配操作。其大小通常约为最大延迟时间乘以采样率；例如 16 kHz、2 秒的环形缓冲区可容纳 32,000 个采样值。

**VAD（语音活动检测）。** 当没有人在说话时，该功能会关闭下游组件的工作。Silero VAD 4.0（2024 年版本）在 CPU 上处理每 30 毫秒一个帧时的耗时低于 1 毫秒。`webrtcvad` 是较早的替代方案。

**流式 ASR。** 这类模型会在音频数据到达时立即输出部分转录结果。NeMo 在 2024 年推出的 Parakeet-CTC-0.6B 流式版本在 320 毫秒的延迟下，词错误率仅为 2–5%。Macháček 等人在 2023 年提出的 Whisper-Streaming 方法则通过分块处理方式实现近乎实时的转录效果，延迟约为 2 秒。

**中断处理。** 当用户在助手正在说话时插话，系统必须 (a) 检测到插入行为，(b) 停止文本转语音功能，(c) 弃用剩余的 LLM 输出结果。所有操作都必须在 100 毫秒内完成，否则用户会感觉助手“听不见”。

**WebRTC Opus 编码传输。** 使用 20 毫秒的帧长、48 kHz 的采样率，自适应比特率为 8–128 kbps。这是浏览器和移动端应用的标准传输格式。LiveKit、Daily.co 和 Pion 是 2026 年用于构建语音应用的常用技术框架。

**抖动缓冲区。** 网络数据包可能会乱序到达或延迟送达。抖动缓冲区负责对这些数据进行重新排序并平滑处理；若其大小过小会导致可听见的间隙，过大则会造成额外的延迟。通常设置为 60–80 毫秒。

### 常见陷阱

- **线程竞争。** Python 的 GIL 加上大型模型可能会阻塞音频线程。建议使用 C 语言回调式的音频库（如 sounddevice、PortAudio），避免让 Python 参与高频执行的路径。
- **采样率转换延迟。** 在处理流程中进行的重采样会带来 5–20 毫秒的延迟。可以在前期完成重采样，或者使用零延迟的重采样器（如 PolyPhase、`soxr_hq`）。
- **TTS 启动准备时间。** 即使是速度较快的 TTS 引擎如 Kokoro，在首次请求时也需要 100–200 毫秒的预热时间。应在第一次实际对话之前，先缓存模型并通过模拟运行对其进行预热。
- **回声消除。** 如果没有 AEC，TTS 输出会重新进入麦克风，从而触发机器人对自己声音的 ASR 处理。WebRTC AEC3 是默认的开源方案。

```figure
nyquist-aliasing
```

## 构建它

### 步骤 1：环形缓冲区

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量决定了最大缓冲延迟。16 kHz频率下32,000个样本的缓冲时间为2秒。

### 步骤 2：VAD 门控

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

在生产环境中，替换为 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步骤 3：流式语音识别

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步骤 4：中断处理程序

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

其实现依赖于异步 I/O 与可取消的 TTS 流式传输。在音频轨道上调用 WebRTC 的 peerconnection.stop() 是标准做法。

## 使用它

2026年技术栈：

| 层级 | 选择方案 |
|-------|----------|
| 传输层 | LiveKit（WebRTC）或 Pion（Go） |
| 语音活动检测 | Silero VAD 4.0 |
| 流式语音识别 | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| 大语言模型首个token生成 | Groq、Cerebras、vLLM-streaming |
| 流式文本转语音 | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回声消除 | WebRTC AEC3 |
| 纯端到端方案 | OpenAI Realtime API 或 Moshi |

## 常见陷阱

- **为确保稳定性，缓冲时间设为500毫秒。**该缓冲区实际上就是延迟的下限，应尽量缩小其大小。
- **未固定线程。**在优先级低于UI的线程上处理音频回调会导致在高负载情况下出现故障。
- **TTS数据块过小。**小于200毫秒的数据块会产生可听见的声码器伪影，320毫秒的数据块为最佳选择。
- **无抖动缓冲区。**实际网络存在抖动现象；若不进行平滑处理，就会产生音调突变。
- **单次错误处理机制。**音频处理流程必须具备抗崩溃能力，任何一次异常都可能导致整个会话中断。

## 发布它

将文件保存为 `outputs/skill-realtime-designer.md`。设计一个实时音频处理流水线，并为每个阶段设定具体的延迟预算。

## 练习题

1. **简单级。** 运行 `code/main.py`。该脚本模拟环形缓冲区与能量基音检测功能；针对一个虚拟的10秒长流媒体数据，输出各处理阶段的延迟时间。
2. **中等级。** 使用 `sounddevice` 构建一个直通循环，以20毫秒为帧长处理麦克风输入，并在每一帧时输出基音检测状态。
3. **高级别。** 利用 `aiortc` 构建完整的双向回声测试环境：浏览器 → WebRTC → Python脚本 → WebRTC → 浏览器。通过1千赫兹的脉冲信号，测量端到端的延迟时间。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 环形缓冲区 | 循环队列 | 用于音频帧的固定大小、无锁（或 SPSC 锁定）的 FIFO 结构。 |
| VAD | 静音检测器 | 通过模型或启发式方法区分语音与非语音信号。 |
| 流式 ASR | 实时文本转录 | 在音频数据到达时即时输出部分文本，具有有限的预测范围。 |
| 抖动缓冲区 | 网络平滑器 | 用于重新排序乱序的数据包；典型延迟为 60–80 毫秒。 |
| AEC | 回声消除 | 用于抵消从说话者到麦克风的反馈信号。 |
| 用户插话 | 用户中断 | 系统在文本转语音过程中检测到用户讲话，必须立即停止播放。 |
| 全双工 | 双向同时通信 | 用户与机器人可以同时进行对话；Moshi 即为全双工系统。 |

## 延伸阅读

- [Macháček 等人 (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743) — 分块式近流式 Whisper 技术。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf) — 双向通信，延迟为 200 毫秒。
- [LiveKit Agents 框架 (2024)](https://docs.livekit.io/agents/) — 生产环境音频智能体编排工具。
- [Silero VAD 代码库](https://github.com/snakers4/silero-vad) — 延迟低于 1 毫秒的 VAD 功能，采用 Apache 2.0 许可证。
- [WebRTC AEC3 相关论文](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/) — 开源环境下的回声消除技术。
