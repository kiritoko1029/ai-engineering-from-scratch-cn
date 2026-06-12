# 构建语音助手处理流程——第六阶段综合项目

> 汇集了第 01-11 课的所有内容。构建一个能够聆听、推理并回应的语音助手。在 2026 年，这已属于可解决的工程问题而非研究课题——但集成细节将决定该产品能否最终面世。

**类型：** 构建
**语言：** Python
**先修要求：** 第 6 阶段 · 04、05、06、07、11 课；第 11 阶段 · 09 课（函数调用）；第 14 阶段 · 01 课（智能体循环）
**时长：** 约 120 分钟

## 问题所在

构建一个端到端的智能助手：

1. 捕获麦克风输入（16 kHz 单声道）。
2. 检测用户语音的开始与结束位置。
3. 对流式语音进行转录。
4. 将转录结果传递给能够调用工具（计时器、天气查询、日历功能等）的大语言模型。
5. 将大语言模型的文本输出发送至文本到语音转换模块。
6. 向用户播放生成的音频。
7. 若用户在响应过程中中断，系统立即停止。

延迟目标：在笔记本电脑的 CPU 上，从用户说完话起，首个 TTS 音频字节应在 800 毫秒内生成。质量目标：不得遗漏任何单词，静默时段不得出现虚假字幕，不得存在声音克隆泄露现象，且不得发生提示注入攻击。

## 概念概述

![语音助手处理流程：麦克风 → 语音活动检测 → 文本转录 → 大语言模型及工具 → 文本转语音 → 扬声器](../assets/voice-assistant.svg)

### 七个组件

1. **音频采集。** 麦克风 → 16 kHz 单声道 → 20 ms 的数据块。通常在 Python 中使用 `sounddevice`，而在生产环境中则使用原生的 AudioUnit/ALSA/WASAPI。  
2. **VAD（第 11 课）。** 使用 Silero VAD，阈值设为 0.5，最小语音时长为 250 ms，静音延续时间为 500 ms。该模块会输出“开始”和“结束”的信号。  
3. **流式文本转语音（第 4-5 课）。** 可采用 Whisper-streaming、Parakeet-TDT 或 Deepgram Nova-3（API）等技术。生成部分内容与最终完整内容的转录结果。  
4. **结合工具调用的大型语言模型。** 使用 GPT-4o / Claude 3.5 / Gemini 2.5 Flash 等模型，并为相关工具定义 JSON 模式，同时实现令牌流处理。  
5. **流式文本转文字（第 7 课）。** 可选用速度最快的开源模型 Kokoro-82M 或商业产品 Cartesia Sonic。在生成 20 个大型语言模型令牌后开始进行文本转语音。  
6. **音频播放。** 通过扬声器输出；针对低带宽网络，采用 opus 编码以压缩数据量。  
7. **中断处理机制。** 若在文本转语音播放过程中 VAD 检测到语音结束信号，则立即停止播放，取消大型语言模型的运行，并重新启动文本转语音流程。

### 你将遇到的三种故障模式

1. **首词裁剪问题。** 语音活动检测（VAD）的启动时机过晚，导致用户的“hey”声被遗漏。应将起始阈值设为0.3，而非0.5。
2. **响应中途中断混淆问题。** 在用户中断后，大型语言模型仍会继续生成内容；助手的声音会盖过用户的话语。需建立从VAD到取消LLM生成的连接。
3. **静默幻觉问题。** Whisper在静音的热身帧上也会输出“Thanks for watching”这类文字。必须始终通过VAD进行过滤。

### 2026年生产环境参考技术栈

| 技术栈 | 延迟 | 许可证类型 | 备注 |
|-------|---------|-----------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 毫秒 | 商业 API | 2026 年行业默认方案 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 毫秒 | 大部分为开源 | 适合自行搭建 |
| Moshi（全双工） | 200-300 毫秒 | CC-BY 4.0 | 单模型；架构不同，见第 15 课 |
| Vapi / Retell（托管型） | 300-500 毫秒 | 商业 | 上线速度最快；定制化功能有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | 离线使用 | 开源 | 适用于隐私保护/边缘计算场景 |

## 构建它

### 步骤 1：分块进行麦克风音频捕获（伪代码）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步骤 2：基于VAD的门控音节捕获

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步骤 3：流式处理 STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步骤 4：在 LLM 循环内部调用工具

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步骤 5：中断处理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 使用它

请查看 `code/main.py`，其中包含一个可运行的模拟程序，它使用占位模型将所有七个组件连接起来，因此即便没有硬件也能看到处理流程的架构。如需实际实现，则需用以下模块替换占位模型：

- `silero-vad`（通过 `pip install silero-vad` 安装）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（如 `gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- 用于输入输出的 `sounddevice`

## 常见陷阱

- **永久记录个人身份信息。**在大多数司法管辖区，完整的音频内容均属于个人身份信息。需保留30天，并以加密方式存储。
- **禁止中途插话。**用户可能会打断对话，助手必须立即停止说话。
- **会阻塞进程的文本转语音功能。**同步式的文本转语音功能会占用事件循环资源，应使用异步方式或独立的线程。
- **缺乏工具调用错误处理机制。**当相关工具出现故障时，大语言模型需获取错误信息并尝试重试一次，之后再以优雅的方式降级处理。
- **过度严格的幻觉过滤机制。**如果过滤过于严格，助手会反复回答“我无法提供帮助”；如果过滤不够严格，则会导致其随意输出内容。需通过独立数据集进行校准。
- **不支持唤醒词功能。**持续监听模式存在隐私风险，应添加唤醒词检测机制（如 Porcupine 或 openWakeWord）。

## 发布它

将文件保存为 `outputs/skill-语音助手架构设计.md`。在预算、规模、语言及合规性约束条件下，制定完整的端到端技术规范。

## 练习题

1. **简单。** 运行 `code/main.py` 即可。该脚本使用占位模块端到端模拟完整的一轮处理流程，并输出各阶段的延迟时间。
2. **中等难度。** 用基于预录制的 `.wav` 文件的真实 Whisper 模型替换 STT 占位模块，随后测量词错误率（WER）及整体端到端延迟。
3. **高难度。** 需要添加工具调用功能：实现 `get_weather`（任意 API）和 `set_timer` 函数。让大语言模型通过这些工具处理请求，并验证当用户说出“设置5分钟计时器”时，正确的函数会被触发，且语音回复能予以确认。

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 轮次 | 用户与助手的来回交互 | 一段由 VAD 划分的用户语音 + 一次 LLM-TTS 生成的响应。 |
| 打断 | 中途插入 | 用户在助手说话时发言，导致助手停止输出。 |
| 唤醒词 | “Hey assistant” | 短关键词检测器；常用实现包括 Porcupine、Snowboy、openWakeWord。 |
| 结束点判定 | 轮次结束 | 通过 VAD 检测及最小静音时长判断用户已停止发言。 |
| 预录音缓冲区 | 语音前的缓冲音频 | 在 VAD 触发前保留 200-400 毫秒的音频，以避免首词被截断。 |
| 工具调用 | 函数调用 | LLM 生成 JSON 数据；运行时系统进行调度；结果反馈至循环中。 |

## 延伸阅读

- [LiveKit — 语音智能体快速入门](https://docs.livekit.io/agents/) — 生产级参考文档。
- [Pipecat — 语音智能体示例](https://github.com/pipecat-ai/pipecat) — 适合自主开发的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 受管的语音原生解决方案。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) — 全双工通信参考实现（第15课）。
- [Porcupine 唤醒词功能](https://picovoice.ai/products/porcupine/) — 唤醒词检测机制。
- [Anthropic — 工具使用指南](https://docs.anthropic.com/en/docs/build-with-claude/tool-use) — 大语言模型函数调用方法。
