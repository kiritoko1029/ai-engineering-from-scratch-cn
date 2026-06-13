# 毕业项目 03 — 实时语音助手（ASR 到 LLM 到 TTS）

> 一个体验良好的语音智能体，端到端延迟应在 800ms 以内，能感知用户何时停止说话，能处理打断，能在不中断音频的情况下调用工具。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都达到了这一水准。它们采用相同的架构：流式 ASR、轮次检测器、流式 LLM 和流式 TTS，全部通过 WebRTC 连接，在每个环节都有严格的延迟预算。构建一个这样的系统，衡量 WER、MOS 和误切率，并在丢包环境下运行。

**类型：** 毕业项目
**语言：** Python（智能体 + 流水线）、TypeScript（Web 客户端）
**前置要求：** 阶段 6（语音与音频）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 13（工具）、阶段 14（智能体）、阶段 17（基础设施）
**涉及阶段：** P6 · P7 · P11 · P13 · P14 · P17
**所需时间：** 30 小时

## 问题所在

语音是 2025-2026 年发展最快的 AI 用户体验类别。技术天花板每个季度都在降低。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70 都将 800ms 以内的首次音频输出变为现实。标准不仅仅是延迟。更重要的是交互体验：不打断用户、不被打断、能从句中打断中恢复、在对话中途调用工具而不中断音频、能承受不稳定的移动网络。

你无法通过拼接三个 REST 调用来实现这一点。架构必须是端到端的流水线式流处理。构建它之后，失败模式就会显现：为电话音频调优的 VAD 在背景电视声中误触发、轮次检测器等待永远不会出现的标点符号、TTS 在输出前缓冲了 400ms。本毕业项目的目标是在负载下逐一修复这些问题，并发布一份延迟与质量报告。

## 概念说明

流水线有五个流式阶段：**音频输入**（来自浏览器或 PSTN 的 WebRTC）、**ASR**（来自 Deepgram Nova-3 或 faster-whisper 的流式部分转录）、**轮次检测**（VAD 加一个小型轮次检测器模型，读取部分转录以判断完成度）、**LLM**（一旦判断轮次完成就流式输出 Token）、**TTS**（在首个 LLM Token 后约 200ms 内流式输出音频）。

三个横切关注点。**打断（Barge-in）**：当用户在智能体说话时开始发言，TTS 立即取消，ASR 立即开始接收。**工具使用**：对话中途的函数调用（天气、日历）必须在旁路上运行而不中断音频；如果延迟超过 300ms，智能体会预填充一个确认词（"稍等一下……"）。**背压（Backpressure）**：在丢包情况下，部分转录被暂缓，VAD 提高语音门限阈值，智能体避免在未确认的消息上继续发言。

衡量标准是定量的。在 15 dB SNR 的 Hamming VAD 基准下 WER 低于 8%。100 次实测通话的首次音频输出 p50 低于 800ms。误切率低于 3%。TTS 的 MOS 高于 4.2。单个 g5.xlarge 支持 50 个并发通话。这些数字就是交付成果。

## 架构

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技术栈

- 传输：LiveKit Agents 1.0（WebRTC）+ Twilio PSTN 网关；Pipecat 0.0.70 作为备选框架
- ASR：Deepgram Nova-3（流式，首次部分结果延迟低于 300ms）或自托管的 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 + LiveKit 轮次检测器（读取部分转录的小型 Transformer）
- LLM：OpenAI GPT-4o-realtime 用于紧密集成，Gemini 2.5 Flash Live，或级联的 Claude Haiku 4.5（流式补全，独立音频路径）
- TTS：Cartesia Sonic-2（最低首字节延迟）、ElevenLabs Flash v3，或开源的 Orpheus（自托管）
- 工具：FastMCP 旁路用于天气/日历/预约；工具耗时超过 300ms 时智能体预输出填充词
- 可观测性：OpenTelemetry 语音 Span，Langfuse 语音追踪含音频回放
- 部署：单个 g5.xlarge（24GB 显存）用于自托管 Whisper + Orpheus；托管 API 用于最低延迟

## 开始构建

1. **WebRTC 会话。** 搭建一个 LiveKit 房间和一个 Web 客户端来流式传输麦克风音频。在服务器端，接入一个加入房间的智能体工作进程。

2. **ASR 流式处理。** 将 20ms PCM 帧发送给 Deepgram Nova-3（或 GPU 上的 faster-whisper）。订阅部分和最终转录。记录每个部分结果的延迟。

3. **VAD 和轮次检测器。** 在帧流上运行 Silero VAD v5。在语音结束事件时，使用最新的部分转录触发 LiveKit 轮次检测器。只有当 VAD 判定静默持续 500ms 且轮次检测器的完成度评分超过 0.6 时，才确认"轮次完成"。

4. **LLM 流式处理。** 轮次完成后，使用累积的对话和最终转录启动 LLM 调用。流式输出 Token。在第一个 Token 时，交给 TTS。

5. **TTS 流式处理。** Cartesia Sonic-2 流式返回音频块。第一个音频块必须在首个 LLM Token 后 200ms 内离开服务器。将音频块发送到 LiveKit 房间；客户端通过 WebRTC 抖动缓冲区播放。

6. **打断处理。** 当 VAD 在 TTS 播放期间检测到新的用户语音时，立即取消 TTS 流，丢弃剩余的 LLM 输出，并重新启用 ASR。发布一个 `tts_canceled` Span。

7. **工具旁路。** 将天气和日历注册为函数调用工具。调用时并发执行；如果 300ms 内未完成，让 LLM 输出"稍等，让我查一下"作为填充；工具返回后继续。

8. **评估框架。** 录制 100 通话。计算 WER（与保留转录对比）、误切率（TTS 在用户说话中途被取消的次数）、首次音频输出 p50、TTS MOS（人工或 NISQA）以及丢包测试（丢弃 3% 的数据包）。

9. **负载测试。** 使用合成呼叫者在单个 g5.xlarge 上驱动 50 个并发通话。衡量持续的首次音频输出 p95。

## 使用示例

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] first token @960ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @1140ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付成果

`outputs/skill-voice-agent.md` 是交付成果。给定一个领域（客户服务、日程安排或自助终端），它会搭建一个 LiveKit 智能体，配备调优到衡量标准的 ASR/VAD/LLM/TTS 流水线。评分标准：

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 端到端延迟 | 100 次录音通话中首次音频输出 p50 低于 800ms |
| 20 | 轮次切换质量 | 在 Hamming VAD 基准上误切率低于 3% |
| 20 | 工具使用正确性 | 对话中途的工具调用返回正确数据且不中断音频 |
| 20 | 丢包下的可靠性 | 注入 3% 丢包后的 WER 和轮次切换稳定性 |
| 15 | 评估框架完整性 | 可复现的测量结果和公开的配置 |
| **100** | | |

## 练习

1. 在 g5.xlarge 上将 Deepgram Nova-3 替换为 faster-whisper v3 turbo。衡量延迟和 WER 差距。识别 CPU 与 GPU 决策在哪些地方产生影响。

2. 添加打断仲裁策略：当用户在工具调用期间打断时，智能体应如何处理？对比三种策略（硬取消、完成工具后停止、排队下一轮）。

3. 运行对抗性轮次检测器测试：给用户在句中设置长停顿。调优 VAD 静默阈值和轮次检测器评分阈值，以最低误切率且不超过 900ms 为目标。

4. 通过 Twilio 将同一智能体部署到 PSTN。对比 PSTN 和 WebRTC 的首次音频输出延迟。解释抖动缓冲区和编解码器的差异。

5. 添加非英语语言（日语、西班牙语）的语音活动检测。衡量 Silero VAD v5 相比语言特定微调版本的误触发率。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 轮次检测 | "话语结束" | 给定 VAD 静默和部分转录后，判断用户已说完的分类器 |
| 打断（Barge-in） | "打断处理" | 当 VAD 检测到新的用户语音时，取消正在播放的 TTS |
| 首次音频输出 | "延迟" | 从用户停止说话到第一个音频包离开服务器的时间 |
| VAD | "语音门限" | 将音频帧分类为语音或静默的模型；Silero VAD v5 是 2026 年的默认选择 |
| 抖动缓冲区 | "音频平滑" | 客户端缓冲区，短暂持有数据包以吸收网络抖动 |
| 填充词 | "确认词" | 智能体在工具响应慢时输出的短语，用于避免沉默 |
| MOS | "平均意见分" | 感知语音质量评分；NISQA 是自动化的代理指标 |

## 延伸阅读

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 参考 WebRTC 智能体框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 备选的 Python 优先流式智能体框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 集成语音模型参考
- [Deepgram Nova-3 文档](https://developers.deepgram.com/docs) — 流式 ASR 参考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 参考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延迟 TTS 参考
- [Retell AI 架构](https://docs.retellai.com) — 生产级语音智能体架构
- [Vapi.ai 生产技术栈](https://docs.vapi.ai) — 备选的生产参考
