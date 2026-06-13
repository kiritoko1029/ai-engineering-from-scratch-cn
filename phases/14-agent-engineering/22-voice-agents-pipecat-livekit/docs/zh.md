# 语音智能体：Pipecat 与 LiveKit

> 语音智能体是 2026 年的一线生产类别。Pipecat 为你提供基于帧的 Python 管道（VAD → STT → LLM → TTS → 传输层）。LiveKit Agents 通过 WebRTC 将 AI 模型连接到用户。高端技术栈的生产延迟目标为端到端 450–600ms。

**类型：** 学习
**语言：** Python（标准库）
**前置要求：** 第 14 阶段 · 01（智能体循环），第 14 阶段 · 12（工作流模式）
**所需时间：** 约60分钟

## 学习目标

- 描述 Pipecat 基于帧的管道：DOWNSTREAM（源→汇）和 UPSTREAM（控制）。
- 说出标准语音管道的各阶段以及 Pipecat 支持的传输方式。
- 解释 LiveKit Agents 的两类语音智能体（MultimodalAgent、VoicePipelineAgent）及其适用场景。
- 总结 2026 年的生产延迟预期以及它们如何驱动架构选择。

## 问题所在

语音智能体并非在文本循环上加个 TTS 就完事。延迟预算非常苛刻（约600ms），部分音频是常态，话轮检测是一个模型，传输方式涵盖从电话 SIP 到 WebRTC。你要么自己构建基于帧的管道（Pipecat），要么依赖一个平台（LiveKit）。

## 概念说明

### Pipecat（pipecat-ai/pipecat）

- Python 基于帧的管道框架。
- `Frame` → `FrameProcessor` 链。
- 两个流方向：
  - **DOWNSTREAM** — 源 → 汇（音频输入，TTS 输出）。
  - **UPSTREAM** — 反馈与控制（取消、指标、打断）。
- `PipelineTask` 通过事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）管理生命周期，并支持用于指标/追踪/RTVI 的观察者。

典型管道：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

传输方式：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 增加了结构化对话（状态机）。Pipecat Cloud 是托管运行时。

### LiveKit Agents（livekit/agents）

- 通过 WebRTC 将 AI 模型连接到用户。
- 核心概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 两类语音智能体：
  - **MultimodalAgent** — 通过 OpenAI Realtime 或等效服务实现的直接音频传输。
  - **VoicePipelineAgent** — STT → LLM → TTS 级联；提供文本级控制。
- 通过 Transformer 模型实现语义话轮检测。
- 原生 MCP 集成。
- 通过 SIP 支持电话。
- 通过 LiveKit Inference 提供 50+ 模型（无需 API 密钥）；通过插件再支持 200+ 模型。

### 商业平台

Vapi（在优化的高端技术栈上约450–600ms）和 Retell（180 次测试通话中端到端约600ms）构建在这些之上。当你想要一个没有 WebRTC 团队也能使用的托管语音技术栈时，选择平台即可。

### 这个模式出问题的地方

- **没有打断处理。** 用户打断时，智能体仍在说话。需要 Pipecat 中的 UPSTREAM 取消帧，LiveKit 中的等效机制。
- **忽略 STT 置信度。** 低置信度的转录文本被当作金科玉律送入 LLM。应基于置信度设门槛或请求确认。
- **TTS 中途截断。** 当管道在话语中途取消时，TTS 需要知晓或切断音频。
- **忽略延迟预算。** 每个组件增加 50–200ms。在发布前汇总你的整条链路。

### 2026 年典型延迟

- VAD：20–60ms
- STT 部分结果：100–250ms
- LLM 首个 token：150–400ms
- TTS 首段音频：100–200ms
- 传输层 RTT：30–80ms

端到端 450–600ms 属于高端水平。800–1200ms 是常见水平。超过 1500ms 会感觉很卡。

## 开始构建

`code/main.py` 是一个基于帧的演示管道，包含：

- `Frame` 类型（音频、转录、文本、tts_audio、控制）。
- `Processor` 接口，带有 `process(frame)` 方法。
- 五阶段管道（VAD → STT → LLM → TTS → 传输层），作为脚本化处理器。
- 用于演示打断的 UPSTREAM 取消帧。

运行方式：

```
python3 code/main.py
```

追踪日志展示了正常流程以及一个打断取消操作——它在 TTS 话语中途将其停止。

## 使用建议

- **Pipecat** — 完全控制，自定义处理器、Python 优先、可插拔提供商。
- **LiveKit Agents** — WebRTC 优先的部署和电话场景。
- **Vapi / Retell** — 无需 WebRTC 团队即可使用托管语音智能体。
- **OpenAI Realtime / Gemini Live** — 直接音频输入/输出（MultimodalAgent）。

## 交付产物

`outputs/skill-voice-pipeline.md` 搭建了一个 Pipecat 风格的语音管道，包含 VAD + STT + LLM + TTS + 传输层以及打断处理。

## 练习

1. 为你的演示管道添加一个指标观察者：统计每个阶段每秒的帧数。延迟累积在哪里？
2. 实现置信度门控 STT：低于阈值时，请求"您能重复一遍吗？"
3. 添加语义话轮检测：简单规则——如果转录以"?"结尾，则为话轮结束。
4. 阅读 Pipecat 的传输层文档。将标准库传输替换为 SmallWebRTCTransport 配置（存根）。
5. 对同一查询测量 OpenAI Realtime 与 STT+LLM+TTS 级联的延迟差异。文本级控制带来多少延迟开销？

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|---------|---------|
| Frame | "事件" | 管道中的类型化数据单元（音频、转录、文本、控制） |
| Processor | "管道阶段" | 带有 process(frame) 的处理器 |
| DOWNSTREAM | "前向流" | 从源到汇：音频输入，语音输出 |
| UPSTREAM | "反馈流" | 控制：取消、指标、打断 |
| VAD | "语音活动检测" | 检测用户何时在说话 |
| 语义话轮检测 | "智能话轮结束" | 基于模型的判断，确定用户已说完 |
| MultimodalAgent | "直接音频智能体" | 音频进、音频出；中间没有文本 |
| VoicePipelineAgent | "级联智能体" | STT + LLM + TTS；文本级控制 |

## 延伸阅读

- [Pipecat 文档](https://docs.pipecat.ai/getting-started/introduction) — 基于帧的管道、处理器、传输层
- [LiveKit Agents 文档](https://docs.livekit.io/agents/) — WebRTC + 语音原语
- [Vapi](https://vapi.ai/) — 托管语音平台
- [Retell AI](https://www.retellai.com/) — 托管语音、延迟基准测试
