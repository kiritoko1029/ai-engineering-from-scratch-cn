# 流式语音转语音技术——Moshi、Hibiki与全双工对话系统

> 2024-2026年重新定义了语音人工智能。Moshi推出了一款能够同时实现监听与说话功能的单一模型，延迟仅为200毫秒。Hibiki则采用逐块处理的方式实现语音到语音的翻译。这两款产品均摒弃了ASR → LLM → TTS的处理流程，转而通过Mimi编解码器令牌构建统一的全双工架构。这便是新的参考设计。

**类型：** 学习
**语言：** Python
**先修课程：** 第6阶段·13（神经音频编解码器）、第6阶段·11（实时音频）、第7阶段·05（全Transformer模型）
**时长：** 约75分钟

## 问题所在

基于第 11 课与第 12 课内容构建的每个语音智能体，其基础延迟下限通常在 300-500 毫秒之间：首先触发语音活动检测（VAD），接着进行文本转录（STT）处理，随后由大语言模型（LLM）进行分析推理，最后通过文本转语音（TTS）生成输出。每个阶段都有其自身的最低延迟要求。虽然可以通过优化配置及并行化手段来降低延迟，但整个处理流程的结构本身仍会限制最终性能。

Moshi（Kyutai，2024-2026 年项目）提出了一个不同的问题：如果不存在这样的处理流程会怎样？假如仅使用一个模型直接连续地接收音频并输出音频，而将文本视为中间层的“内心独白”，而非必需的处理步骤，又会如何？

其答案便是**全双工语音转语音技术**。理论上的延迟为 160 毫秒（80 毫秒为 Mimi 框架处理时间，80 毫秒为声学延迟）。在单块 L4 级别的 GPU 上，实际延迟约为 200 毫秒。这一数值仅为目前最优秀的流水线式语音智能体所达到的延迟水平的一半。

## 概念概述

![Moshi 架构：两条并行的 Mimi 流 + 内心独白文本](../assets/moshi-hibiki.svg)

### Moshi 架构

**输入。** 两个 Mimi 编码流，均采用 12.5 Hz × 8 个代码本：

- 流 1：用户音频（经过 Mimi 编码，持续输入）
- 流 2：Moshi 自身的音频（由 Moshi 生成）

**变换器。** 一个包含 70 亿参数的时序变换器用于处理这两个流以及一条文本“内心独白”流。在每 80 毫秒的时间步中，它执行以下操作：

1. 接收最新的用户 Mimi 令牌（来自 8 个代码本）。
2. 接收最新生成的 Moshi Mimi 令牌（同样来自 8 个代码本）。
3. 生成下一个 Moshi 文本令牌（即内心独白内容）。
4. 通过一个小型深度变换器，基于前序结果生成下一组 Moshi Mimi 令牌（仍使用 8 个代码本）。

用户音频、Moshi 音频以及 Moshi 文本这三条流同时并行处理。Moshi 能在说话时听到用户的声音；当用户打断时它可以自行中断当前发言；还能在不中断主要话语的情况下发出回应声（如“嗯”）。 

**深度变换器。** 在单个帧内，这 8 个代码本并非并行预测——它们之间存在代码本间的依赖关系。一个小型的两层“深度变换器”会在 80 毫秒的时间窗口内依次对这些代码本进行预测。这是 AR 编码器语言模型所采用的标准分解方式（VALL-E 和 VibeVoice 也使用相同机制）。

### 为何内心独白文本有助于学习

在没有显式文本的情况下，模型必须在其音频流中隐式地对语言进行建模。Moshi 的创新思路在于：强制模型在输出音频的同时也生成文本标记。该文本流实际上就是 Moshi 所说内容的转录稿。这不仅提升了语义连贯性，还便于更换不同的语言模型头，并能免费提供转录内容。

### Hibiki：流式语音转语音翻译引擎

采用相同的架构，通过翻译对进行训练。输入源语言音频，输出目标语言音频，实现持续处理。Hibiki-Zero（2026年2月版本）无需词级对齐的训练数据——它利用句子级数据结合GRPO强化学习来实现延迟优化。

初始支持四种语言对；仅需约1000小时的训练时间即可适配新的语言对。

### 更广泛的 Kyutai 技术栈（2026 版）

- **Moshi** — 双向对话功能（以法语为主，英语支持良好）
- **Hibiki / Hibiki-Zero** — 同时语音翻译
- **Kyutai STT** — 流式语音识别技术（具有500毫秒或2.5秒的预测能力）
- **Kyutai Pocket TTS** — 基于CPU运行的1亿参数级文本转语音引擎（预计2026年1月推出）
- **Unmute** — 在公共服务器上整合上述各项技术的完整处理流程

在L40S GPU上的处理能力：可同时支持64个会话，处理速度为实时速度的3倍。

### Sesame CSM — 其表亲版本

Sesame CSM（2025版）采用了类似的设计思路——以Llama-3作为核心模型，再搭配Mimi编解码器头部。但CSM为单向处理模式（输入上下文与文本并生成语音），而非全双工模式。它是目前市场上具备最佳“声音表现力”的文本转语音技术；虽不及Moshi的全双工功能。

### 2026年的性能指标数据

| 模型 | 延迟 | 应用场景 | 许可协议 |
|-------|---------|----------|---------|
| Moshi | 200 毫秒（L4层） | 英语/法语全双工对话 | CC-BY 4.0 |
| Hibiki | 12.5 赫兹的帧率 | 法语 ↔ 英语流式翻译 | CC-BY 4.0 |
| Hibiki-Zero | 同上 | 支持5种语言对，无对齐数据 | CC-BY 4.0 |
| Sesame CSM-1B | 200 毫秒的TTFA时间 | 基于上下文的文本转语音技术 | Apache-2.0 |
| GPT-4o Realtime | 约300毫秒 | 闭源，基于OpenAI API | 商业用途 |
| Gemini 2.5 Live | 约350毫秒 | 闭源，基于Google API | 商业用途 |

## 构建它

### 步骤 1：接口

Moshi 提供了一个 WebSocket 服务器，该服务器持续不断地接收以 Mimi 编码格式传输的 80 毫秒长度的音频数据，并返回同样格式的 80 毫秒长度的音频数据。双向传输均保持此频率。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步骤 2：全双工循环

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

两个方向同时运行。Python 的 asyncio 或 Rust 的 futures 是标准的传输机制。

### 步骤 3：训练目标（概念层面）

对于每个 80 毫秒的帧 `t`：

- 输入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 输出预测：首先预测 `moshi_text[t]`，随后在深度变换器中按码本顺序依次预测 `moshi_mimi[t, codebook_0..7]`

文本会在音频之前被预测（即内心独白）；而音频则是在深度变换器内部按照码本顺序进行预测。

### 步骤 4：Moshi 的优势与不足之处

Moshi 的优势：

- 在低成本硬件上实现低于 250 毫秒的端到端响应时间。
- 支持自然的反向通道交互与中断处理。
- 无需额外的流水线粘合代码。

Moshi 的不足之处：

- 不支持工具调用功能（未针对该功能进行训练；需使用独立的 LLM 路径）。
- 缺乏长文本推理能力（Moshi 属于约 80 亿参数的对话模型，不同于 Claude/GPT-4）。
- 在小众主题上的事实准确性较差。
- 不适用于大多数企业级生产环境场景（2026 年仍以传统流水线为主）。

## 使用它

| 场景 | 推荐产品 |
|-----------|----------|
| 最低延迟的语音伴侣 | Moshi |
| 实时翻译通话 | Hibiki |
| 语音演示/研究 | Moshi、CSM |
| 配备工具的企业级智能体 | Pipeline（第12课），而非 Moshi |
| 基于上下文的自定义语音 TTS | Sesame CSM |
| 支持任意语言的口语转文字 | GPT-4o Realtime 或 Gemini 2.5 Live（商业版） |

## 常见陷阱

- **工具调用能力有限。** Moshi 是一个对话模型，而非代理框架，需结合流水线来使用各类工具。
- **特定语音条件控制。** Moshi 仅使用一个已训练好的角色形象；若要克隆其他角色，则需要单独进行训练。
- **语言支持范围。** 法语和英语的支持表现优异，其他语言的支持则较为有限。虽然 Hibiki-Zero 能提供一定帮助，但仍需相应的训练数据。
- **资源成本较高。** 一次完整的 Moshi 会话会占用一个 GPU 资源；这并非一种成本低廉的共享租户部署方案。

## 发布它

将文件保存为 `outputs/skill-duplex-pipeline.md`。针对语音智能体任务，选择单向流水线架构还是全双工架构，并说明理由。

## 练习题

1. **简单。** 运行 `code/main.py` 即可。该脚本通过符号化方式模拟双流 + 内心独白架构。
2. **中等难度。** 从 HuggingFace 下载 Moshi，启动其服务器，并测试一次对话。测量从用户停止说话到 Moshi 开始响应之间的实际延迟时间。
3. **高难度。** 使用你在第12课中设计的流水线代理，在20条匹配的测试语句上对比 P50 延迟与 Moshi 的延迟表现。并分析在何种架构条件下该流水线仍具有优势。

## 关键术语

| 术语 | 人们通常的说法 | 实际含义 |
|------|-----------------|----------|
| 全双工 | 同时收听与说话 | 在同一模型上同时启用两个音频流。 |
| 内心独白 | 模型的文本流 | Moshi 在输出音频的同时也会生成文本标记。 |
| 深度变换器 | 代码本间预测器 | 一种小型变换器，可在 80 毫秒的时间帧内预测 8 个代码本。 |
| Mimi | Kyutai 的编解码器 | 12.5 Hz × 8 个代码本；结合语义与声学特征；为 Moshi 提供支持。 |
| 流式端到端翻译 | 音频 → 音频实时转换 | 按块进行翻译或对话处理，无需多个处理阶段。 |
| 反向通道反馈 | “嗯”之类的回应 | Moshi 能在不停止当前发言的情况下发出简单的确认音。 |

## 延伸阅读

- [Défossez 等人 (2024)。Moshi — 语音转文本基础模型](https://arxiv.org/html/2410.00037v2) — 相关论文。
- [Kyutai Labs (2026)。Hibiki-Zero](https://arxiv.org/abs/2602.12345) — 无需对齐数据的流式翻译技术。
- [Sesame (2025)。跨越语音的“恐怖谷”](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice) — CSM 规范文档。
- [Kyutai — Moshi 代码仓库](https://github.com/kyutai-labs/moshi) — 安装指南及服务器配置。
- [OpenAI — 实时 API](https://platform.openai.com/docs/guides/realtime) — 商业化的同类封闭服务。
- [Kyutai — 延迟流处理建模](https://github.com/kyutai-labs/delayed-streams-modeling) — 底层的语音转文本/文本转语音框架。
