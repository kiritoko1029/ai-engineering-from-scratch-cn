# 全能模型：Qwen2.5-Omni 与 Thinker-Talker 架构

> GPT-4o 在 2024 年 5 月的产品演示之所以具有颠覆性，不是因为底层模型，而是因为产品形态——一个语音界面，你说话，模型看到摄像头所见，并在 250ms 内语音回应。开放生态在 2024 年余下时间和 2025 年都在追赶这一产品形态。Qwen2.5-Omni（2025 年 3 月）是参考性的开放设计：一个 Thinker（大型文本生成 Transformer）加上一个 Talker（并行语音生成 Transformer），通过流式语音 token 连接。Mini-Omni 对其进行了简化，Moshi 匹配了其延迟，GLM-4-Voice 将其扩展到中文。本课将解读 Thinker-Talker 架构以及实现实时流式对话的延迟预算。

**类型：** 构建
**语言：** Python（标准库，流式管道延迟模拟器 + VAD 循环）
**前置要求：** 第 12 阶段 · 第 19 课（音频 LLM），第 12 阶段 · 第 16 课（任意到任意）
**所需时间：** 约180分钟

## 学习目标

- 将推理管道拆分为 Thinker（文本推理）和 Talker（语音合成），解释为什么并行流式有效。
- 逐组件计算对话交互的首音频字节时间（TTFAB）预算。
- 描述 TMRoPE 在 Thinker 内部跨视觉、音频和文本的时间对齐位置编码。
- 列出三种实时对话模式：半双工、轮流对话、全双工。

## 问题所在

实时语音助手需要快速完成很多事情：

1. 听用户说话。实时语音 token 化，语音活动检测（VAD）以判断用户何时说完。
2. 可选地看。2-4 FPS 的摄像头输入，与音频一起流入 Thinker。
3. 思考。根据对话历史组合回复。
4. 说话。合成语音 token，解码为波形，流式输出到用户扬声器。

每一步都增加延迟。对话感觉需要总往返延迟 < 500ms——低于此值，用户不再注意到延迟。GPT-4o 声称约 250ms。Moshi 约 160ms。Qwen2.5-Omni 约 350-500ms。

每个组件都需要流式处理。不能"先批量处理再解码"。

## 概念说明

### Thinker 和 Talker

Qwen2.5-Omni 的分解：

- Thinker：一个 7B-80B 的文本生成 Transformer。消费交错的文本 + 图像 + 音频 token。输出代表"说什么"的文本 token。
- Talker：一个较小的语音生成 Transformer（200M-1B）。消费 Thinker 的文本输出 token 加上近期的语音上下文 token。输出离散语音 token（残差 VQ 索引）。
- 语音解码器：流式波形解码器（SNAC、MoVQGAN 系列），将语音 token 实时转换为音频样本。

分离的意义在于：Thinker 必须足够大以获得良好的推理能力。Talker 可以很小，因为其工作是局部的——将文本转换为语音 token。更大的 Talker 并不会更有表现力；只会更慢。

两者并行运行：

1. Thinker 输出文本 token t_i。
2. Talker 通过流式消费 t_i，输出语音 token s_i、s_{i+1}、...、s_{i+k}。
3. 语音解码器在语音 token 到达时消费它们，输出音频样本。
4. 当 Thinker 到达文本 token t_{i+3} 时，Talker 已经流式输出了 t_0..t_{i+2} 的音频。

### TMRoPE——时间对齐的多模态位置

Thinker 需要整合图像帧（比如 4 FPS 到达）、音频帧（50 帧/秒到达）和对话历史中的文本。朴素的序列顺序（所有图像，然后所有音频，然后文本）会丢失时间对齐。

TMRoPE 为每个 token 分配绝对时间戳。视觉 token 在 t=2.3s。音频 token 在 t=2.32s。用户文本 token "stop" 在 t=2.35s。RoPE 根据时间戳旋转注意力；模型将它们视为时间上并发的。

这是"他一边挥手一边说你好"能够工作的基础设施——模型在相同的概念时刻看到视频帧和音频。

### 流式语音合成

语音 token 必须流式输出。Mini-Omni（Xie & Wu，2024）引入了"语言模型能听、能在思考时说话"的流式模式：Thinker 输出 token 和 Talker 输出 token 在同一序列中交错。Talker 在 Thinker 提交下一个文本 token 后立即触发。没有批量边界。

Moshi（Défossez 等人，2024 年 10 月）是最快的开放实现。在单个 A100 上 160ms TTFAB。架构：单个 7B Transformer 在交替位置输出文本和语音 token，通过"内心独白"将思考流与说话流分离。这实际上是将 Thinker + Talker 融合到一个模型中，通过精心训练实现。

### VAD 和轮流对话

语音活动检测在输入端运行。两种模式：

- 半双工：用户说话，模型倾听。模型说话，用户倾听。通过 VAD 静音检测（约 200ms）实现清晰切换。
- 全双工：双方可以同时说话。模型可以进行反馈（"嗯嗯"）或打断。难度大得多。Moshi 支持此模式。

Qwen2.5-Omni 默认支持半双工，通过静音阈值实现轮流对话。全双工需要应用层处理。

### Qwen3-Omni（2025 年 11 月）

后续版本。Qwen3-80B Thinker，更大的 Talker，改进的 TMRoPE-v2。延迟接近 GPT-4o 的 250ms。开放权重。OmniBench 基准测试与 Gemini 2.0 Live 竞争力强。

### 生产延迟预算

典型流式交互：

- 麦克风 → 音频 token：40-80ms。
- 预填充（提示 + 历史）：7B 上 100-200ms，70B 上更多。
- 第一个 Thinker 文本 token：40ms。
- Talker 处理第一个文本 token：20ms。
- 第一批语音 token 提交：40ms。
- 残差 VQ 解码：30ms。
- 语音波形解码：50-80ms。

总 TTFAB：7B 上 320-510ms，70B 上 600-900ms。前沿质量通常意味着 70B+；因此存在前沿延迟差距。

### token 速率计算

16kHz 语音、50Hz 基础语音 token，每秒输出需要 50 个语音 token。Talker 必须输出 ≥50 tok/s 才能跟上。在 H100 上典型 LLM 吞吐量为 30-80 tok/s，小型（200-300M）Talker 足够快；7B Talker 会跟不上。

这就是为什么存在小型专用 Talker 模型，而非"直接用主模型"。

## 开始构建

`code/main.py`：

- 模拟带有模拟 token 发射速率的 Thinker-Talker 管道。
- 计算可配置模型大小和麦克风采样率下的 TTFAB。
- 演示带 VAD 静音阈值的半双工轮流对话。

## 交付成果

本课生成 `outputs/skill-omni-streaming-budget.md`。给定实时语音产品的目标 TTFAB 和功能集（视觉输入、双语、全双工），在 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 之间做出选择，并确定 Thinker/Talker 的规模。

## 练习

1. 你的目标 TTFAB 是 300ms。在 7B Thinker 和 300M Talker 上，写出每个组件的延迟。

2. Qwen2.5-Omni 使用 TMRoPE。描述当用户在 t=1s 开始说话、摄像头在 t=1.2s 捕获手势时，模型看到的内容。

3. 全双工支持需要模型在倾听的同时输出音频。提出一种训练数据格式来教授此能力。

4. 阅读 Moshi 论文第 4 节。描述"内心独白"分离以及为什么它避免了 Thinker-Talker 拆分。

5. 计算吞吐量预算：Talker 必须以多快的速度输出 token 才能跟上 16kHz 语音、50 基础层 token/秒？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| Thinker | "推理大脑" | 大型文本生成 Transformer，产生"说什么" |
| Talker | "说话嘴巴" | 小型 Transformer，从 Thinker 的文本产生离散语音 token |
| TTFAB | "延迟预算" | 首音频字节时间：从用户语音结束到首个音频样本输出 |
| TMRoPE | "时间对齐 RoPE" | 使用绝对时间戳跨视觉、音频、文本的位置编码 |
| 半双工 | "轮流对话" | 用户和模型交替发言；VAD 静音检测用户是否说完 |
| 全双工 | "同时对话" | 模型可以同时说话和倾听；具有反馈能力 |
| 内心独白 | "Moshi 分离" | 单模型设计，思考流和说话流交错 |

## 延伸阅读

- [Xu 等人 — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez 等人 — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng 等人 — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
