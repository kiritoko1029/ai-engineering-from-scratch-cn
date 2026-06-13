# 视频语言模型：时序 token 与时间定位

> 视频不是照片的堆叠。一段 5 秒的视频片段具有因果顺序、动作动词和事件时间信息，这些是图像模型无法表达的。Video-LLaMA（Zhang 等人，2023 年 6 月）发布了首个具有视听基础能力的开放视频 LLM。VideoChat 和 Video-LLaVA 扩展了这一模式。到 2025 年，Qwen2.5-VL 的 TMRoPE 缩小了与前沿闭源模型的差距。每个系统以不同方式解决了时序 token 问题——每个片段一个 Q-former、每帧拼接池化、每个 token 一个 TMRoPE。本课将解读这些模式，构建统一与动态帧采样器，并在时间定位任务上进行评估。

**类型：** 构建
**语言：** Python（标准库，帧采样器 + 时间定位评估器）
**前置要求：** 第 12 阶段 · 第 08 课（LLaVA-OneVision）
**所需时间：** 约180分钟

## 学习目标

- 解释为什么时序位置编码独立于视觉编码器影响视频 VLM 的性能。
- 比较统一、动态 FPS 和事件驱动帧采样在每秒 token 数与定位精度上的差异。
- 描述 Q-former-per-clip（Video-LLaMA）、pooled-per-frame（Video-LLaVA）和 M-RoPE-per-token（Qwen2.5-VL）的设计。
- 列出四个视频基准测试：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 问题所在

一段 1 分钟、30 FPS 的视频有 1800 帧。按每帧 196 个视觉 token（ViT-B，224 分辨率）计算，共 352k 个 token——超过 2024 年任何 LLM 的上下文窗口。

三种缩减策略：

1. 亚采样帧（根据内容 1-8 FPS）。
2. 对每帧的 patch token 进行激进池化（3x3 或 4x4 双线性池化）。
3. 通过 Q-former 压缩，将 16 帧片段映射为 64 个 token。

每种权衡各不相同。亚采样丢失时序细节。池化丢失空间细节。Q-former 两者都丢失一点但节省 token。

时序位置编码是另一个维度：模型如何知道第 5 帧在第 6 帧之前？选项包括简单的 1D 时序 RoPE（Video-LLaMA）、学习的时序嵌入（Video-LLaVA）和 TMRoPE（Qwen2.5-VL，完整的 3D）。

## 概念说明

### Video-LLaMA：每个片段一个 Q-former + 音频分支

Video-LLaMA（2023）是首个开放的视频 LLM。架构：

- 2 FPS 下的 16 帧片段（即 8 秒）。
- 每帧 ViT 特征 → 视频 Q-former 对所有 16 帧进行交叉注意力 → 32 个可学习查询 → LLM。
- 并行音频分支：波形 → ImageBind 音频编码器 → 音频 Q-former → 32 个查询 → LLM。

优势：视听联合推理。劣势：固定片段长度，无法任意时间定位。

### VideoChat 和 Video-LLaVA

VideoChat 保留了 Video-LLaMA 的思路但去掉了音频并做了简化。Video-LLaVA（Lin 等人，2023）在图像和视频帧上训练了单一视觉编码器（"投影前对齐"），提供统一表示。两者都是冻结 CLIP 编码器 + MLP + LLM。

两者都不处理长视频。都是 8-16 帧的系统。

### Qwen2.5-VL 与 TMRoPE

Qwen2.5-VL 引入了 TMRoPE——时序模态旋转位置编码。每个 patch token 携带一个 (t, h, w) 位置，其中 t 是实际时间戳（而非帧索引）。

与简单时序嵌入的关键区别：

- 绝对时间，而非索引。模型看到的是"在 4.2 秒处"而非"在第 15 帧"。
- 每个 token 独立旋转，而非每个片段。每个视觉 token 根据其时间戳独立旋转。
- 兼容动态 FPS。如果你在这里用 2 FPS 采样，在那里用 4 FPS 采样，TMRoPE 能原生处理不均匀间隔。

TMRoPE 使得"猫在第几秒跳起来？"这类查询成为可能。模型可以输出"在 4.2 秒处"。Video-LLaMA 只能说"在片段开头"。

### 帧采样策略

统一采样：在时长内均匀采样 N 帧。简单，但丢失运动峰值。

动态 FPS：根据运动强度自适应采样。光流或帧差分选出高运动片段进行更密集采样。Qwen2.5-VL 使用此方案训练。

事件驱动：运行轻量级检测器，在动作发生处采样更多帧。VideoAgent 使用此方案。

关键帧 + 上下文：在镜头边界处采样 + 少量相邻帧。用于影视内容。

### 每帧池化

1 FPS 下每帧 567 个 token，5 分钟片段为 172,800 个 token。Qwen2.5-VL-72B 的 128k 上下文可以容纳，但成本高昂。

3x3 双线性池化将每帧缩减到 64 个 token → 5 分钟为 19,200 个 token。对大多数任务来说是最佳平衡点。

更激进的池化（6x6 → 每帧 16 个 token）适用于空间细节不太重要的智能体工作流。

### 四个视频基准测试

- VideoMME：全面的视频理解，短 + 中 + 长时长。
- TempCompass：细粒度时序推理，"之前" / "之后"问题。
- EgoSchema：长时域第一人称视频。
- Video-MMMU：多模态多学科视频问题。

完整的视频 VLM 评估需要覆盖全部四个。它们侧重不同维度——TempCompass 全部关于顺序，EgoSchema 关于 3 分钟以上的推理，VideoMME 跨越不同时长。

### 定位输出格式

时间定位的输出格式：

- 自由文本："猫大约在第 4 秒跳起来。"易于解析但不精确。
- 结构化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 训练此格式。
- 基于 token：特殊的 `<time>4.1</time>` token 与答案交错。Qwen2.5-VL 的内部格式。

基于 token 的格式对下游使用最准确。Qwen2.5-VL 的 JSON 输出格式可以直接解析。

### 2026 年最佳实践

2026 年的视频 VLM：

- 编码器：SigLIP 2 配合 M-RoPE 或 TMRoPE（Qwen2.5-VL）。
- 帧采样：动态 FPS（根据运动 1-4），设有最大帧数限制。
- 每帧池化：3x3 双线性。
- 输出：包含时间和事件字段的结构化 JSON。
- 基准测试：通用场景用 VideoMME + TempCompass；长时域用 EgoSchema。

## 开始构建

`code/main.py` 包含：

- 统一和动态 FPS 帧采样器。
- 一个玩具时间定位评估器：给定时间 T 的"真实"事件和模型输出，以容差评分准确度。
- 跨 Video-LLaMA（16 帧，Q-former）、Video-LLaVA（8 帧，MLP）、Qwen2.5-VL（动态 FPS + TMRoPE）的比较。

## 交付成果

本课生成 `outputs/skill-video-vlm-frame-planner.md`。给定一个视频任务（监控、动作识别、时间定位、摘要），选择帧采样器、池化因子、输出格式和预期精度等级。

## 练习

1. 对于一段 3 分钟的烹饪演示视频，选择统一 vs 动态 FPS。用 token 数量进行论证。

2. TMRoPE 相比简单时序嵌入表具体增加了什么能力？

3. 为时间定位设计一个 VLM 可以学会输出的 JSON schema。包含错误情况。

4. 阅读 Video-LLaVA 第 3 节关于"投影前对齐"的内容。为什么这比分别训练图像和视频编码器更好？

5. 根据 VideoMME 排行榜，2026 年顶级开放模型与顶级闭源模型之间的差距有多大？其中多少差距可归因于时序编码 vs 基础 LLM 规模？

## 关键术语

| 术语 | 通俗说法 | 实际含义 |
|------|----------|----------|
| 时间定位 | "时间定位答案" | VLM 输出事件发生的具体时间戳范围 |
| TMRoPE | "时间多模态 RoPE" | 使用绝对时间戳的 3D 旋转位置编码，Qwen2.5-VL 使用 |
| 动态 FPS | "运动感知采样" | 在高运动片段采样更多帧，在静态片段采样更少 |
| 帧池化 | "每帧空间压缩" | 在 LLM 之前通过双线性插值减少每帧的 patch 数量 |
| 视频 Q-former | "片段压缩器" | 将 N 帧映射为 K 个可学习查询的交叉注意力瓶颈 |
| VideoMME | "视频基准测试" | 全面的短/中/长视频基准测试，2500+ 样本 |

## 延伸阅读

- [Zhang 等人 — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li 等人 — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin 等人 — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin 等人 — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
