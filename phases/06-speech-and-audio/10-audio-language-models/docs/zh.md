# 音频语言模型 —— Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026年，音频语言模型将能够对语音、环境音及音乐进行综合推理。在MMAU-Pro评测中，Qwen2.5-Omni-7B的性能可与GPT-4o Audio相媲美；而在LongAudioBench测试中，Audio Flamingo Next则优于Gemini 2.5 Pro。开源模型与闭源模型之间的差距已基本消失——仅在多音频任务领域，各模型的表现仍接近随机水平。

**类型：** 学习
**语言：** Python
**先修知识：** 第6阶段·04（语音识别ASR）、第12阶段·03（视觉语言模型）、第7阶段·10（音频变换器）
**时长：** 约45分钟

## 问题所在

你有5秒钟的音频内容：狗叫声、有人喊“停下！”，随后是寂静。相关问题涉及多个维度：

- **文本转写。** “说了什么？”——属于ASR的范畴。
- **语义推理。** “那个人处于危险之中吗？”——需要综合理解狗叫声、喊声以及寂静部分。
- **音乐推理。** “哪些乐器演奏了这段旋律？”
- **长音频检索。** “在90分钟的讲座中，讲师是在哪里讲解梯度下降的？”

能够通过一个提示词同时回答所有这些问题的单一模型即为**音频语言模型**（LALM / ALM）。它与纯ASR不同：LALM能生成自由形式的自然语言答案，而不仅仅是转写文本。

## 概念概述

![音频语言模型：音频编码器 + 投影器 + 大语言模型解码器](../assets/alm-architecture.svg)

### 三组件模板

所有 2026 系列的 LALM 都具有相同的架构框架：

1. **音频编码器。** Whisper 编码器 · BEATs · CLAP · WavLM，或根据模型不同使用的自定义编码器。
2. **投影器。** 一种线性结构或 MLP 结构，用于将音频编码器的特征映射到 LLM 的令牌嵌入空间中。
3. **LLM。** 基于 Llama / Qwen / Gemma 的解码器。该解码器接收交错排列的文本与音频令牌，并生成文本。

训练流程：

- **第一阶段。** 冻结编码器和 LLM，仅在 ASR/字幕生成数据上对投影器进行训练。
- **第二阶段。** 在指令遵循类音频任务（如问答、推理、音乐理解）上进行全量微调或 LoRA 微调。
- **第三阶段（可选）。** 若需要实现语音输入/输出功能，则需添加语音解码器。Qwen2.5-Omni 和 AF3-Chat 即采用了此方案。

### 2026年模型地图

| 模型 | 核心框架 | 音频编码器 | 输出模态 | 许可协议 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | 自定义 + Whisper | 文本 + 语音 | Apache-2.0 |
| Qwen3-Omni | Qwen3 | 自定义 | 文本 + 语音 | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | 文本 | NVIDIA 非商业用途许可 |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | 文本 | NVIDIA 非商业用途许可 |
| SALMONN | Vicuna | Whisper + BEATs | 文本 | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | 文本 | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | 文本 | Apache-2.0 |
| Gemini 2.5 Flash/Pro（封闭版） | Gemini | 专有技术 | 文本 + 语音 | API 接口 |
| GPT-4o Audio（封闭版） | GPT-4o | 专有技术 | 文本 + 语音 | API 接口 |

### 现实基准测试（2026年）

**MMAU-Pro。** 包含1800对QA数据，涵盖语音、声音、音乐及混合类型音频，同时还包含多音频子集。

| 模型 | 总体得分 | 语音 | 声音 | 音乐 | 多音频 |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | 约60% | 73.4% | 51.9% | 64.9% | 约22% |
| Gemini 2.5 Flash | 约57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | 约20% |
| Audio Flamingo 3 | 约54% | — | — | — | — |
| Audio Flamingo Next | 在LongAudioBench测试中达到当前最佳水平 | — | — | — | — |

**多音频列的得分对所有模型而言都相当糟糕。** 在四选一的多项选择题中，随机作答的正确率仅为25%，大多数模型的得分也处于这一水平。大语言模型在对比两段音频片段方面依然存在困难。

### 2026年大语言模型可发挥作用的场景

- **呼叫中心录音的合规性审核。** “客服是否提到了必须披露的信息？”
- **无障碍功能。** 为听障用户描述声音事件（而不仅仅是提供文字转录）。
- **内容审查。** 检测暴力语言、威胁性语气以及背景上下文。
- **播客/会议章节划分。** 提供语义层面的总结，而不仅仅是按说话者轮流来划分。
- **音乐曲目目录分析。** “找出所有B段调性发生变化的曲目。”

### 它们尚未派上用场的地方

- 细粒度音乐理论（低于和弦层级）。  
- 长时间对话中的说话人归属推理（超过10分钟后性能下降）。  
- 多音频对比（22-26%的准确率仅略高于随机水平）。  
- 实时流式推理（目前大多为离线批量推理）。

## 构建它

### 步骤 1：查询 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步骤 2：投影器模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就是这样。投影器通常包含1到3个线性层。使用ASR配对数据（音频→文本转录）对其进行训练，即为第一阶段的预训练任务。

### 步骤 3：对 MMAU / LongAudioBench 进行基准测试

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

按类别（语音/声音/音乐/多音频）分别生成报告。汇总数据会掩盖模型出现故障的部分。

## 使用它

| 任务 | 2026年推荐模型 |
|------|---------------|
| 自由格式音频质检（开放式） | Qwen2.5-Omni-7B |
| 长时长音频的最佳开放式模型 | Audio Flamingo Next |
| 最佳封闭式模型 | Gemini 2.5 Pro |
| 语音输入/输出智能体 | Qwen2.5-Omni 或 GPT-4o Audio |
| 音乐推理 | Audio Flamingo 3 或 2（专用音乐模型 AF-CLAP） |
| 客服中心审核 | 通过 API 调用 Gemini 2.5 Pro，并结合政策文档的 RAG 技术进行处理 |

## 常见陷阱

- **过度依赖多音频处理。** 若任务需要判断“哪个片段包含X”，其性能实际上仅处于随机水平。
- **长音频质量下降。** 超过10分钟后，大多数模型的说话人识别功能会失效。建议先进行对话分割（第6课），然后再进行总结。
- **静音段的幻觉问题。** 使用Whisper编码器的LALM也会出现与Whisper相同的缺陷。需通过VAD机制加以解决。
- **基准测试数据的选择性展示。** 厂商的博客文章通常只展示最佳案例类别。建议自行使用MMAU-Pro的多音频子集进行测试。

## 发布它

将文件保存为 `outputs/skill-alm-picker.md`。针对特定的音频理解任务，选择 LALM、基准数据子集以及输出模态（文本或语音）。

## 练习题

1. **简单级。** 运行 `code/main.py` 即可查看一个简易的投影器模型结构，以及将（音频嵌入、文本令牌）→ 输出令牌的伪 LALM 路由过程。
2. **中等级。** 使用 Qwen2.5-Omni-7B 模型对 100 个 MMAU-Pro 语音数据集进行评分，并与论文中报告的结果进行对比。
3. **高级别。** 构建一个最简的音频字幕生成基准模型：包含 BEATs 编码器、2 层投影器以及冻结状态的 Llama-3.2-1B 模型。仅在 AudioCaps 数据集上对投影器部分进行微调，并与 Clotho-AQA 数据集上的 SALMONN 模型进行性能对比。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| LALM | Audio ChatGPT | 音频编码器 + 投影器 + 大语言模型解码器。 |
| Projector | 适配器 | 用于将音频特征映射到大语言模型嵌入空间的小型多层感知机。 |
| MMAU | 该基准测试 | 涵盖语音、声音、音乐领域的1万对音频问答数据对。 |
| MMAU-Pro | 更难的MMAU版本 | 包含1800道多音频/强推理要求的题目。 |
| LongAudioBench | 长文本评估基准 | 包含数分钟长片段及语义查询的测试集。 |
| Voice-in / voice-out | 原生语音处理 | 模型直接输入语音并输出语音，无需经过文本转换。 |

## 延伸阅读

- [Chu 等人 (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — 参考架构。
- [阿里巴巴 (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — 语音输入语音输出模型。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — 开源长音频处理领域的领先方案。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — 在 LongAudioBench 测试中达到当前最佳性能水平。
- [Tang 等人 (2023). SALMONN](https://arxiv.org/abs/2310.13289) — 双编码器架构的先驱。
- [MMAU-Pro 排名榜](https://mmaubenchmark.github.io/) — 2026 年实时排名。
