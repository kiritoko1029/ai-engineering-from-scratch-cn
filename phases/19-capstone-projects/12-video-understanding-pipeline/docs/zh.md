# 毕业项目 12 — 视频理解流水线（场景、问答、搜索）

> Twelve Labs 将 Marengo + Pegasus 产品化。VideoDB 提供了视频 CRUD API。AI2 的 Molmo 2 发布了开放 VLM 检查点。Gemini 长上下文原生处理数小时视频。TimeLens-100K 定义了大规模时序定位。2026 年的流水线已经确定：场景分割、每场景描述 + 嵌入、转录对齐、多向量索引，以及带（开始、结束）时间戳和帧预览的查询回答。毕业项目是摄入 100 小时视频，达到公开基准，并衡量计数和动作问题上的幻觉。

**类型：** 毕业项目
**语言：** Python（流水线）、TypeScript（UI）
**前置要求：** 阶段 4（CV）、阶段 6（语音）、阶段 7（Transformer）、阶段 11（LLM 工程）、阶段 12（多模态）、阶段 17（基础设施）
**涉及阶段：** P4 · P6 · P7 · P11 · P12 · P17
**所需时间：** 30 小时

## 问题所在

长视频问答是 2026 年规模下最耗带宽的多模态问题。Gemini 2.5 Pro 可以原生读取 2 小时视频，但将 100 小时视频摄入为可查询语料库仍然需要场景级索引。生产形态结合了场景分割（TransNetV2 或 PySceneDetect）、使用 VLM 的每场景描述（Gemini 2.5、Qwen3-VL-Max 或 Molmo 2）、转录对齐（带词级时间戳的 Whisper-v3-turbo），以及存储描述、帧嵌入和转录的多向量索引。查询流水线以（开始、结束）时间戳和帧预览回答。

基准公开（ActivityNet-QA、NeXT-GQA）加上你自己 100 个查询的自定义集。计数和动作类型问题上的幻觉是已知的困难失败类别；毕业项目明确衡量它。

## 概念说明

摄入时三条流水线并行运行。**场景分割**将视频切割为场景。**VLM 描述**为每个场景生成描述和从关键帧生成帧嵌入。**ASR 对齐**产生词级时间戳。三条流通过（scene_id，时间范围）连接。每个场景在多向量索引（Qdrant）中有三种向量类型：描述嵌入、关键帧嵌入、转录嵌入。

查询时，自然语言问题对三种向量触发；结果用 RRF 合并；时序定位适配器（TimeLens 风格）在 top 场景内细化（开始、结束）窗口。VLM 综合器（Gemini 2.5 Pro 或 Qwen3-VL-Max）接收查询 + top 场景 + 裁剪帧，以引用的时间戳和帧预览回答。

幻觉衡量很重要。计数（"有多少人进入房间？"）和动作类型（"厨师是先倒再搅拌吗？"）问题出了名地不可靠。与描述性问题分开报告准确率。

## 架构

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## 技术栈

- 场景分割：TransNetV2（2024-26 最先进）或 PySceneDetect
- ASR：通过 faster-whisper 的 Whisper-v3-turbo，带词级时间戳
- VLM 描述器 + 回答器：Gemini 2.5 Pro 或 Qwen3-VL-Max 或 Molmo 2
- 时序定位：TimeLens-100K 训练的适配器或 VideoITG
- 索引：Qdrant，支持多向量（描述 / 帧 / 转录）
- UI：Next.js 15，带 HTML5 视频播放器和场景缩略图
- 评估：ActivityNet-QA、NeXT-GQA、自定义 100 个问题的人工标注集
- 幻觉基准：计数和动作类型子集，带人工标注

## 开始构建

1. **摄入遍历器。** 接受 YouTube URL 或本地 MP4。需要时降采样到 720p。持久化 `{video_id, file_path}`。

2. **场景分割。** 运行 TransNetV2 或 PySceneDetect 生成 `[{scene_id, start_ms, end_ms, keyframe_path}]`。目标 100 小时：约 6k-8k 个场景。

3. **ASR 遍历。** 在音频上运行 Whisper-v3-turbo；导出词级时间戳；拆分为每场景转录切片。

4. **VLM 描述。** 每个场景，使用关键帧和简短描述模板调用 Gemini 2.5 Pro（或 Qwen3-VL-Max）。生成描述 + 帧嵌入。

5. **多向量索引。** Qdrant 集合，三个命名向量。负载：`{video_id, scene_id, start_ms, end_ms, keyframe_url}`。

6. **查询。** 自然语言问题触发三个稠密查询；用倒数排名融合合并；top-k=5 个场景。

7. **时序定位。** 在 top 场景上运行 TimeLens 风格适配器，细化场景内的（开始、结束）窗口。

8. **VLM 综合。** 使用查询 + top-3 场景片段（作为图像或短视频）+ 转录调用 Gemini 2.5 Pro。要求 `(video_id, start_ms, end_ms)` 引用。

9. **评估。** 运行 ActivityNet-QA 和 NeXT-GQA。构建 100 个查询的自定义集。报告整体准确率 + 每类别分解（计数、动作、描述）。

## 使用示例

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## 交付成果

`outputs/skill-video-qa.md` 是交付成果。给定 YouTube URL 或上传的视频，流水线索引场景并以带时间戳的引用回答问题。

| 权重 | 评估标准 | 衡量方式 |
|:-:|---|---|
| 25 | 时序定位 IoU | 在保留定位集上的交并比 |
| 20 | 问答准确率 | NeXT-GQA 和自定义 100 个查询 |
| 20 | 摄入吞吐量 | 每美元支出的视频小时数 |
| 20 | UI 和引用 UX | 时间戳链接、缩略图条、跳转到帧 |
| 15 | 幻觉率 | 计数和动作类型准确率单独报告 |
| **100** | | |

## 练习

1. 在描述遍历中将 Gemini 2.5 Pro 替换为 Qwen3-VL-Max。在人工评分的 50 个场景样本上报告描述质量差异。

2. 将每场景帧嵌入减少为单个池化向量而非多向量。衡量检索退化。

3. 构建"严格计数"模式：综合器提取每个计数实例及时间戳，用户点击验证。衡量用户验证是否减少幻觉。

4. 基准摄入成本：三种 VLM 选择下的每美元视频小时数。选择最优方案。

5. 添加说话人分离转录：在音频上运行 pyannote 说话人分离，嵌入每说话人转录。演示"Alice 说了什么关于 X？"的查询。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| 场景分割 | "镜头检测" | 在镜头边界处将视频切割为场景 |
| 多向量索引 | "描述 + 帧 + 转录" | Qdrant 集合，每种表示有命名向量 |
| 时序定位 | "它到底什么时候发生的" | 为查询答案细化（开始、结束）窗口 |
| 帧嵌入 | "视觉表示" | 关键帧的向量嵌入；用于场景视觉相似度 |
| RRF 融合 | "倒数排名融合" | 跨多个排序列表的合并策略；经典混合检索技巧 |
| 计数幻觉 | "计数错误" | VLM 在"有多少 X"问题上的已知失败模式 |
| ActivityNet-QA | "视频问答基准" | 长视频问答准确率基准 |

## 延伸阅读

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — 开放 VLM 检查点
- [TimeLens（CVPR 2026）](https://github.com/TencentARC/TimeLens) — 大规模时序定位
- [Gemini Video 长上下文](https://deepmind.google/technologies/gemini) — 托管参考
- [VideoDB](https://videodb.io) — 视频 CRUD API 参考
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — 商用参考
- [TransNetV2](https://github.com/soCzech/TransNetV2) — 场景分割模型
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — 经典开源替代
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — 参考评估基准
