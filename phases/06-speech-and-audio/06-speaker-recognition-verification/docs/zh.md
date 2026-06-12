# 语音识别与验证

> ASR会询问“他们说了什么？”，而语音识别系统则会询问“是谁说的？”。其数学原理看似相同——即嵌入向量加上余弦相似度计算——但所有实际应用中的决策都取决于一个EER数值。

**类型：** 构建
**语言：** Python
**先修知识：** 第6阶段·02（频谱图与梅尔频率），第5阶段·22（嵌入模型）
**耗时：** 约45分钟

## 问题所在

用户输入一个口令。你需要判断：该口令是否属于用户自称的身份（*验证*，1:1关系），还是你注册数据库中的第一个用户身份（*识别*，1:N关系）？又或者两者皆非——即该用户为未知说话者（*开放集*）？

2018年之前：采用GMM-UBM与i向量技术。虽然等错误率表现尚可，但对通话环境变化（如手机与笔记本电脑差异）及情绪状态较为敏感。2018年至2022年：使用x向量技术，其核心TDNN模型通过角边界法进行训练。2022年之后：出现ECAPA-TDNN及WavLM-large嵌入模型。到2026年时，该领域已由三种模型和一种评估指标主导。

该评估指标即为**EER**——等错误率。需设定决策阈值，使得误接受率等于误拒绝率，两者的交点即为EER值。这一指标被广泛应用于所有学术论文、排行榜以及采购讨论中。

## 概念概述

![包含嵌入、余弦相似度及EER指标的注册与验证流程图](../assets/speaker-verification.svg)

**该流程如下。** 注册阶段：录制目标说话者的5–30秒语音；计算固定维度的嵌入向量（ECAPA-TDNN为192维，WavLM-large为256维）。验证阶段：获取测试语音的嵌入向量；计算余弦相似度；并与预设阈值进行比较。

**ECAPA-TDNN（2020年提出，2026年仍占据主导地位）。** 该模型侧重于通道注意力机制以及传播与聚合型时延神经网络。其结构包括带有挤压激励功能的一维卷积块、多头注意力池化层，随后通过线性层将特征映射为192维向量。该模型在VoxCeleb 1+2数据集（包含2,700名说话者及110万条语音记录）上使用加法角距损失函数（AAM-softmax）进行训练。

**WavLM-SV（2022年及以后推出）。** 在预训练的WavLM-large SSL模型骨干网络上通过AAM损失函数进行微调。该模型质量更高，但计算速度较慢——模型大小为300+ MB，而基础版本仅为15 MB。

**x-vector（基准模型）。** 采用时延神经网络结合统计池化技术。属于经典方案，在CPU及边缘设备上仍具有实用价值。

**AAM-softmax。** 在标准softmax函数的基础上，在角空间中加入边距`m`：正确类别的相似度计算公式为`cos(θ + m)`。该机制旨在增大不同类别之间的角距离。典型参数设置为`m=0.2`，缩放因子`s=30`。

### 评分

- 登录嵌入与测试嵌入之间的**余弦相似度**。基于阈值进行决策。
- **PLDA（概率LDA）**：将嵌入投影到潜在空间中，使得同一说话者与不同说话者的似然比具有封闭形式的表达式。该方法在余弦相似度基础上使用，可降低EER值10–20%。为2020年之前的标准方法；目前仅用于封闭集场景。
- **分数归一化**：`S-norm`或`AS-norm`：根据一组冒名顶替者的均值和标准差对每个分数进行归一化。这对于跨领域评估至关重要。

### 你需要掌握的数字（2026年）

| 模型 | VoxCeleb1-O EER | 参数量 | A100 上的吞吐量 |
|-------|-----------------|--------|-------------------|
| x-vector（经典版） | 3.10% | 5 百万 | 400× 实时速率 |
| ECAPA-TDNN | 0.87% | 15 百万 | 200× 实时速率 |
| WavLM-SV large | 0.42% | 3.16 亿 | 20× 实时速率 |
| Pyannote 3.1 分割 + 嵌入技术 | 0.65% | 6 百万 | 100× 实时速率 |
| ReDimNet（2024版） | 0.39% | 24 百万 | 100× 实时速率 |

### 语音分割

多说话人音频片段中的“谁在何时发言”。处理流程：语音活动检测（VAD）→ 分段 → 对每个片段进行嵌入 → 聚类（凝聚式或频谱聚类）→ 平滑边界。现代常用工具为 `pyannote.audio` 3.1，它通过一次调用即可完成说话人分割、嵌入及聚类操作。2026年针对AMI数据集的最新最佳分离率约为15%（低于2022年的23%）。

## 构建它

### 步骤 1：基于 MFCC 统计量的玩具嵌入生成

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

远非最先进的技术——仅用于教学目的。`code/main.py` 使用该示例在合成说话人数据上展示概念验证。

### 步骤 2：余弦相似度 + 阈值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步骤 3：基于相似度对计算EER值

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

返回 (eer, threshold_at_eer) 两个值。均需报告。

### 步骤 4：使用 SpeechBrain 进行生产环境部署

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 第 5 步：使用 pyannote 进行时间轴标注

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 使用它

2026年技术栈：

| 场景 | 推荐方案 |
|-----------|----------|
| 闭集式1:1验证，边缘端 | ECAPA-TDNN + 余弦阈值 |
| 开集式验证，云端 | WavLM-SV + AS-norm |
| 语音分段（会议、播客） | `pyannote/speaker-diarization-3.1` |
| 反欺诈（重放/深度伪造检测） | AASIST 或 RawNet2 |
| 极小型嵌入式系统（关键词识别+注册功能） | Titanet-Small (NeMo) |

## 常见陷阱

- **通道不匹配。** 在 VoxCeleb（网络视频）数据集上训练的模型与电话通话音频不兼容。务必在目标通道上进行评估。
- **语音片段过短。** 当测试音频长度低于 3 秒时，EER 指标会急剧下降。
- **含噪声的注册数据。** 单个含噪声的注册样本会影响基准模型的性能。建议使用至少 3 个干净样本并取平均值。
- **所有情况下阈值固定不变。** 应始终在目标领域中预留的测试集上调整阈值。
- **对未归一化的嵌入向量使用余弦相似度计算。** 首先需对其进行 L2 归一化，否则向量的幅度会主导相似度结果。

## 发布它

将文件保存为 `outputs/skill-speaker-verifier.md`。需选定模型、注册协议、阈值调整方案以及欺诈防范措施。

## 练习题

1. **简单级。** 运行 `code/main.py`。该脚本会生成合成“说话人”数据（具有不同音调特征），完成说话人注册，并在 100 对的测试列表上计算 EER 值。
2. **中等级。** 使用 SpeechBrain ECAPA 算法对 30 条 VoxCeleb1 发言数据进行评估（共 5 名说话人，每人 6 条发言）。分别使用余弦相似度与 PLDA 方法计算 EER 值。
3. **高级别。** 利用 `pyannote.audio` 构建完整的说话人注册 → 语音分段 → 验证处理流程，并在 AMI 开发数据集上评估 DER 值。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| EER | 核心指标 | 错误接受率等于错误拒绝率的阈值。 |
| 验证 | 1:1 | “这是不是爱丽丝？” |
| 识别 | 1:N | “说话的是谁？” |
| 开集 | 可能包含未知人员 | 测试集中可以有未注册的说话者。 |
| 注册 | 登记 | 计算说话者的参考嵌入向量。 |
| AAM-softmax | 损失函数 | 带加法角边距的Softmax；用于强制实现聚类分离。 |
| PLDA | 经典评分方法 | 概率LDA；在嵌入向量的基础上进行似然比评分。 |
| DER | 语音分段指标 | 语音分段错误率——包括漏分、误报和混淆情况。 |

## 延伸阅读

- [Snyder 等人（2018）。X-Vectors：用于说话人识别的鲁棒深度神经网络嵌入方法](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf) —— 经典的深度嵌入论文。
- [Desplanques 等人（2020）。ECAPA-TDNN](https://arxiv.org/abs/2005.07143) —— 2020–2026 年间的主流架构。
- [Chen 等人（2022）。WavLM：用于全栈语音处理的大规模自监督预训练模型](https://arxiv.org/abs/2110.13900) —— 用于说话人识别与对话分割的 SSL 主干网络。
- [Bredin 等人（2023）。pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) —— 用于实际应用中的对话分割及嵌入处理工具链。
- [VoxCeleb 排名榜（数据更新至 2026 年）](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/) —— 各模型的当前 EER 成绩排名。
