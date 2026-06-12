# 音频分类——从基于MFCC的k-NN到AST与BEATs

> 从“狗叫声与警报声的区分”到“这是哪种语言”，都属于音频分类任务。其特征提取方法为 MEL 频谱。架构会随着技术发展而不断演进，但评估指标仍为 AUC、F1 分数以及各类别的召回率。

**类型：** 构建
**编程语言：** Python
**先修课程：** 第 6 阶段 · 02（频谱图与 MEL 频谱）、第 3 阶段 · 06（卷积神经网络）、第 5 阶段 · 08（用于文本处理的卷积神经网络与循环神经网络）
**所需时间：** 约 75 分钟

## 问题所在

你将获得一段10秒长的音频片段，需要判断其类别：城市环境音（警报声、钻头声、狗叫声）、语音指令（是/否/停止）、语言标识（en/es/ar）、说话者情绪（愤怒/中性）或环境音（室内/室外、嘈杂声）。所有这些都属于*音频分类*任务。到2026年，该领域的基准架构已经十分成熟：对数梅尔谱 → CNN或Transformer → softmax。

核心难点不在于网络结构，而在于数据。音频数据集存在严重的类别不平衡问题、明显的领域差异（纯净环境与嘈杂环境），以及标签噪声（究竟谁来界定“城市嘈杂声”和“餐厅噪音”？）。解决这类问题的80%关键在于数据的筛选、增强处理以及评估方法，而非将CNN替换为Transformer。

## 概念概述

![音频分类技术演进路线图：从基于MFCC的k-NN到AST再到BEATs](../assets/audio-classification.svg)

**基于MFCC的k-NN（1990年代的基准方法）。** 将每个音频片段的MFCC特征展平，计算其与已标注特征库的余弦相似度，返回Top K个结果中的多数投票值。在数据量较小且纯净的数据集上表现惊人（如Speech Commands、ESC-50），无需GPU即可运行。

**基于对数梅尔谱的2D CNN（2015-2019年）。** 将形状为`(T, n_mels)`的对数梅尔谱视为图像，应用ResNet-18或VGG风格的架构。通过对时间轴进行全局均值池化处理，再对各个类别应用Softmax函数。在2026年的大多数Kaggle竞赛中仍作为基准模型使用。

**音频频谱图变换器AST（2021-2024年）。** 将对数梅尔谱分割为固定大小的块（例如16×16的块），加入位置嵌入信息，随后输入到ViT架构中。在带有监督学习的AudioSet数据集上，其性能处于当前最先进水平，mAP值可达0.485。

**BEATs与WavLM-base（2024-2026年）。** 通过数百万小时的自我监督预训练进行模型训练，仅需使用原本所需监督数据的1-10%即可针对特定任务进行微调。在2026年，该模型已成为非语音音频任务的默认起点。BEATs-iter3在AudioSet数据集上的mAP值比AST高出1-2个百分点，且计算资源消耗仅为AST的1/4。

**将Whisper编码器作为固定骨干网络（2024年）。** 采用Whisper的编码器结构，去掉解码器部分，再连接一个线性分类器。在无需任何音频增强处理的情况下，该模型在语言识别及简单事件分类任务上的性能接近当前最先进水平，堪称“免费的高性能基准模型”。

### 类别不平衡才是真正的挑战

ESC-50：包含50个类别，每个类别有40个音频片段——数据分布均衡，难度较低。UrbanSound8K：包含10个类别，且数据分布为10:1的失衡状态。AudioSet：包含632个类别，长尾比例高达100,000:1。有效的处理技术包括：

- 训练阶段进行均衡采样（评估阶段则不采用）。
- Mixup：通过线性插值方式结合两个音频片段及其标签作为数据增强手段。
- SpecAugment：随机遮蔽时间域和频域的特定区域。该方法简单但至关重要。

### 评估

- 多类互斥任务（语音指令）：Top-1准确率、Top-5准确率。
- 多类多标签任务（AudioSet、UrbanSound风格）：平均精度均值（mAP）。
- 严重不平衡数据集：各类别召回率 + 宏F1值。

2026年需掌握的关键指标：

| 测试基准 | 基线模型 | 2026年最新最佳模型 | 数据来源 |
|-----------|----------|-------------------|--------|
| ESC-50 | 82%（AST） | 97.0%（BEATs-iter3） | BEATs论文（2024年） |
| AudioSet mAP | 0.485（AST） | 0.548（BEATs-iter3） | 2026年HEAR排行榜 |
| Speech Commands v2 | 98%（CNN） | 99.0%（Audio-MAE） | HEAR v2测试结果 |

## 构建它

### 步骤 1：特征工程

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步骤 2：固定长度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

简单而高效：通过计算时间序列上的均值与方差，即可为13个系数的MFCC生成一个26维的固定嵌入向量。该算法几乎可瞬间运行，并在2017年的ESC-50数据集测试中超越了当时最先进的神经网络基准模型。

### 步骤 3：k-近邻算法

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步骤 4：在 log-mel 数据上升级为 CNN 模型

在 PyTorch 中：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M 参数。在 ESC-50 数据集上，使用单块 RTX 4090 即可在约 10 分钟内完成训练，准确率可达 80% 以上。

### 步骤 5：2026 年的默认设置——对 BEATs 进行微调

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

对于 BEATs，可通过 `beats` 库使用 `microsoft/BEATs-base`；其 transformers API 的结构保持一致。

## 使用它

2026年技术栈：

| 场景 | 推荐起点 |
|-----------|-----------|
| 极小数据集（<1000条音频片段） | 基于MFCC均值值的k-NN算法（作为基准）+ 音频增强处理 |
| 中等数据集（1K–100K条） | BEATs或AST模型微调 |
| 大型数据集（>100K条） | 从零开始训练或对Whisper编码器进行微调 |
| 实时边缘部署 | 40-MFCC CNN，量化为int8格式（采用KWS风格） |
| 多标签任务（AudioSet） | BEATs-iter3模型结合BCE损失函数、mixup技术及SpecAugment增强方法 |
| 语言识别 | MMS-LID模型以及SpeechBrain VoxLingua107基准模型 |

决策规则：**应从已冻结的骨干网络开始，而非全新构建模型**。通过对BEATs头部进行微调，仅需数小时即可达到当前最先进水平的95%性能，而无需耗费数周时间。

## 发布它

将文件保存为 `outputs/skill-classifier-designer.md`。针对特定的音频分类任务，选择架构、数据增强方法、类别平衡策略以及评估指标。

## 练习题

1. **简单。** 运行 `code/main.py`。该脚本会在一个包含4个类别的合成数据集（不同音高的纯音）上训练k-NN MFCC基线模型，并输出混淆矩阵。
2. **中等难度。** 将 `summarize` 函数替换为 [mean, var, skew, kurtosis] 这四个统计量。在同一个合成数据集上，4阶矩聚合方法是否比 mean+var 方法表现更好？
3. **高难度。** 使用 `torchaudio` 库，在ESC-50数据集的第1折上训练一个2D CNN模型。需输出5折交叉验证的准确率。同时加入 SpecAugment 数据增强技术（时间掩码长度为20，频率掩码长度为10），并报告其带来的性能提升幅度。

## 关键术语

| 术语 | 常见说法 | 实际含义 |
|------|----------|----------|
| AudioSet | 音频领域的 ImageNet | Google 提供的包含 200 万段音频、632 个类别的弱标注 YouTube 数据集。 |
| ESC-50 | 小型分类基准测试 | 包含 50 个类别及每类 40 段环境声音的测试集。 |
| AST | 音频频谱图变换器 | 基于对数梅尔分块的 ViT 模型；2021 年的最先进技术。 |
| BEATs | 自监督音频模型 | 微软开发的模型，截至 2026 年 iter3 在 AudioSet 数据集上的表现优于该模型。 |
| Mixup | 配对增强技术 | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 基于掩码的增强技术 | 随机将频谱图中的时间和频率区域置零。 |
| mAP | 主要多标签评估指标 | 按类别和阈值计算的平均精度均值。 |

## 延伸阅读

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778) — 2021年至2024年间最著名的架构。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058) — 2024年及之后的默认选择。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779) — 最常用的音频增强技术。
- [Piczak (2015). ESC-50 数据集](https://github.com/karolpiczak/ESC-50) — 依然广泛使用的50类基准数据集。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/) — 包含632个类别的YouTube分类体系；至今仍是行业黄金标准。
